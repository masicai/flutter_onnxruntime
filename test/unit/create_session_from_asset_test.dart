// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

/// Points [getTemporaryDirectory] at a real, test-controlled directory.
class _FakePathProviderPlatform extends PathProviderPlatform with MockPlatformInterfaceMixin {
  _FakePathProviderPlatform(this.tempPath);

  final String tempPath;

  @override
  Future<String?> getTemporaryPath() async => tempPath;
}

/// Simulates native ONNX Runtime: it reads the model file passed to
/// [createSession] and "fails to parse" (throws, mirroring
/// `ORT_INVALID_PROTOBUF`) whenever the file bytes do not exactly match the
/// bundled asset — i.e. when a truncated/corrupt cached model is reused.
class _FileValidatingPlatform extends FlutterOnnxruntimePlatform with MockPlatformInterfaceMixin {
  _FileValidatingPlatform(this.expectedBytes);

  final Uint8List expectedBytes;

  int createSessionCallCount = 0;
  final List<int> observedFileSizes = [];

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) async {
    createSessionCallCount++;
    final bytes = await File(modelPath).readAsBytes();
    observedFileSizes.add(bytes.length);
    if (!_bytesEqual(bytes, expectedBytes)) {
      throw PlatformException(
        code: 'ORT_ERROR',
        message:
            'Error code - ORT_INVALID_PROTOBUF - message: '
            'Load model from $modelPath failed:Protobuf parsing failed.',
      );
    }
    return {
      'sessionId': 'session_ok',
      'inputNames': ['input'],
      'outputNames': ['output'],
    };
  }

  static bool _bytesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

void main() {
  final binding = TestWidgetsFlutterBinding.ensureInitialized();

  // A stand-in "model asset": deterministic, non-trivial bytes.
  final assetBytes = Uint8List.fromList(List<int>.generate(4096, (i) => (i * 7) % 256));
  const assetKey = 'assets/models/separation_model.onnx';
  const fileName = 'separation_model.onnx';

  final initialOrtPlatform = FlutterOnnxruntimePlatform.instance;
  final initialPathPlatform = PathProviderPlatform.instance;

  late Directory tempDir;
  late _FileValidatingPlatform fakeOrt;

  String cachedPath() => '${tempDir.path}${Platform.pathSeparator}$fileName';

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('ort_asset_test');

    PathProviderPlatform.instance = _FakePathProviderPlatform(tempDir.path);

    fakeOrt = _FileValidatingPlatform(assetBytes);
    FlutterOnnxruntimePlatform.instance = fakeOrt;

    // Intercept rootBundle asset loads for our asset key.
    rootBundle.clear();
    binding.defaultBinaryMessenger.setMockMessageHandler('flutter/assets', (ByteData? message) async {
      final key = utf8.decode(message!.buffer.asUint8List());
      if (key == assetKey) {
        return ByteData.view(assetBytes.buffer);
      }
      return null;
    });
  });

  tearDown(() async {
    binding.defaultBinaryMessenger.setMockMessageHandler('flutter/assets', null);
    FlutterOnnxruntimePlatform.instance = initialOrtPlatform;
    PathProviderPlatform.instance = initialPathPlatform;
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  group('createSessionFromAsset', () {
    test('extracts the asset and creates a session on first run', () async {
      final session = await OnnxRuntime().createSessionFromAsset(assetKey);

      expect(session.id, 'session_ok');
      final cached = File(cachedPath());
      expect(await cached.exists(), isTrue);
      expect(await cached.length(), assetBytes.length);
    });

    test('reuses an intact cached model without re-extracting', () async {
      // A valid, complete model already sits in the cache.
      await File(cachedPath()).writeAsBytes(assetBytes);

      final session = await OnnxRuntime().createSessionFromAsset(assetKey);

      expect(session.id, 'session_ok');
      // Only one createSession attempt: the intact cache was used directly.
      expect(fakeOrt.createSessionCallCount, 1);
    });

    test('re-extracts and recovers when the cached model is truncated/corrupt', () async {
      // Simulate a previous interrupted extraction (app killed, low storage,
      // cache eviction): a partial file already sits at the cache path. The old
      // implementation trusted its mere existence and reused it forever, which
      // surfaced to users as ORT_INVALID_PROTOBUF / "Protobuf parsing failed".
      await File(cachedPath()).writeAsBytes(assetBytes.sublist(0, 128));

      final session = await OnnxRuntime().createSessionFromAsset(assetKey);

      // It must self-heal: re-extract the intact asset and return a session.
      expect(session.id, 'session_ok');
      expect(await File(cachedPath()).length(), assetBytes.length);
      // First attempt on the corrupt cache failed, retry after re-extraction ran.
      expect(fakeOrt.createSessionCallCount, 2);
    });

    test('does not re-extract or retry when a freshly extracted model fails to load', () async {
      // Nothing is cached, and the runtime rejects the model regardless of its
      // bytes (e.g. an unsupported op or bad session options). The bytes were
      // just extracted straight from the bundle, so re-extracting the identical
      // data cannot help — it must fail fast without a redundant second attempt.
      final alwaysFail = _FileValidatingPlatform(Uint8List.fromList([0, 1, 2, 3]));
      FlutterOnnxruntimePlatform.instance = alwaysFail;

      await expectLater(OnnxRuntime().createSessionFromAsset(assetKey), throwsA(isA<PlatformException>()));
      // Exactly one attempt: no wasteful re-extract + retry on a fresh extraction.
      expect(alwaysFail.createSessionCallCount, 1);
    });

    test('re-extracts when the cached model is empty (zero-byte)', () async {
      await File(cachedPath()).writeAsBytes(Uint8List(0));

      final session = await OnnxRuntime().createSessionFromAsset(assetKey);

      expect(session.id, 'session_ok');
      expect(await File(cachedPath()).length(), assetBytes.length);
    });

    test('does not leave a partial temp file after extraction (atomic write)', () async {
      await OnnxRuntime().createSessionFromAsset(assetKey);

      final leftoverTmp = File('${cachedPath()}.tmp');
      expect(await leftoverTmp.exists(), isFalse);
    });
  });
}
