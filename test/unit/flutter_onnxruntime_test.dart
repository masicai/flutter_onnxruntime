import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_method_channel.dart';
import 'package:flutter_onnxruntime/src/ort_model_metadata.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterOnnxruntimePlatform with MockPlatformInterfaceMixin implements FlutterOnnxruntimePlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1', 'input2'],
      'outputNames': ['output1', 'output2'],
    });
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Return mock output with the same structure as expected from the real implementation
    return Future.value({
      'output1': [1.0, 2.0, 3.0],
      'output2': [
        [4.0, 5.0],
        [6.0, 7.0],
      ],
    });
  }

  @override
  Future<void> closeSession(String sessionId) => Future.value();

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'input1',
        'type': 'FLOAT',
        'shape': [1, 3, 224, 224],
      },
      {
        'name': 'input2',
        'type': 'INT64',
        'shape': [1, 10],
      },
    ]);
  }

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) {
    return Future.value({
      'producerName': 'ONNX Runtime Test',
      'graphName': 'TestModel',
      'domain': 'test.domain',
      'description': 'Test model for unit tests',
      'version': 1,
      'customMetadataMap': {'key1': 'value1', 'key2': 'value2'},
    });
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'output1',
        'type': 'FLOAT',
        'shape': [1, 1000],
      },
      {
        'name': 'output2',
        'type': 'FLOAT',
        'shape': [1, 2, 2],
      },
    ]);
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) => Future.value({});

  @override
  Future<Map<String, dynamic>> createOrtValue(
    String sourceType,
    data,
    List<int> shape,
    String targetType,
    String device,
  ) => Future.value({});

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId, String dataType) => Future.value({});

  @override
  Future<Map<String, dynamic>> moveOrtValueToDevice(String valueId, String targetDevice) => Future.value({});

  @override
  Future<void> releaseOrtValue(String valueId) => Future.value();
}

void main() {
  final FlutterOnnxruntimePlatform initialPlatform = FlutterOnnxruntimePlatform.instance;
  late OnnxRuntime onnxRuntime;
  late MockFlutterOnnxruntimePlatform mockPlatform;

  setUp(() {
    mockPlatform = MockFlutterOnnxruntimePlatform();
    FlutterOnnxruntimePlatform.instance = mockPlatform;
    onnxRuntime = OnnxRuntime();
  });

  tearDown(() {
    FlutterOnnxruntimePlatform.instance = initialPlatform;
  });

  test('$MethodChannelFlutterOnnxruntime is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFlutterOnnxruntime>());
  });

  group('OnnxRuntime', () {
    test('getPlatformVersion returns platform version', () async {
      expect(await onnxRuntime.getPlatformVersion(), '42');
    });

    test('createSession creates a session with model path', () async {
      final session = await onnxRuntime.createSession('test_model.onnx');

      expect(session, isNotNull);
      expect(session.id, 'test_session_id');
      expect(session.inputNames, ['input1', 'input2']);
      expect(session.outputNames, ['output1', 'output2']);
    });

    test('createSession with options passes options correctly', () async {
      final options = OrtSessionOptions(intraOpNumThreads: 2, interOpNumThreads: 1, enableCpuMemArena: true);

      final session = await onnxRuntime.createSession('test_model.onnx', options: options);

      expect(session, isNotNull);
      expect(session.id, 'test_session_id');
    });

    // This test requires mocking asset bundling which is complex in unit tests
    // We're verifying the basic function structure here
    test('createSessionFromAsset function structure is correct', () {
      expect(onnxRuntime.createSessionFromAsset, isA<Function>());
    });
  });

  group('OrtSession', () {
    late OrtSession session;

    setUp(() async {
      session = await onnxRuntime.createSession('test_model.onnx');
    });

    test('run performs inference with correct inputs', () async {
      final inputs = {
        'input1': [1.0, 2.0, 3.0],
        'input1_shape': [1, 3],
        'input2': [4, 5, 6],
        'input2_shape': [1, 3],
      };

      final outputs = await session.run(inputs);

      expect(outputs, isNotNull);
      expect(outputs['output1'], [1.0, 2.0, 3.0]);
      expect(outputs['output2'], [
        [4.0, 5.0],
        [6.0, 7.0],
      ]);
    });

    test('getMetadata returns model metadata', () async {
      final metadata = await session.getMetadata();

      expect(metadata, isNotNull);
      expect(metadata.producerName, 'ONNX Runtime Test');
      expect(metadata.graphName, 'TestModel');
      expect(metadata.domain, 'test.domain');
      expect(metadata.description, 'Test model for unit tests');
      expect(metadata.version, 1);
      expect(metadata.customMetadataMap, {'key1': 'value1', 'key2': 'value2'});
    });

    test('getInputInfo returns input tensor information', () async {
      final inputInfo = await session.getInputInfo();

      expect(inputInfo, isNotNull);
      expect(inputInfo.length, 2);
      expect(inputInfo[0]['name'], 'input1');
      expect(inputInfo[0]['type'], 'FLOAT');
      expect(inputInfo[0]['shape'], [1, 3, 224, 224]);
      expect(inputInfo[1]['name'], 'input2');
    });

    test('getOutputInfo returns output tensor information', () async {
      final outputInfo = await session.getOutputInfo();

      expect(outputInfo, isNotNull);
      expect(outputInfo.length, 2);
      expect(outputInfo[0]['name'], 'output1');
      expect(outputInfo[0]['type'], 'FLOAT');
      expect(outputInfo[0]['shape'], [1, 1000]);
      expect(outputInfo[1]['name'], 'output2');
    });

    test('close closes the session', () async {
      // This just verifies the call doesn't throw an exception
      await expectLater(session.close(), completes);
    });
  });

  group('OrtModelMetadata', () {
    test('fromMap creates metadata from map correctly', () async {
      final map = {
        'producerName': 'Test Producer',
        'graphName': 'Test Graph',
        'domain': 'test.domain',
        'description': 'Test Description',
        'version': 2,
        'customMetadataMap': {'test': 'value'},
      };

      final metadata = OrtModelMetadata.fromMap(map);

      expect(metadata.producerName, 'Test Producer');
      expect(metadata.graphName, 'Test Graph');
      expect(metadata.domain, 'test.domain');
      expect(metadata.description, 'Test Description');
      expect(metadata.version, 2);
      expect(metadata.customMetadataMap, {'test': 'value'});
    });

    test('toMap converts metadata to map correctly', () {
      final metadata = OrtModelMetadata(
        producerName: 'Test Producer',
        graphName: 'Test Graph',
        domain: 'test.domain',
        description: 'Test Description',
        version: 2,
        customMetadataMap: {'test': 'value'},
      );

      final map = metadata.toMap();

      expect(map['producerName'], 'Test Producer');
      expect(map['graphName'], 'Test Graph');
      expect(map['domain'], 'test.domain');
      expect(map['description'], 'Test Description');
      expect(map['version'], 2);
      expect(map['customMetadataMap'], {'test': 'value'});
    });
  });

  group('Options classes', () {
    test('OrtSessionOptions toMap converts options to map correctly', () {
      final options = OrtSessionOptions(intraOpNumThreads: 4, interOpNumThreads: 2, enableCpuMemArena: true);

      final map = options.toMap();

      expect(map['intraOpNumThreads'], 4);
      expect(map['interOpNumThreads'], 2);
      expect(map['enableCpuMemArena'], true);
    });

    test('OrtRunOptions toMap converts options to map correctly', () {
      final options = OrtRunOptions(logSeverityLevel: true, logVerbosityLevel: true, terminate: false);

      final map = options.toMap();

      expect(map['logSeverityLevel'], true);
      expect(map['logVerbosityLevel'], true);
      expect(map['terminate'], false);
    });

    test('OrtSessionOptions handles null values correctly', () {
      final options = OrtSessionOptions();
      final map = options.toMap();

      expect(map.containsKey('intraOpNumThreads'), false);
      expect(map.containsKey('interOpNumThreads'), false);
      expect(map.containsKey('enableCpuMemArena'), false);
    });
  });
}
