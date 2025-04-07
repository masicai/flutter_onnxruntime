// Integration test for ONNX Runtime functionality
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'dart:io';
import 'package:path/path.dart' as path;

void main() {
  late OnnxRuntime onnxRuntime;
  late OrtSession session;
  final String modelPath = path.join(Directory.current.path, 'example', 'assets', 'models', 'addition_model.ort');

  setUp(() async {
    onnxRuntime = OnnxRuntime();
    session = await onnxRuntime.createSession(modelPath);
  });

  tearDown(() async {
    await session.close();
  });

  test('Get platform version', () async {
    final version = await onnxRuntime.getPlatformVersion();
    expect(version, isNotNull);
    expect(version!.isNotEmpty, true);
  });

  test('Get model metadata', () async {
    final metadata = await session.getMetadata();
    expect(metadata, isNotNull);
    // The addition model may have limited metadata, but we can check it exists
    expect(metadata.producerName, isNotNull);
  });

  test('Get input/output info', () async {
    final inputInfo = await session.getInputInfo();
    expect(inputInfo, isNotNull);
    expect(inputInfo.length, 2); // Addition model has inputs A and B

    // Verify the input names
    expect(inputInfo.map((i) => i['name']), containsAll(['A', 'B']));

    final outputInfo = await session.getOutputInfo();
    expect(outputInfo, isNotNull);
    expect(outputInfo.length, 1); // Addition model has single output C
    expect(outputInfo[0]['name'], 'C');
  });

  test('Add two numbers', () async {
    final inputs = {
      'A': [3.0],
      'B': [4.0],
    };

    final outputs = await session.run(inputs);
    expect(outputs, isNotNull);
    expect(outputs['outputs']['C'][0], 7.0); // 3 + 4 = 7
  });

  test('Add two arrays of numbers', () async {
    final inputs = {
      'A': [1.0, 2.0, 3.0],
      'B': [4.0, 5.0, 6.0],
    };

    final outputs = await session.run(inputs);
    expect(outputs, isNotNull);

    final result = outputs['outputs']['C'];
    expect(result.length, 3);
    expect(result[0], 5.0); // 1 + 4 = 5
    expect(result[1], 7.0); // 2 + 5 = 7
    expect(result[2], 9.0); // 3 + 6 = 9
  });
}
