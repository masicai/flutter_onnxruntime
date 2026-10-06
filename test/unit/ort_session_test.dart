// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
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

  // Track method calls for verification
  String? lastSessionIdForRun;
  Map<String, dynamic>? lastInputsForRun;
  Map<String, dynamic>? lastRunOptions;

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, OrtValue> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Track the invocation for verification
    lastSessionIdForRun = sessionId;
    lastInputsForRun = {
      for (final entry in inputs.entries) entry.key: {'valueId': entry.value.id},
    };
    lastRunOptions = runOptions;

    // Return mock output - simulate the new output format with OrtValue properties
    return Future.value({
      'output1': [
        'test_output_value_1',
        'float32',
        [1, 3],
      ],
      'output2': [
        'test_output_value_2',
        'float32',
        [2, 2],
      ],
    });
  }

  // Track session close calls
  String? lastClosedSessionId;

  @override
  Future<void> closeSession(String sessionId) {
    lastClosedSessionId = sessionId;
    return Future.value();
  }

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
  Future<Map<String, dynamic>> createOrtValue(String sourceType, dynamic data, List<int> shape) {
    return Future.value({'valueId': 'test_value_id', 'dataType': sourceType, 'shape': shape});
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) => Future.value({});

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) => Future.value({
    'data': [1.0, 2.0, 3.0, 4.0],
    'shape': [2, 2],
  });

  @override
  Future<void> releaseOrtValue(String valueId) => Future.value();

  @override
  Future<List<String>> getAvailableProviders() => Future.value(['CPU']);
}

class CustomDataMockFlutterOnnxruntimePlatform extends MockFlutterOnnxruntimePlatform {
  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) {
    if (valueId == 'test_output_value_1') {
      return Future.value({
        'data': [1.0, 2.0, 3.0],
        'shape': [1, 3],
      });
    } else if (valueId == 'test_output_value_2') {
      return Future.value({
        'data': [4.0, 5.0, 6.0, 7.0],
        'shape': [2, 2],
      });
    }
    return Future.value({
      'data': [0.0, 0.0],
      'shape': [1, 2],
    });
  }

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1'],
      'outputNames': ['output1', 'output2'],
    });
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, OrtValue> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    return Future.value({
      'output1': [
        'test_output_value_1',
        'float32',
        [1, 3],
      ],
      'output2': [
        'test_output_value_2',
        'float32',
        [2, 2],
      ],
    });
  }
}

// Add this class outside the test function, at the top level or before the test group
class SessionOptionsMock extends MockFlutterOnnxruntimePlatform {
  Map<String, dynamic>? capturedOptions;

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    // Capture the options
    capturedOptions = sessionOptions;
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1', 'input2'],
      'outputNames': ['output1', 'output2'],
    });
  }
}

class InferenceOutputsMock extends MockFlutterOnnxruntimePlatform {
  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Track the call
    lastInputsForRun = {};
    for (final entry in inputs.entries) {
      lastInputsForRun![entry.key] = {'valueId': (entry.value as OrtValue).id};
    }

    // Return mock outputs with proper structure
    return Future.value({
      'output1': [
        'test_output_value_1',
        'float32',
        [1, 3],
      ],
      'output2': [
        'test_output_value_2',
        'float32',
        [2, 2],
      ],
    });
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) {
    // This method simulates retrieving data for an OrtValue
    // It returns different data based on the valueId
    if (valueId.contains('output1')) {
      return Future.value({
        'data': [1.0, 2.0, 3.0],
        'shape': [1, 3],
      });
    } else if (valueId.contains('output2')) {
      return Future.value({
        'data': [4.0, 5.0, 6.0, 7.0],
        'shape': [2, 2],
      });
    }
    return Future.value({'data': [], 'shape': []});
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late OrtSession session;
  late MockFlutterOnnxruntimePlatform mockPlatform;
  final FlutterOnnxruntimePlatform initialPlatform = FlutterOnnxruntimePlatform.instance;

  setUp(() async {
    mockPlatform = MockFlutterOnnxruntimePlatform();
    FlutterOnnxruntimePlatform.instance = mockPlatform;

    // Create a session for testing
    final onnxRuntime = OnnxRuntime();
    session = await onnxRuntime.createSession('test_model.onnx');
  });

  tearDown(() {
    FlutterOnnxruntimePlatform.instance = initialPlatform;
  });

  group('OrtSession initialization', () {
    test('fromMap creates session with correct properties', () {
      final map = {
        'sessionId': 'custom_session_id',
        'inputNames': ['custom_input1', 'custom_input2'],
        'outputNames': ['custom_output1', 'custom_output2'],
      };

      final customSession = OrtSession.fromMap(map);

      expect(customSession.id, 'custom_session_id');
      expect(customSession.inputNames, ['custom_input1', 'custom_input2']);
      expect(customSession.outputNames, ['custom_output1', 'custom_output2']);
    });

    test('session created with correct initial values', () {
      expect(session.id, 'test_session_id');
      expect(session.inputNames, ['input1', 'input2']);
      expect(session.outputNames, ['output1', 'output2']);
    });
  });

  group('OrtSession run method', () {
    test('run passes correct parameters to platform', () async {
      // Create mock OrtValues
      final ortValue1 = OrtValue.fromMap({
        'valueId': 'test_value_1',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      final ortValue2 = OrtValue.fromMap({
        'valueId': 'test_value_2',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      final inputs = {'input1': ortValue1, 'input2': ortValue2};

      final runOptions = OrtRunOptions(logSeverityLevel: 1, logVerbosityLevel: 1, terminate: true);

      await session.run(inputs, options: runOptions);

      // Verify the correct parameters were passed to the platform implementation
      expect(mockPlatform.lastSessionIdForRun, 'test_session_id');
      expect(mockPlatform.lastInputsForRun!['input1'], {'valueId': 'test_value_1'});
      expect(mockPlatform.lastInputsForRun!['input2'], {'valueId': 'test_value_2'});
      expect(mockPlatform.lastRunOptions, runOptions.toMap());
    });

    test('run returns OrtValue outputs in new format', () async {
      // Create a mock OrtValue
      final ortValue = OrtValue.fromMap({
        'valueId': 'test_value_id',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      final outputs = await session.run({'input1': ortValue});

      expect(outputs, isNotNull);
      expect(outputs, isA<Map<String, OrtValue>>());
      expect(outputs.keys, containsAll(['output1', 'output2']));

      // Verify the output OrtValues have the correct properties
      expect(outputs['output1']!.id, 'test_output_value_1');
      expect(outputs['output1']!.dataType, OrtDataType.float32);
      expect(outputs['output1']!.shape, [1, 3]);

      expect(outputs['output2']!.id, 'test_output_value_2');
      expect(outputs['output2']!.dataType, OrtDataType.float32);
      expect(outputs['output2']!.shape, [2, 2]);
    });

    test('run processes OrtValue inputs correctly', () async {
      // Create a mock OrtValue
      final mockOrtValue = OrtValue.fromMap({
        'valueId': 'test_value_id',
        'dataType': 'float32',
        'shape': [2, 2],
      });

      // Use the OrtValue in session.run()
      await session.run({'input1': mockOrtValue});

      // Verify the processed inputs
      expect(mockPlatform.lastInputsForRun, isNotNull);
      expect(mockPlatform.lastInputsForRun!['input1'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastInputsForRun!['input1']['valueId'], 'test_value_id');
    });

    test('run with invalid inputs should be caught at compile time', () {
      // This test doesn't need assertions because it checks compile-time type safety
      // If we tried to pass Map<String, dynamic> with non-OrtValue objects,
      // it would fail at compile time with the new interface

      // Create a mock OrtValue
      final ortValue = OrtValue.fromMap({
        'valueId': 'test_value_id',
        'dataType': 'float32',
        'shape': [2, 2],
      });

      // This should work fine
      session.run({'input1': ortValue});

      // The following would not compile with our new interface:
      // session.run({'input1': [1.0, 2.0]}); // Compile error expected
    });

    test('run with multiple OrtValue inputs of different types', () async {
      // Create mock OrtValues with different data types
      final float32Value = OrtValue.fromMap({
        'valueId': 'float32_value',
        'dataType': 'float32',
        'shape': [2, 2],
      });

      final int32Value = OrtValue.fromMap({
        'valueId': 'int32_value',
        'dataType': 'int32',
        'shape': [3, 3],
      });

      await session.run({'input1': float32Value, 'input2': int32Value});

      // Verify the processed inputs
      expect(mockPlatform.lastInputsForRun, isNotNull);
      expect(mockPlatform.lastInputsForRun!['input1'], {'valueId': 'float32_value'});
      expect(mockPlatform.lastInputsForRun!['input2'], {'valueId': 'int32_value'});
    });

    test('run outputs can be used to extract data', () async {
      // Use a custom mock implementation specifically for this test
      final customMock = CustomDataMockFlutterOnnxruntimePlatform();
      FlutterOnnxruntimePlatform.instance = customMock;

      // Create a test session
      final onnxRuntime = OnnxRuntime();
      final testSession = await onnxRuntime.createSession('test_model.onnx');

      // Create a mock input OrtValue
      final inputValue = OrtValue.fromMap({
        'valueId': 'test_input_value',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      // Run inference
      final outputs = await testSession.run({'input1': inputValue});

      // Extract data from output OrtValues
      final output1Data = await outputs['output1']!.asList();
      final output2Data = await outputs['output2']!.asList();

      // Verify the extracted data - shape [1, 3] becomes a 2D array
      expect(output1Data, [
        [1.0, 2.0, 3.0],
      ]);
      expect(outputs['output1']!.shape, [1, 3]);

      // Verify the extracted data - shape [2, 2] becomes a 2D array
      expect(output2Data, [
        [4.0, 5.0],
        [6.0, 7.0],
      ]);
      expect(outputs['output2']!.shape, [2, 2]);

      // To demonstrate the difference, also use asFlattenedList()
      final output1FlatData = await outputs['output1']!.asFlattenedList();
      final output2FlatData = await outputs['output2']!.asFlattenedList();

      // Flattened data should be 1D arrays
      expect(output1FlatData, [1.0, 2.0, 3.0]);
      expect(output2FlatData, [4.0, 5.0, 6.0, 7.0]);

      // Restore the original mock
      FlutterOnnxruntimePlatform.instance = mockPlatform;
    });

    test('run method returns output OrtValues and can retrieve data', () async {
      // Replace with specialized mock
      final mockInferenceOutputs = InferenceOutputsMock();
      FlutterOnnxruntimePlatform.instance = mockInferenceOutputs;

      // Create a test session using fromMap constructor
      final testSession = OrtSession.fromMap({
        'sessionId': 'test_session_id',
        'inputNames': ['input1'],
        'outputNames': ['output1', 'output2'],
      });

      // Create a mock input OrtValue
      final inputValue = OrtValue.fromMap({
        'valueId': 'test_input_value',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      // Run inference
      final outputs = await testSession.run({'input1': inputValue});

      // Verify the processed inputs
      expect(mockInferenceOutputs.lastInputsForRun, isNotNull);
      expect(mockInferenceOutputs.lastInputsForRun!['input1'], {'valueId': 'test_input_value'});

      // Verify the output OrtValues have the correct properties
      expect(outputs['output1']!.id, 'test_output_value_1');
      expect(outputs['output1']!.dataType, OrtDataType.float32);
      expect(outputs['output1']!.shape, [1, 3]);

      expect(outputs['output2']!.id, 'test_output_value_2');
      expect(outputs['output2']!.dataType, OrtDataType.float32);
      expect(outputs['output2']!.shape, [2, 2]);
    });
  });

  group('OrtSession close method', () {
    test('close calls platform implementation with correct session ID', () async {
      await session.close();

      expect(mockPlatform.lastClosedSessionId, 'test_session_id');
    });
  });

  group('OrtSession metadata methods', () {
    test('getMetadata returns properly structured metadata', () async {
      final metadata = await session.getMetadata();

      expect(metadata, isA<OrtModelMetadata>());
      expect(metadata.producerName, 'ONNX Runtime Test');
      expect(metadata.graphName, 'TestModel');
      expect(metadata.domain, 'test.domain');
      expect(metadata.description, 'Test model for unit tests');
      expect(metadata.version, 1);
      expect(metadata.customMetadataMap, {'key1': 'value1', 'key2': 'value2'});
    });

    test('getInputInfo returns input tensor information', () async {
      final inputInfo = await session.getInputInfo();

      expect(inputInfo, isA<List<Map<String, dynamic>>>());
      expect(inputInfo.length, 2);
      expect(inputInfo[0]['name'], 'input1');
      expect(inputInfo[0]['type'], 'FLOAT');
      expect(inputInfo[0]['shape'], [1, 3, 224, 224]);
    });

    test('getOutputInfo returns output tensor information', () async {
      final outputInfo = await session.getOutputInfo();

      expect(outputInfo, isA<List<Map<String, dynamic>>>());
      expect(outputInfo.length, 2);
      expect(outputInfo[0]['name'], 'output1');
      expect(outputInfo[0]['type'], 'FLOAT');
      expect(outputInfo[0]['shape'], [1, 1000]);
    });
  });

  group('OrtSessionOptions', () {
    test('constructs with all options and produces correct map', () {
      final options = OrtSessionOptions(
        intraOpNumThreads: 2,
        interOpNumThreads: 4,
        providers: [OrtProvider.CUDA, OrtProvider.CPU],
        useArena: true,
        deviceId: 0,
      );

      final map = options.toMap();

      expect(map['intraOpNumThreads'], 2);
      expect(map['interOpNumThreads'], 4);
      expect(map['providers'], ['CUDA', 'CPU']);
      expect(map['useArena'], true);
      expect(map['deviceId'], 0);
    });

    test('constructs with partial options and produces correct map', () {
      final options = OrtSessionOptions(intraOpNumThreads: 2, providers: [OrtProvider.CUDA]);

      final map = options.toMap();

      expect(map['intraOpNumThreads'], 2);
      expect(map['providers'], ['CUDA']);
      expect(map.containsKey('interOpNumThreads'), false);
      expect(map.containsKey('useArena'), false);
      expect(map.containsKey('deviceId'), false);
    });

    test('empty providers list is not included in map', () {
      final options = OrtSessionOptions(intraOpNumThreads: 2, providers: []);

      final map = options.toMap();

      expect(map['intraOpNumThreads'], 2);
      expect(map.containsKey('providers'), false);
    });

    test('session configs are included in map only when non-empty', () {
      final options = OrtSessionOptions(sessionConfigs: {'mlas.disable_kleidiai': '1'});
      expect(options.toMap()['sessionConfigs'], {'mlas.disable_kleidiai': '1'});

      expect(OrtSessionOptions(sessionConfigs: {}).toMap().containsKey('sessionConfigs'), false);
      expect(OrtSessionOptions().toMap().containsKey('sessionConfigs'), false);
    });

    test('options affect session creation', () async {
      // Create a specialized tracking mock platform
      final optionsMock = SessionOptionsMock();
      FlutterOnnxruntimePlatform.instance = optionsMock;

      final options = OrtSessionOptions(providers: [OrtProvider.CUDA, OrtProvider.CPU], deviceId: 1);

      final onnxRuntime = OnnxRuntime();
      await onnxRuntime.createSession('test_model.onnx', options: options);

      expect(optionsMock.capturedOptions, isNotNull);
      expect(optionsMock.capturedOptions!['providers'], ['CUDA', 'CPU']);
      expect(optionsMock.capturedOptions!['deviceId'], 1);

      // Restore the original mock
      FlutterOnnxruntimePlatform.instance = mockPlatform;
    });
  });
}
