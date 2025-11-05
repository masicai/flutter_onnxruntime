// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Single file for all integration tests
//
// All integration tests are grouped into a single file due to an issue in Linux and macOS reported at:
// https://github.com/flutter/flutter/issues/135673
// Three models used in this Test: Addition model, Transpose-Avg Model and String Concat Model
//
// 1. The Addition Model is a simple model which perform Add operation between two tensor A and B, results in C
//
// 2. The Transpose-Avg model operation is defined as follows:
// def forward(self, A, B):
//     # Transpose tensor B (from [batch,n,m] to [batch,m,n])
//     B_transposed = B.transpose(1, 2)
//     # Add transposed B to A
//     summed = A + B_transposed
//     # Multiply element-wise with fixed tensor
//     C = summed * 0.5
//     return C
//
// The model has two inputs: A and B
// A is a tensor with shape [-1, 2, 3]
// B is a tensor with shape [-1, 3, 2]
// The model has one output: C
// C is a tensor with shape [-1, 2, 3]
//
// The model has three versions:
// * FP32: input and output are float32
// * INT32: input and output are int32
// * FP16: model is fp16
//
// 3. The String Concat model is a similar version of Addition model but with string inputs and outputs
//

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

// Set to true to enable benchmark timing prints during test runs
const bool _enableBenchmarkPrints = false;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('OrtValue Tests', () {
    group('Round-trip tests', () {
      testWidgets('Float32 round-trip test', (WidgetTester tester) async {
        final inputData = Float32List.fromList([1.1, 2.2, 3.3, 4.4]);
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], closeTo(inputData[i], 1e-5));
        }

        await tensor.dispose();
      });

      testWidgets('Int32 round-trip test', (WidgetTester tester) async {
        final inputData = Int32List.fromList([1, 2, 3, 4, 5, 6]);
        final shape = [2, 3]; // 2x3 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 6);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
      });

      testWidgets('Int64 round-trip test', (WidgetTester tester) async {
        // Skip the test for web platform as BigInt64Array required by ONNX Runtime Web for int64 tensors
        // is not supported in all browsers
        if (kIsWeb) {
          return;
        }
        // Use numbers outside the range of Int32 (e.g., greater than 2^31 - 1)
        final inputData = Int64List.fromList([2147483648, -2147483649, 9223372036, -9223372036]);
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int64);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }
        await tensor.dispose();
      });

      testWidgets('Uint8 round-trip test', (WidgetTester tester) async {
        final inputData = Uint8List.fromList([10, 20, 30, 40]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.uint8);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
      });

      testWidgets('Boolean round-trip test', (WidgetTester tester) async {
        final inputData = [true, false, true, false];
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.bool);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          final retrievedBoolValue = retrievedData[i] == 1;
          expect(retrievedBoolValue, inputData[i]);
        }

        await tensor.dispose();
      });

      testWidgets('String round-trip test', (WidgetTester tester) async {
        final inputData = ['Hello', 'World'];
        final shape = [2];

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.string);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 2);
        expect(retrievedData[0], 'Hello');
        expect(retrievedData[1], 'World');
      });

      testWidgets('Round-trip test with regular list of float32', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4];
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], closeTo(inputData[i], 1e-5));
        }

        await tensor.dispose();
      });

      testWidgets('Round-trip test with regular list of int32', (WidgetTester tester) async {
        // List<int> is not detected as interger in web platform, we have to
        // use typed list to specify it explicitly
        final inputData = kIsWeb ? Int32List.fromList([1, 2, 3, 4]) : [1, 2, 3, 4];
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asFlattenedList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
      });
    });

    group('Round-trip benchmark tests', () {
      testWidgets('Float32 benchmark test', (WidgetTester tester) async {
        final shape = [512, 512, 3];
        final totalElements = shape.fold(1, (a, b) => a * b); // 786,432 elements

        // Create input data with a pattern
        final inputData = Float32List(totalElements);
        for (int i = 0; i < totalElements; i++) {
          inputData[i] = (i % 100) / 10.0; // Values from 0.0 to 9.9
        }

        await _runBenchmarkTest(
          inputData: inputData,
          shape: shape,
          dataTypeName: 'Float32',
          bytesPerElement: 4,
          expectedType: OrtDataType.float32,
          validatePerformance: true,
        );
      });

      testWidgets('Int32 benchmark test', (WidgetTester tester) async {
        final shape = [512, 512, 3];
        final totalElements = shape.fold(1, (a, b) => a * b); // 786,432 elements

        // Create input data with a pattern
        final inputData = Int32List(totalElements);
        for (int i = 0; i < totalElements; i++) {
          inputData[i] = i % 1000; // Values from 0 to 999
        }

        await _runBenchmarkTest(
          inputData: inputData,
          shape: shape,
          dataTypeName: 'Int32',
          bytesPerElement: 4,
          expectedType: OrtDataType.int32,
          validatePerformance: true,
        );
      });

      testWidgets('Int64 benchmark test', (WidgetTester tester) async {
        // Skip for web platform as BigInt64Array is not supported in all browsers
        if (kIsWeb) {
          return;
        }

        final shape = [512, 512, 3];
        final totalElements = shape.fold(1, (a, b) => a * b); // 786,432 elements

        // Create input data with a pattern
        final inputData = Int64List(totalElements);
        for (int i = 0; i < totalElements; i++) {
          inputData[i] = i;
        }

        await _runBenchmarkTest(
          inputData: inputData,
          shape: shape,
          dataTypeName: 'Int64',
          bytesPerElement: 8,
          expectedType: OrtDataType.int64,
          validatePerformance: true,
        );
      });

      testWidgets('Uint8 benchmark test', (WidgetTester tester) async {
        final shape = [512, 512, 3];
        final totalElements = shape.fold(1, (a, b) => a * b); // 786,432 elements

        // Create input data with a pattern
        final inputData = Uint8List(totalElements);
        for (int i = 0; i < totalElements; i++) {
          inputData[i] = i % 256; // Values from 0 to 255
        }

        await _runBenchmarkTest(
          inputData: inputData,
          shape: shape,
          dataTypeName: 'Uint8',
          bytesPerElement: 1,
          expectedType: OrtDataType.uint8,
          validatePerformance: true,
        );
      });

      testWidgets('Boolean benchmark test', (WidgetTester tester) async {
        final shape = [512, 512, 3];
        final totalElements = shape.fold(1, (a, b) => a * b); // 786,432 elements

        // Create input data with a pattern
        final inputData = List<bool>.generate(totalElements, (i) => i % 2 == 0);

        await _runBenchmarkTest(
          inputData: inputData,
          shape: shape,
          dataTypeName: 'Boolean',
          bytesPerElement: 1,
          expectedType: OrtDataType.bool,
          validatePerformance: false,
        );
      });

      testWidgets('String benchmark test', (WidgetTester tester) async {
        final shape = [512, 512, 3];
        final totalElements = shape.fold(1, (a, b) => a * b); // 786,432 elements

        // Create input data with a pattern
        final inputData = List<String>.generate(totalElements, (i) => 'str_$i');

        await _runBenchmarkTest(
          inputData: inputData,
          shape: shape,
          dataTypeName: 'String',
          bytesPerElement: 10,
          expectedType: OrtDataType.string,
          isMemoryEstimated: true,
          validatePerformance: false,
        );
      });
    });

    group('Type conversion tests', () {
      testWidgets('Float32 to Int32 conversion', (WidgetTester tester) async {
        final inputData = Float32List.fromList([1.1, 2.2, 3.3, 4.4]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);

        final convertedTensor = await tensor.to(OrtDataType.int32);
        expect(convertedTensor.dataType, OrtDataType.int32);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        expect(retrievedData[0], 1);
        expect(retrievedData[1], 2);
        expect(retrievedData[2], 3);
        expect(retrievedData[3], 4);

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Int32 to Float32 conversion', (WidgetTester tester) async {
        final inputData = Int32List.fromList([1, 2, 3, 4]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);

        final convertedTensor = await tensor.to(OrtDataType.float32);
        expect(convertedTensor.dataType, OrtDataType.float32);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        expect(retrievedData[0], closeTo(1.0, 1e-5));
        expect(retrievedData[1], closeTo(2.0, 1e-5));
        expect(retrievedData[2], closeTo(3.0, 1e-5));
        expect(retrievedData[3], closeTo(4.0, 1e-5));

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Int32 to Int64 conversion', (WidgetTester tester) async {
        // skip the test for web platform as BigInt64Array required by ONNX Runtime Web for int64 tensors
        // is not supported in all browsers
        if (kIsWeb) {
          return;
        }

        final inputData = Int32List.fromList([1, 2, 3, 4]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);

        final convertedTensor = await tensor.to(OrtDataType.int64);
        expect(convertedTensor.dataType, OrtDataType.int64);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Int64 to Int32 conversion with cutoff values', (WidgetTester tester) async {
        // skip the test for web platform as BigInt64Array required by ONNX Runtime Web for int64 tensors
        // is not supported in all browsers
        if (kIsWeb) {
          return;
        }
        // Use numbers outside the range of Int32 (e.g., greater than 2^31 - 1)
        final inputData = Int64List.fromList([2147483647, -2147483648, 9223372036, -9223372036]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int64);

        final convertedTensor = await tensor.to(OrtDataType.int32);
        expect(convertedTensor.dataType, OrtDataType.int32);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        expect(retrievedData[0], 2147483647); // 2e31-1
        expect(retrievedData[1], -2147483648); // -2e31
        // Android does not support cutoff values
        if (!Platform.isAndroid) {
          expect(retrievedData[2], 2147483647); // 2e31-1 - cutoff value
          expect(retrievedData[3], -2147483648); // -2e31 - cutoff value
        }

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Same type conversion Float32 to Float32', (WidgetTester tester) async {
        // same type conversion should clone the tensor to a new tensor
        final inputData = Float32List.fromList([1.1, 2.2]);
        final shape = [2]; // 1D array

        final tensor0 = await OrtValue.fromList(inputData, shape);
        expect(tensor0.dataType, OrtDataType.float32);

        final tensor1 = await tensor0.to(OrtDataType.float32);
        expect(tensor1.dataType, OrtDataType.float32);
        expect(tensor1.shape, shape);

        // release tensor0 but tensor1 should still be valid
        tensor0.dispose();
        expect(tensor1.dataType, OrtDataType.float32);
        expect(tensor1.shape, shape);

        final retrievedData = await tensor1.asList();
        expect(retrievedData.length, 2);
        expect(retrievedData[0], closeTo(1.1, 1e-5));
        expect(retrievedData[1], closeTo(2.2, 1e-5));

        await tensor1.dispose();
      });

      testWidgets('Same type conversion String to String', (WidgetTester tester) async {
        // same type conversion should clone the tensor to a new tensor
        final inputData = ['Hello', 'World'];
        final shape = [2]; // 1D array

        final tensor0 = await OrtValue.fromList(inputData, shape);
        expect(tensor0.dataType, OrtDataType.string);

        final tensor1 = await tensor0.to(OrtDataType.string);
        expect(tensor1.dataType, OrtDataType.string);
        expect(tensor1.shape, shape);

        // release tensor0 but tensor1 should still be valid
        tensor0.dispose();
        expect(tensor1.dataType, OrtDataType.string);
        expect(tensor1.shape, shape);

        final retrievedData = await tensor1.asList();
        expect(retrievedData.length, 2);
        expect(retrievedData[0], 'Hello');
        expect(retrievedData[1], 'World');

        await tensor1.dispose();
      });
    });

    group('Tensor Creation Tests', () {
      testWidgets('Tensor size and target shape mismatch', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4, 5.5];
        final shape = [2, 2];

        // expect to throw an PlatformException
        expect(() async => await OrtValue.fromList(inputData, shape), throwsA(isA<ArgumentError>()));
      });

      testWidgets('Nested list to tensor', (WidgetTester tester) async {
        final inputData = [
          [1.1],
          [2.2, 3.3, 4.4],
        ];
        final shape = [2, 2];

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, shape[0]);
        expect(retrievedData[0].length, shape[1]);
      });

      testWidgets('Negative value in shape test', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4, 5.5];
        final shape = [-1, 2, 2];

        // expect to throw an ArgumentError
        expect(() async => await OrtValue.fromList(inputData, shape), throwsA(isA<ArgumentError>()));
      });
    });

    group('Tensor Data Extraction Shape Test', () {
      testWidgets('Test 2x2 tensor data as multi-dim list', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4];
        final shape = [2, 2];
        final tensor = await OrtValue.fromList(inputData, shape);

        final tensorData = await tensor.asList();
        expect(tensorData, isA<List>());
        expect(tensorData[0], isA<List>());

        expect(tensorData.length, shape[0]);
        var id = 0;
        for (final sublist in tensorData) {
          expect(sublist.length, shape[1]);
          for (final num in sublist) {
            expect(num, closeTo(inputData[id], 1e-5));
            id++;
          }
        }
      });

      testWidgets('Test 2x2 tensor data as flat list', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4];
        final shape = [2, 2];
        final tensor = await OrtValue.fromList(inputData, shape);

        final tensorData1d = await tensor.asFlattenedList();
        expect(tensorData1d, isA<List>());
        expect(tensorData1d.length, 4);

        for (int i = 0; i < inputData.length; i++) {
          expect(tensorData1d[i], closeTo(inputData[i], 1e-5));
        }
      });
    });

    group('Error handling tests', () {
      testWidgets('Empty list error test', (WidgetTester tester) async {
        final emptyList = [];
        final shape = [0];

        expect(() async => await OrtValue.fromList(emptyList, shape), throwsArgumentError);
      });

      testWidgets('Mismatched shape error test', (WidgetTester tester) async {
        final inputData = [1.0, 2.0, 3.0, 4.0];
        // Total elements in shape doesn't match list length
        final wrongShape = [5, 5];

        expect(() async => await OrtValue.fromList(inputData, wrongShape), throwsA(anything));
      });

      testWidgets('Unsupported conversion test', (WidgetTester tester) async {
        final inputData = [true, false, true];
        final shape = [3];

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.bool);

        // This test checks that converting boolean tensor to float32 either
        // throws an exception or successfully converts the values
        try {
          final convertedTensor = await tensor.to(OrtDataType.float32);
          // If conversion succeeds, verify the data
          expect(convertedTensor.dataType, OrtDataType.float32);
          final retrievedData = await convertedTensor.asList();
          expect(retrievedData.length, 3);
          await convertedTensor.dispose();
        } catch (e) {
          // If an exception is thrown, that's also acceptable
          // as boolean to float conversion might not be supported
          expect(e, isNotNull);
        }

        await tensor.dispose();
      });

      testWidgets('Using disposed tensor test', (WidgetTester tester) async {
        final tensor = await OrtValue.fromList([1.0, 2.0, 3.0], [3]);
        await tensor.dispose();
        expect(
          () async => await tensor.asList(),
          throwsA(isA<PlatformException>().having((e) => e.code, 'code', "INVALID_VALUE")),
        );
      });
    });
  });

  group('Environment setup', () {
    late OnnxRuntime onnxRuntime;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
    });

    testWidgets('Get platform version', (WidgetTester tester) async {
      final version = await onnxRuntime.getPlatformVersion();
      // ignore: avoid_print
      print('Platform Version: $version');
      expect(version, isNotNull);
      expect(version!.isNotEmpty, true);
    });

    testWidgets('Available providers', (WidgetTester tester) async {
      try {
        final providers = await onnxRuntime.getAvailableProviders();
        expect(providers, isNotNull);
        expect(providers.isNotEmpty, true);
      } catch (e) {
        fail('Failed to get available providers: $e');
      }
    });

    testWidgets('Create session with CPU provider', (WidgetTester tester) async {
      final session = await onnxRuntime.createSessionFromAsset(
        'assets/models/addition_model.ort',
        options: OrtSessionOptions(providers: [OrtProvider.CPU]),
      );
      expect(session, isNotNull);
      await session.close();
    });

    testWidgets('Create session with unavailable provider', (WidgetTester tester) async {
      // set a negative provider
      // we assume that CORE_ML is never available on Android and Linux
      // and XNNPACK is never available on iOS and MacOS
      OrtProvider negativeProvider = OrtProvider.CORE_ML;
      // Note: the Platform.is<OS> call is not supported on web and will cause issue when testing in web environment
      if (!kIsWeb) {
        if (Platform.isIOS || Platform.isMacOS) {
          negativeProvider = OrtProvider.AZURE;
        }
      }
      try {
        await onnxRuntime.createSessionFromAsset(
          'assets/models/addition_model.ort',
          options: OrtSessionOptions(providers: [negativeProvider]),
        );
        fail('Session should not be created');
      } catch (e) {
        expect(e, isA<PlatformException>());
      }
    });
  });

  group('Session Info Tests', () {
    late OnnxRuntime onnxRuntime;
    late OrtSession session;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
      try {
        // Load model from assets
        session = await onnxRuntime.createSessionFromAsset('assets/models/addition_model.ort');
      } catch (e) {
        fail('Failed to create session: $e');
      }
    });

    tearDownAll(() async {
      await session.close();
    });

    testWidgets('Get model metadata', (WidgetTester tester) async {
      final metadata = await session.getMetadata();
      expect(metadata, isNotNull);
      expect(metadata.producerName, isNotNull);
    });

    testWidgets('Get input/output info', (WidgetTester tester) async {
      final inputInfo = await session.getInputInfo();
      expect(inputInfo, isNotNull);
      expect(inputInfo.length, 2); // Addition model has inputs A and B

      // Verify the input names
      expect(inputInfo.map((i) => i['name']).toList(), containsAll(['A', 'B']));

      final outputInfo = await session.getOutputInfo();
      expect(outputInfo, isNotNull);
      expect(outputInfo.length, 1); // Addition model has single output C
      expect(outputInfo[0]['name'], 'C');

      // check shape of input and output tensor
      // Note: Tensor input and output shapes are not available for iOS and macOS
      if (!kIsWeb && !Platform.isIOS && !Platform.isMacOS) {
        expect(inputInfo[0]['shape'], [-1]);
        expect(inputInfo[1]['shape'], [-1]);
        expect(outputInfo[0]['shape'], [-1]);
      }
    });
  });

  group('Inference Tests with Addition Model', () {
    late OnnxRuntime onnxRuntime;
    late OrtSession session;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
      try {
        // Load model from assets
        session = await onnxRuntime.createSessionFromAsset('assets/models/addition_model.ort');
      } catch (e) {
        fail('Failed to create session: $e');
      }
    });

    tearDownAll(() async {
      await session.close();
    });

    testWidgets('Add two numbers', (WidgetTester tester) async {
      // Create OrtValue inputs instead of raw arrays
      final inputA = await OrtValue.fromList([3.0], [1]);
      final inputB = await OrtValue.fromList([4.0], [1]);

      final inputs = {'A': inputA, 'B': inputB};

      final outputs = await session.run(inputs);
      final outputTensor = outputs['C'];
      final outputData = await outputTensor!.asList();

      expect(outputs.length, 1);
      expect(outputTensor.dataType, OrtDataType.float32);
      expect(outputTensor.shape, [1]);
      expect(outputData[0], 7.0); // 3 + 4 = 7

      // Clean up
      await inputA.dispose();
      await inputB.dispose();
    });

    testWidgets('Add two arrays of numbers', (WidgetTester tester) async {
      // Create OrtValue inputs instead of raw arrays
      final inputA = await OrtValue.fromList([1.1, 2.2, 3.3], [3]);
      final inputB = await OrtValue.fromList([4.4, 5.5, 6.6], [3]);

      final inputs = {'A': inputA, 'B': inputB};

      final outputs = await session.run(inputs);
      final outputTensor = outputs['C'];
      final outputData = await outputTensor!.asList();

      expect(outputs.length, 1);
      expect(outputTensor.dataType, OrtDataType.float32);
      expect(outputTensor.shape, [3]);
      expect(outputData.length, 3);
      expect(outputData[0], closeTo(5.5, 1e-5)); // 1.1 + 4.4 ≈ 5.5
      expect(outputData[1], closeTo(7.7, 1e-5)); // 2.2 + 5.5 ≈ 7.7
      expect(outputData[2], closeTo(9.9, 1e-5)); // 3.3 + 6.6 ≈ 9.9

      // Clean up
      await inputA.dispose();
      await inputB.dispose();
    });

    testWidgets('Run inference with run options', (WidgetTester tester) async {
      final runOptions = OrtRunOptions(logSeverityLevel: 1, logVerbosityLevel: 1, terminate: false);
      final inputs = {
        'A': await OrtValue.fromList([3.0], [1]),
        'B': await OrtValue.fromList([4.0], [1]),
      };
      final outputs = await session.run(inputs, options: runOptions);
      final outputTensor = outputs['C'];
      final outputData = await outputTensor!.asList();

      expect(outputs.length, 1);
      expect(outputTensor.dataType, OrtDataType.float32);
      expect(outputTensor.shape, [1]);
      expect(outputData[0], 7.0); // 3 + 4 = 7

      // Clean up
      await inputs['A']!.dispose();
      await inputs['B']!.dispose();
      await outputs['C']!.dispose();
    });

    testWidgets('Run inference with run options and terminate', (WidgetTester tester) async {
      // Note: the Platform.is<OS> call is not supported on web and will cause issue when testing in web environment
      if (!kIsWeb) {
        // Skip the test for iOS and macOS as terminate is not supported
        if (Platform.isIOS || Platform.isMacOS) {
          return; // Skip the test
        }
      }
      final runOptions = OrtRunOptions(logSeverityLevel: 1, logVerbosityLevel: 1, terminate: true);
      final inputs = {
        'A': await OrtValue.fromList([3.0], [1]),
        'B': await OrtValue.fromList([4.0], [1]),
      };

      // expect the run to throw an exception since terminate is true
      expect(() async {
        await session.run(inputs, options: runOptions);
      }, throwsA(isA<Exception>()));

      // Clean up
      await inputs['A']!.dispose();
      await inputs['B']!.dispose();
    });
  });

  group('Transpose and Avg Model Tests', () {
    group('FP32 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;
      late Map<String, OrtValue> inputs;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_fp32.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('Inference test with single batch', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]),
          'B': await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 3, 2]),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.float32);
        expect(output.shape, [1, 2, 3]);
        final outputData = await output.asList();
        expect(outputData.length, output.shape[0]);
        expect(outputData[0].length, output.shape[1]);
        expect(outputData[0][0].length, output.shape[2]);
        expect(outputData.expand((e0) => e0).expand((e) => e).every((element) => element == 1.5), true);

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });

      testWidgets('Inference test with multi-batch', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList(
            [
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [2, 2, 3],
          ),
          'B': await OrtValue.fromList(
            [
              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            [2, 3, 2],
          ),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.float32);
        expect(output.shape, [2, 2, 3]);
        final outputData = await output.asList();
        final outputData1d = await output.asFlattenedList();
        // check values of 1d list
        expect(outputData1d.length, 12);
        expect(outputData1d.every((e) => e == 1.5), true);
        // check shape of multi-dim list
        expect(outputData.length, output.shape[0]);
        expect(outputData[0].length, output.shape[1]);
        expect(outputData[0][0].length, output.shape[2]);

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });

      testWidgets('Invalid input rank test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2, 3]),
          'B': await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [3, 2]),
        };
        // expect to throw an exeption
        expect(() async => await session.run(inputs), throwsA(isA<Exception>()));

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
      });

      testWidgets('Invalid input shape test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]),
          'B': await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 2, 3]),
        };
        // expect to throw an exeption
        expect(() async => await session.run(inputs), throwsA(isA<Exception>()));

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
      });

      testWidgets('Invalid input type', (WidgetTester tester) async {
        // create int tensors
        final tensorA = await OrtValue.fromList(Int32List.fromList([1, 1, 1, 1, 1, 1]), [1, 2, 3]);
        final tensorB = await OrtValue.fromList(Int32List.fromList([2, 2, 2, 2, 2, 2]), [1, 3, 2]);
        // expect to throw an exeption
        expect(() async => await session.run({'A': tensorA, 'B': tensorB}), throwsA(isA<Exception>()));

        // clean up
        await tensorA.dispose();
        await tensorB.dispose();
      });

      testWidgets('Invalid input name test', (WidgetTester tester) async {
        // Create tensors with correct shapes but using wrong input name
        final tensorA = await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]);
        final tensorB = await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 3, 2]);
        // await session.run({'X': tensorA, 'B': tensorB});

        // Use wrong input name (X instead of A)
        // Expect to throw an exception with code "INFERENCE_ERROR"
        expect(
          () async => await session.run({'X': tensorA, 'B': tensorB}),
          throwsA(isA<PlatformException>().having((e) => e.code, 'code', "INFERENCE_ERROR")),
        );

        // Clean up
        await tensorA.dispose();
        await tensorB.dispose();
      });

      testWidgets('Random input order test', (WidgetTester tester) async {
        // Create tensors with correct shapes
        final tensorA = await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]);
        final tensorB = await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 3, 2]);

        // Run inference with inputs in normal order
        final outputsNormal = await session.run({'A': tensorA, 'B': tensorB});
        final outputNormal = outputsNormal['C'];

        // Run inference with inputs in reverse order
        final outputsReversed = await session.run({'B': tensorB, 'A': tensorA});
        final outputReversed = outputsReversed['C'];

        // Verify both outputs are the same
        expect(outputReversed!.dataType, outputNormal!.dataType);
        expect(outputReversed.shape, outputNormal.shape);

        final outputDataNormal = await outputNormal.asFlattenedList();
        final outputDataReversed = await outputReversed.asFlattenedList();

        expect(outputDataNormal.length, outputDataReversed.length);
        for (int i = 0; i < outputDataNormal.length; i++) {
          expect(outputDataReversed[i], outputDataNormal[i]);
        }

        // Clean up
        await tensorA.dispose();
        await tensorB.dispose();
        await outputNormal.dispose();
        await outputReversed.dispose();
      });
    });

    group('INT32 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;
      late Map<String, OrtValue> inputs;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_int32.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('INT32 model inference test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList(Int32List.fromList([1, 1, 1, 1, 1, 1]), [1, 2, 3]),
          'B': await OrtValue.fromList(Int32List.fromList([2, 2, 2, 2, 2, 2]), [1, 3, 2]),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.int32);
        expect(output.shape, [1, 2, 3]);

        final outputData = await output.asFlattenedList();
        expect(outputData.length, 6);
        expect(outputData.every((e) => e == 1), true); // 1 + 2 = 3, 3 * 0.5 = 1.5 -> 1

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });
    });

    group('INT64 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;
      late Map<String, OrtValue> inputs;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_int64.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('INT64 model inference test', (WidgetTester tester) async {
        // skip the test for web platform as BigInt64Array required by ONNX Runtime Web for int64 tensors
        // is not supported in all browsers
        if (kIsWeb) {
          return;
        }
        inputs = {
          'A': await OrtValue.fromList(Int64List.fromList([1, 1, 1, 1, 1, 1]), [1, 2, 3]),
          'B': await OrtValue.fromList(Int64List.fromList([2, 2, 2, 2, 2, 2]), [1, 3, 2]),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.int64);
        expect(output.shape, [1, 2, 3]);

        final outputData = await output.asFlattenedList();
        expect(outputData.length, 6);
        expect(outputData.every((e) => e == 1), true); // 1 + 2 = 3, 3 * 0.5 = 1.5 -> 1

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });
    });

    group('FP16 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_fp16.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('FP16 model inference test', (WidgetTester tester) async {
        final tensorA = await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]);
        final tensorB = await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 3, 2]);
        // only support Android
        if (!kIsWeb && Platform.isAndroid) {
          // convert to fp16
          final tensorAFp16 = await tensorA.to(OrtDataType.float16);
          final tensorBFp16 = await tensorB.to(OrtDataType.float16);

          final outputs = await session.run({'A': tensorAFp16, 'B': tensorBFp16});
          final output = outputs['C'];
          expect(output!.dataType, OrtDataType.float16);
          expect(output.shape, [1, 2, 3]);

          final outputData = await output.asList();
          // check shape
          expect(outputData.length, output.shape[0]);
          expect(outputData[0].length, output.shape[1]);
          expect(outputData[0][0].length, output.shape[2]);
          // check value
          expect(outputData.expand((e0) => e0).expand((e) => e).every((element) => element == 1.5), true);

          // clean up
          await tensorA.dispose();
          await tensorB.dispose();
          await tensorAFp16.dispose();
          await tensorBFp16.dispose();
          await output.dispose();
        } else {
          expect(
            () async => await tensorA.to(OrtDataType.float16),
            throwsA(isA<PlatformException>().having((e) => e.code, 'code', "CONVERSION_ERROR")),
          );
        }
      });
    });
  });

  group('StringConcat Model Test', () {
    late OnnxRuntime onnxRuntime;
    late OrtSession session;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
      session = await onnxRuntime.createSessionFromAsset('assets/models/string_concat_model.onnx');
    });

    tearDownAll(() async {
      await session.close();
    });

    testWidgets('StringConcat model inference test', (WidgetTester tester) async {
      final inputs = {
        'input1': await OrtValue.fromList(['Hello'], [1]),
        'input2': await OrtValue.fromList(['World'], [1]),
      };

      final outputs = await session.run(inputs);
      final output = outputs['output'];
      expect(output!.dataType, OrtDataType.string);
      expect(output.shape, [1]);
      final outputData = await output.asFlattenedList();
      expect(outputData, ['HelloWorld']);

      // clean up
      for (var input in inputs.values) {
        input.dispose();
      }
      await output.dispose();
    });

    testWidgets('StringConcat model inference test with multi-batch', (WidgetTester tester) async {
      final inputs = {
        'input1': await OrtValue.fromList(['Hello', 'flutter'], [2]),
        'input2': await OrtValue.fromList(['World!', '_onnxruntime'], [2]),
      };

      final outputs = await session.run(inputs);
      final output = outputs['output'];
      expect(output!.dataType, OrtDataType.string);
      expect(output.shape, [2]);
      final outputData = await output.asFlattenedList();
      expect(outputData, ['HelloWorld!', 'flutter_onnxruntime']);

      // clean up
      for (var input in inputs.values) {
        input.dispose();
      }
      await output.dispose();
    });
  });
}

/// Prints benchmark results for data transfer operations.
///
/// This function outputs timing information and memory statistics for tensor operations.
/// Output is controlled by the global [_enableBenchmarkPrints] constant.
void _printBenchmarkResults({
  required String dataTypeName,
  required int totalElements,
  required int bytesPerElement,
  required Duration creationTime,
  required Duration flattenedTime,
  required Duration nestedTime,
  bool isMemoryEstimated = false,
}) {
  if (!_enableBenchmarkPrints) return;

  final memoryMB = (totalElements * bytesPerElement / (1024 * 1024)).toStringAsFixed(2);
  final memoryPrefix = isMemoryEstimated ? '~' : '';
  final memorySuffix = isMemoryEstimated ? ' (estimated)' : '';

  // ignore: avoid_print
  print('\n=== $dataTypeName Data Transfer Benchmark [512x512x3] ===');
  // ignore: avoid_print
  print('Total elements: $totalElements');
  // ignore: avoid_print
  print('Memory size: $memoryPrefix$memoryMB MB$memorySuffix');
  // ignore: avoid_print
  print('Creation time (Dart→Native): ${creationTime.inMilliseconds}ms');
  // ignore: avoid_print
  print('Flattened retrieval (Native→Dart): ${flattenedTime.inMilliseconds}ms');
  // ignore: avoid_print
  print('Nested retrieval (Native→Dart): ${nestedTime.inMilliseconds}ms');
  // ignore: avoid_print
  print('Total round-trip time: ${(creationTime.inMilliseconds + flattenedTime.inMilliseconds)}ms');
  // ignore: avoid_print
  print('================================================\n');
}

/// Runs a benchmark test for tensor data transfer operations.
///
/// This function performs the following steps:
/// 1. Creates a tensor from the input data
/// 2. Measures timing for creation, flattened retrieval, and nested retrieval
/// 3. Optionally prints benchmark results based on [_enableBenchmarkPrints]
/// 4. Validates tensor properties and data integrity
/// 5. Disposes the tensor
///
/// Parameters:
/// - [inputData]: The input data for tensor creation (can be TypedData or List)
/// - [shape]: The shape of the tensor
/// - [dataTypeName]: Name of the data type for display purposes
/// - [bytesPerElement]: Number of bytes per element for memory calculation
/// - [expectedType]: Expected OrtDataType for validation
/// - [isMemoryEstimated]: Whether memory size is an estimate (for strings)
/// - [validatePerformance]: Whether to validate that flattened retrieval is faster than creation
Future<void> _runBenchmarkTest({
  required dynamic inputData,
  required List<int> shape,
  required String dataTypeName,
  required int bytesPerElement,
  required OrtDataType expectedType,
  bool isMemoryEstimated = false,
  bool validatePerformance = false,
}) async {
  final totalElements = shape.fold(1, (a, b) => a * b);

  // Benchmark: Dart → Native (tensor creation)
  final creationStart = DateTime.now();
  final tensor = await OrtValue.fromList(inputData, shape);
  final creationTime = DateTime.now().difference(creationStart);

  // Benchmark: Native → Dart (flattened retrieval)
  final flattenedStart = DateTime.now();
  final flattenedData = await tensor.asFlattenedList();
  final flattenedTime = DateTime.now().difference(flattenedStart);

  // Benchmark: Native → Dart (nested/reshaped retrieval)
  final nestedStart = DateTime.now();
  final nestedData = await tensor.asList();
  final nestedTime = DateTime.now().difference(nestedStart);

  // Print benchmark results (controlled by global constant)
  _printBenchmarkResults(
    dataTypeName: dataTypeName,
    totalElements: totalElements,
    bytesPerElement: bytesPerElement,
    creationTime: creationTime,
    flattenedTime: flattenedTime,
    nestedTime: nestedTime,
    isMemoryEstimated: isMemoryEstimated,
  );

  // Minimal validation: shape checks only
  expect(tensor.dataType, expectedType);
  expect(tensor.shape, shape);
  expect(flattenedData.length, totalElements);
  expect(nestedData.length, shape[0]);
  expect(nestedData[0].length, shape[1]);
  expect(nestedData[0][0].length, shape[2]);

  // Validation: for typed arrays, flattened retrieval should be faster than or equal to creation
  if (validatePerformance) {
    expect(
      flattenedTime.inMilliseconds <= creationTime.inMilliseconds,
      true,
      reason: 'Flattened retrieval should be faster than creation for typed arrays',
    );
  }

  await tensor.dispose();
}
