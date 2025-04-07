// ONNX Runtime integration test
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter/widgets.dart';

void main() {
  // Initialize both bindings to ensure everything is set up properly
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  WidgetsFlutterBinding.ensureInitialized();

  group('ONNX Runtime Integration Tests', () {
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

    testWidgets('Get platform version', (WidgetTester tester) async {
      final version = await onnxRuntime.getPlatformVersion();
      expect(version, isNotNull);
      expect(version!.isNotEmpty, true);
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
    });

    testWidgets('Add two numbers', (WidgetTester tester) async {
      final inputs = {
        'A': [3.0],
        'B': [4.0],
      };

      final outputs = await session.run(inputs);
      expect(outputs, isNotNull);
      expect(outputs['outputs']['C'][0], 7.0); // 3 + 4 = 7
    });

    testWidgets('Add two arrays of numbers', (WidgetTester tester) async {
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
  });
}
