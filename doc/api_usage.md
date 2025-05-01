# Flutter ONNX Runtime API Usage Guide

This guide provides examples of how to use the Flutter ONNX Runtime plugin to run machine learning models in your Flutter applications.

## Installation

Add the following dependency to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_onnxruntime: ^1.4.0
```

## Basic Usage

### Importing the Library

```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
```

### Creating a Session

```dart
// Create an instance of OnnxRuntime
final ort = OnnxRuntime();

// Create a session from a model file
final session = await ort.createSession(
  'path/to/model.onnx',
  options: OrtSessionOptions(
    intraOpNumThreads: 2,
    interOpNumThreads: 1,
    providers: ['CPU'],
    useArena: true,
    deviceId: 0,
  ),
);

// Session information
print('Input names: ${session.inputNames}');
print('Output names: ${session.outputNames}');
```

### Creating a Session from an Asset

```dart
// Create a session from an asset file
final session = await ort.createSessionFromAsset(
  'assets/model.onnx',
);
```

### Getting Available Providers

```dart
final providers = await ort.getAvailableProviders();
print('Available providers: $providers');
```

### Running Inference

```dart
// Create OrtValue tensors for input
final inputTensor = await OrtValue.fromList(
  [1.0, 2.0, 3.0, 4.0],
  [2, 2], // Shape: 2x2 matrix
);

// Create inputs map with OrtValue objects
final inputs = {
  'input_name': inputTensor,
};

// Run inference
final outputs = await session.run(inputs);

// Process outputs - the result is a map of output names to OrtValue objects
final outputTensor = outputs['output_name'];
print('Output shape: ${outputTensor.shape}');

// Get the data from the output tensor
final outputData = await outputTensor.asList();
print('Output data: $outputData');

// Always dispose tensors to free resources
await inputTensor.dispose();
for (final tensor in outputs.values) {
  await tensor.dispose();
}
```

### Closing the Session

```dart
// Always close the session when done to free resources
await session.close();
```

## Working with OrtValue

The `OrtValue` class provides a way to manage tensors:

### Creating Tensors

```dart
// Create from Float32List
final float32Tensor = await OrtValue.fromList(
  Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
  [2, 2], // Shape: 2x2 matrix
);

// Create from Int32List
final int32Tensor = await OrtValue.fromList(
  Int32List.fromList([1, 2, 3, 4]),
  [4], // Shape: vector of 4 elements
);

// Create from Uint8List (for images)
final uint8Tensor = await OrtValue.fromList(
  Uint8List.fromList([255, 0, 255, 0]),
  [2, 2],
);

// Create from regular List (auto-converts to appropriate type)
final autoTensor = await OrtValue.fromList(
  [1.0, 2.0, 3.0, 4.0],
  [2, 2],
);
```

### Tensor Data Type Conversion

```dart
// Convert to different data type
final float16Tensor = await float32Tensor.to(OrtDataType.float16);
```

### Accessing Tensor Data

```dart
// Get data as a multi-dimensional List following the tensor's shape
final tensorData = await float32Tensor.asList();
print('Tensor data (shaped): $tensorData');
// For a tensor with shape [2, 2], this prints: [[1.0, 2.0], [3.0, 4.0]]

// Get data as a flattened 1D List
final flattenedData = await float32Tensor.asFlattenedList();
print('Tensor data (flattened): $flattenedData');
// This prints: [1.0, 2.0, 3.0, 4.0]
```

### Important Memory Management

OrtValue instances must be explicitly disposed to free native resources:

```dart
// Dispose of tensors when no longer needed
await float32Tensor.dispose();
await float16Tensor.dispose();
```

## Advanced Usage

### Getting Model Metadata

```dart
// Get model metadata
final metadata = await session.getMetadata();
print('Producer: ${metadata.producerName}');
print('Graph name: ${metadata.graphName}');
print('Domain: ${metadata.domain}');
print('Description: ${metadata.description}');
print('Version: ${metadata.version}');
print('Custom metadata: ${metadata.customMetadataMap}');
```

### Getting Input/Output Information

```dart
// Get detailed input information
final inputInfo = await session.getInputInfo();
for (final info in inputInfo) {
  print('Input: ${info['name']}');
  print('  Shape: ${info['shape']}');
  print('  Type: ${info['type']}');
}

// Get detailed output information
final outputInfo = await session.getOutputInfo();
for (final info in outputInfo) {
  print('Output: ${info['name']}');
  print('  Shape: ${info['shape']}');
  print('  Type: ${info['type']}');
}
```

## Best Practices

1. **Resource Management**
   - Always call `session.close()` when done with a session
   - Always call `tensor.dispose()` when done with tensors
   - Use try/finally blocks to ensure resources are released even if errors occur

2. **Tensor Lifecycle Management**
   
   ❌ **Don't**: Reassign a tensor without disposing the original
   ```dart
   var tensor = await OrtValue.fromList([1.0, 2.0], [2]);
   tensor = await tensor.to(OrtDataType.int32); // Memory leak!
   ```
   
   ✅ **Do**: Create a new variable for converted tensors
   ```dart
   var floatTensor = await OrtValue.fromList([1.0, 2.0], [2]);
   var intTensor = await floatTensor.to(OrtDataType.int32);
   await floatTensor.dispose(); // Properly dispose original
   // Use intTensor
   await intTensor.dispose(); // Dispose when done
   ```

3. **Performance Optimization**
   - Reuse OrtValue instances when possible rather than creating new ones for each inference
   - Use appropriate data types (e.g., Float32List is generally more efficient than List<double>)
   - Consider batch processing instead of individual inferences when appropriate

4. **Memory Efficiency**
   - Dispose of large tensors immediately after use
   - Be mindful of tensor shapes and sizes, especially for mobile devices 