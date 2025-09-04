import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;

import 'package:flutter_web_plugins/flutter_web_plugins.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:web/web.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter/services.dart'; // Import for PlatformException

// Import the onnxruntime-web package via JS interop
@JS('ort')
external JSObject get _ort;

@JS('window')
external WindowJS get _window;

// Add global context from dart:js_interop with proper annotation
@JS()
external JSObject get globalThis;

// Object constructor for creating JavaScript objects
@JS('Object')
external JSFunction get objectConstructor;

// Array constructor for creating JavaScript arrays
@JS('Array')
external JSFunction get arrayConstructor;

// TypedArray constructors
@JS('Float32Array')
external JSFunction get float32ArrayConstructor;

@JS('Int32Array')
external JSFunction get int32ArrayConstructor;

@JS('BigInt64Array')
external JSFunction get bigInt64ArrayConstructor;

@JS('Uint8Array')
external JSFunction get uint8ArrayConstructor;

// ONNX Runtime Tensor constructor
@JS('ort.Tensor')
external JSFunction get tensorConstructor;

// JavaScript Object global for Object.keys() operations
@JS('Object')
external JSObject get objectGlobal;

@JS()
@staticInterop
class WindowJS {}

extension WindowJSExtension on WindowJS {
  external NavigatorJS get navigator;
}

@JS()
@staticInterop
class NavigatorJS {}

extension NavigatorJSExtension on NavigatorJS {
  external String get userAgent;
}

/// Web implementation of [FlutterOnnxruntimePlatform] using Dart JS interop.
class FlutterOnnxruntimeWebPlugin extends FlutterOnnxruntimePlatform {
  // A map to store the created inference sessions
  final Map<String, JSObject> _sessions = {};

  /// Registers this class as the default instance of [FlutterOnnxruntimePlatform].
  static void registerWith(Registrar registrar) {
    FlutterOnnxruntimePlatform.instance = FlutterOnnxruntimeWebPlugin();
  }

  @override
  Future<String?> getPlatformVersion() async {
    return _window.navigator.userAgent;
  }

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) async {
    try {
      // Initialize session options for onnxruntime-web
      final jsSessionOptions = createJsSessionOptions(sessionOptions);

      // Create the session using onnxruntime-web
      final session = await createInferenceSession(modelPath, jsSessionOptions);

      // Generate a unique session ID
      final sessionId = DateTime.now().millisecondsSinceEpoch.toString();

      // Get input and output names
      final inputNames = getInputNames(session);
      final outputNames = getOutputNames(session);

      // Store the session for later use
      _sessions[sessionId] = session;

      // Return the required information
      return {'sessionId': sessionId, 'inputNames': inputNames, 'outputNames': outputNames};
    } catch (e) {
      throw PlatformException(code: 'PLUGIN_ERROR', message: 'Failed to create ONNX session: $e', details: null);
    }
  }

  /// Creates JavaScript session options object
  JSObject createJsSessionOptions(Map<String, dynamic>? options) {
    // Get the SessionOptions constructor from onnxruntime-web
    final jsOptions = createJSObject();

    if (options != null) {
      // Set execution providers if specified
      if (options.containsKey('providers')) {
        final providers = options['providers'] as List<String>;
        final jsProviders =
            providers.map((provider) {
              // Map Flutter provider names to onnxruntime-web provider names
              switch (provider) {
                case 'CPU':
                case 'WEB_ASSEMBLY':
                  return 'wasm';
                case 'WEB_GL':
                  return 'webgl';
                case 'WEB_GPU':
                  return 'webgpu';
                case 'WEB_NN':
                  return 'webnn';
                default:
                  throw PlatformException(
                    code: 'INVALID_PROVIDER',
                    message: 'Provider $provider is not supported',
                    details: null,
                  );
              }
            }).toList();

        // Set executionProviders property
        setProperty(jsOptions, 'executionProviders', jsArrayFrom(jsProviders));
      }

      // Set other options like intraOpNumThreads, interOpNumThreads if needed
      if (options.containsKey('intraOpNumThreads')) {
        setProperty(jsOptions, 'intraOpNumThreads', options['intraOpNumThreads']);
      }

      if (options.containsKey('interOpNumThreads')) {
        setProperty(jsOptions, 'interOpNumThreads', options['interOpNumThreads']);
      }

      // Handle graph optimization level
      if (options.containsKey('graphOptimizationLevel')) {
        setProperty(jsOptions, 'graphOptimizationLevel', options['graphOptimizationLevel']);
      }

      // Add more options as needed
    }

    return jsOptions;
  }

  /// Creates an inference session using onnxruntime-web
  Future<JSObject> createInferenceSession(String modelPath, JSObject options) async {
    final completer = Completer<JSObject>();

    try {
      // Get the InferenceSession class from onnxruntime-web
      final inferenceSession = getProperty(_ort, 'InferenceSession');

      // Use the create method to create a session - async operation
      final createPromise = callMethod(inferenceSession, 'create', [modelPath, options]);

      // Convert Promise to Future
      final session = await promiseToFuture<JSObject>(createPromise);
      completer.complete(session);
    } catch (e) {
      completer.completeError('Failed to create inference session: $e');
    }

    return completer.future;
  }

  /// Get input names from the session
  List<String> getInputNames(JSObject session) {
    final inputNames = <String>[];
    try {
      final inputs = getProperty(session, 'inputNames');
      final length = getProperty(inputs, 'length') as int;

      for (var i = 0; i < length; i++) {
        final name = callMethod(inputs, 'at', [i]);
        inputNames.add(name.toString());
      }
    } catch (e) {
      // print('Error getting input names: $e');
    }
    return inputNames;
  }

  /// Get output names from the session
  List<String> getOutputNames(JSObject session) {
    final outputNames = <String>[];
    try {
      final outputs = getProperty(session, 'outputNames');
      final length = getProperty(outputs, 'length') as int;

      for (var i = 0; i < length; i++) {
        final name = callMethod(outputs, 'at', [i]);
        outputNames.add(name.toString());
      }
    } catch (e) {
      // print('Error getting output names: $e');
    }
    return outputNames;
  }

  // Helper JS interop utilities
  JSObject createJSObject() => objectConstructor.callAsConstructor();

  void setProperty(JSObject obj, String name, dynamic value) {
    obj.setProperty(name.toJS, value);
  }

  dynamic getProperty(JSObject obj, String name) {
    return obj.getProperty(name.toJS);
  }

  // Method overloads for different argument counts
  dynamic callMethod0(JSObject obj, String method) {
    return obj.callMethod(method.toJS);
  }

  dynamic callMethod1(JSObject obj, String method, dynamic arg1) {
    return obj.callMethod(method.toJS, arg1);
  }

  dynamic callMethod2(JSObject obj, String method, dynamic arg1, dynamic arg2) {
    return obj.callMethod(method.toJS, arg1, arg2);
  }

  // General case that routes to specific methods
  dynamic callMethod(JSObject obj, String method, List<dynamic> args) {
    switch (args.length) {
      case 0:
        return callMethod0(obj, method);
      case 1:
        return callMethod1(obj, method, args[0]);
      case 2:
        return callMethod2(obj, method, args[0], args[1]);
      default:
        // For 3+ arguments, fall back to VarArgs if needed
        // Note: the cast does not work properly at the moment
        return obj.callMethodVarArgs(method.toJS, args.cast<JSAny?>());
    }
  }

  JSObject jsArrayFrom(List<dynamic> list) {
    return callMethod(arrayConstructor, 'from', [list]);
  }

  Future<T> promiseToFuture<T extends JSAny?>(JSObject promise) {
    return (promise as JSPromise<T>).toDart;
  }

  @override
  Future<List<String>> getAvailableProviders() async {
    // Return the list of execution providers supported by onnxruntime-web
    final providers = <String>[];
    // Check for WebAssembly support
    if (globalThis.has('WebAssembly')) {
      providers.add('WEB_ASSEMBLY');
    }
    // Check for WebGL support
    final canvas = HTMLCanvasElement();
    final webgl = canvas.getContext('webgl') ?? canvas.getContext('experimental-webgl');
    if (webgl != null) {
      providers.add('WEB_GL');
    }
    // Check for WebGPU support
    if ((_window.navigator as JSObject).has('gpu')) {
      providers.add('WEB_GPU');
    }
    // Check for WebNN support
    if ((_window.navigator as JSObject).has('ml')) {
      providers.add('WEB_NN');
    }
    return providers;
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, OrtValue> inputs, {
    Map<String, dynamic>? runOptions,
  }) async {
    try {
      // Check if the session exists
      if (!_sessions.containsKey(sessionId)) {
        throw PlatformException(code: "INVALID_SESSION", message: "Session not found", details: null);
      }

      final session = _sessions[sessionId]!;

      // Create a JavaScript object for inputs
      final jsInputs = createJSObject();

      // Process inputs - expecting OrtValue references
      for (final entry in inputs.entries) {
        final name = entry.key;
        final ortValue = entry.value;

        // Get the tensor by ID
        final valueId = ortValue.id;
        final tensor = _ortValues[valueId];
        if (tensor == null) {
          throw PlatformException(code: "INVALID_VALUE", message: "OrtValue with ID $valueId not found", details: null);
        }

        // Add to inputs object
        setProperty(jsInputs, name, tensor);
      }

      // Create run options if provided
      JSObject? jsRunOptions;
      if (runOptions != null && runOptions.isNotEmpty) {
        jsRunOptions = createJSObject();

        // Set run options if needed
        if (runOptions.containsKey('logSeverityLevel')) {
          setProperty(jsRunOptions, 'logSeverityLevel', runOptions['logSeverityLevel']);
        }

        if (runOptions.containsKey('logVerbosityLevel')) {
          setProperty(jsRunOptions, 'logVerbosityLevel', runOptions['logVerbosityLevel']);
        }

        if (runOptions.containsKey('terminate')) {
          setProperty(jsRunOptions, 'terminate', runOptions['terminate']);
        }
      }

      // Run inference
      final runPromise =
          jsRunOptions != null
              ? callMethod(session, 'run', [jsInputs, jsRunOptions])
              : callMethod(session, 'run', [jsInputs]);

      // Wait for the promise to resolve
      final jsOutputs = await promiseToFuture<JSObject>(runPromise);

      // Process outputs - create OrtValue objects for each output
      final outputMap = <String, dynamic>{};
      final outputNames = getOutputNames(session);

      for (final name in outputNames) {
        if (hasProperty(jsOutputs, name)) {
          final tensor = getProperty(jsOutputs, name);

          // Create a new OrtValue and store it
          final valueId = '${DateTime.now().millisecondsSinceEpoch}_${math.Random().nextInt(10000)}';

          _ortValues[valueId] = tensor;

          // Get tensor type
          final type = getProperty(tensor, 'type').toString();

          // Get tensor shape
          final jsShape = getProperty(tensor, 'dims');
          final shapeLength = getProperty(jsShape, 'length') as int;
          final shape = <int>[];

          for (var i = 0; i < shapeLength; i++) {
            shape.add(callMethod(jsShape, 'at', [i]) as int);
          }

          // Format parameters like native platforms do
          outputMap[name] = [valueId, _mapOrtTypeToSourceType(type), shape];
        }
      }

      return outputMap;
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "INFERENCE_ERROR", message: "Failed to run inference: $e", details: null);
    }
  }

  // Convert ONNX Runtime type to source type
  String _mapOrtTypeToSourceType(String ortType) {
    switch (ortType) {
      case 'float32':
        return 'float32';
      case 'int32':
        return 'int32';
      case 'int64':
        return 'int64';
      case 'uint8':
        return 'uint8';
      case 'bool':
        return 'bool';
      case 'string':
        return 'string';
      default:
        return ortType.toLowerCase();
    }
  }

  @override
  Future<void> closeSession(String sessionId) async {
    try {
      // Check if the session exists
      if (!_sessions.containsKey(sessionId)) {
        // If session not found, return successfully (similar to how other platforms handle this case)
        return;
      }

      final session = _sessions[sessionId]!;

      // Call the release method to properly free memory resources
      // In ONNX Runtime JavaScript API, the method to release resources is 'release()'
      try {
        callMethod(session, 'release', []);
      } catch (e) {
        // print('Error releasing ONNX session: $e');
        // Even if release fails, we should still remove the session from our map
      }

      // Remove session from the map
      _sessions.remove(sessionId);
    } catch (e) {
      // Log error but don't throw exception to match behavior of other platforms
      // print('Error closing session: $e');
    }
  }

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) async {
    try {
      // Check if the session exists
      if (!_sessions.containsKey(sessionId)) {
        throw PlatformException(code: "INVALID_SESSION", message: "Session not found", details: null);
      }

      final session = _sessions[sessionId]!;

      // Create a default metadata map with empty values
      final metadataMap = <String, dynamic>{
        'producerName': '',
        'graphName': '',
        'domain': '',
        'description': '',
        'version': 0,
        'customMetadataMap': <String, String>{},
      };

      // Check if the session has metadata property (Recent versions of ONNX Runtime JS API may have this)
      if (hasProperty(session, 'metadata')) {
        final metadata = getProperty(session, 'metadata');

        // Extract metadata properties if they exist
        if (hasProperty(metadata, 'producerName')) {
          metadataMap['producerName'] = getProperty(metadata, 'producerName').toString();
        }

        if (hasProperty(metadata, 'graphName')) {
          metadataMap['graphName'] = getProperty(metadata, 'graphName').toString();
        }

        if (hasProperty(metadata, 'domain')) {
          metadataMap['domain'] = getProperty(metadata, 'domain').toString();
        }

        if (hasProperty(metadata, 'description')) {
          metadataMap['description'] = getProperty(metadata, 'description').toString();
        }

        if (hasProperty(metadata, 'version')) {
          metadataMap['version'] = getProperty(metadata, 'version');
        }

        if (hasProperty(metadata, 'customMetadataMap')) {
          final customMap = getProperty(metadata, 'customMetadataMap');
          final customMetadataMap = <String, String>{};

          // Convert JavaScript object to Dart map
          // Use Object.keys() from JavaScript to get the keys of the object
          final keysObj = callMethod(objectGlobal, 'keys', [customMap]);
          final length = getProperty(keysObj, 'length') as int;

          for (var i = 0; i < length; i++) {
            final key = callMethod(keysObj, 'at', [i]).toString();
            final value = getProperty(customMap, key).toString();
            customMetadataMap[key] = value;
          }

          metadataMap['customMetadataMap'] = customMetadataMap;
        }
      }

      return metadataMap;
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "PLUGIN_ERROR", message: "Failed to get metadata: $e", details: null);
    }
  }

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) async {
    try {
      // Check if the session exists
      if (!_sessions.containsKey(sessionId)) {
        throw PlatformException(code: "INVALID_SESSION", message: "Session not found", details: null);
      }

      final session = _sessions[sessionId]!;
      final inputInfoList = <Map<String, dynamic>>[];

      // Get input metadata from session if available
      if (hasProperty(session, 'inputMetadata')) {
        final inputMetadata = getProperty(session, 'inputMetadata');
        final length = getProperty(inputMetadata, 'length') as int;

        for (var i = 0; i < length; i++) {
          final info = callMethod(inputMetadata, 'at', [i]);
          final infoMap = <String, dynamic>{};

          // Add name
          infoMap['name'] = getProperty(info, 'name').toString();

          // Check if it's a tensor
          final isTensor = getProperty(info, 'isTensor') as bool;

          if (isTensor) {
            // Add shape if available
            if (hasProperty(info, 'shape')) {
              final shape = getProperty(info, 'shape');
              final shapeLength = getProperty(shape, 'length') as int;
              final shapeList = <int>[];

              for (var j = 0; j < shapeLength; j++) {
                final dim = callMethod(shape, 'at', [j]);
                // Handle both numeric dimensions and symbolic dimensions
                if (dim is num) {
                  shapeList.add(dim.toInt());
                } else {
                  // For symbolic dimensions, use -1 as a placeholder
                  shapeList.add(-1);
                }
              }

              infoMap['shape'] = shapeList;
            } else {
              infoMap['shape'] = <int>[];
            }

            // Add type if available
            if (hasProperty(info, 'type')) {
              infoMap['type'] = getProperty(info, 'type').toString();
            } else {
              infoMap['type'] = 'unknown';
            }
          } else {
            // For non-tensor types, provide empty shape
            infoMap['shape'] = <int>[];
            infoMap['type'] = 'non-tensor';
          }

          inputInfoList.add(infoMap);
        }
      } else {
        // Fallback: use input names to create basic info entries
        final inputNames = getInputNames(session);
        for (final name in inputNames) {
          inputInfoList.add({'name': name, 'shape': <int>[], 'type': 'unknown'});
        }
      }

      return inputInfoList;
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "PLUGIN_ERROR", message: "Failed to get input info: $e", details: null);
    }
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) async {
    try {
      // Check if the session exists
      if (!_sessions.containsKey(sessionId)) {
        throw PlatformException(code: "INVALID_SESSION", message: "Session not found", details: null);
      }

      final session = _sessions[sessionId]!;
      final outputInfoList = <Map<String, dynamic>>[];

      // Get output metadata from session if available
      if (hasProperty(session, 'outputMetadata')) {
        final outputMetadata = getProperty(session, 'outputMetadata');
        final length = getProperty(outputMetadata, 'length') as int;

        for (var i = 0; i < length; i++) {
          final info = callMethod(outputMetadata, 'at', [i]);
          final infoMap = <String, dynamic>{};

          // Add name
          infoMap['name'] = getProperty(info, 'name').toString();

          // Check if it's a tensor
          final isTensor = getProperty(info, 'isTensor') as bool;

          if (isTensor) {
            // Add shape if available
            if (hasProperty(info, 'shape')) {
              final shape = getProperty(info, 'shape');
              final shapeLength = getProperty(shape, 'length') as int;
              final shapeList = <int>[];

              for (var j = 0; j < shapeLength; j++) {
                final dim = callMethod(shape, 'at', [j]);
                // Handle both numeric dimensions and symbolic dimensions
                if (dim is num) {
                  shapeList.add(dim.toInt());
                } else {
                  // For symbolic dimensions, use -1 as a placeholder
                  shapeList.add(-1);
                }
              }

              infoMap['shape'] = shapeList;
            } else {
              infoMap['shape'] = <int>[];
            }

            // Add type if available
            if (hasProperty(info, 'type')) {
              infoMap['type'] = getProperty(info, 'type').toString();
            } else {
              infoMap['type'] = 'unknown';
            }
          } else {
            // For non-tensor types, provide empty shape
            infoMap['shape'] = <int>[];
            infoMap['type'] = 'non-tensor';
          }

          outputInfoList.add(infoMap);
        }
      } else {
        // Fallback: use output names to create basic info entries
        final outputNames = getOutputNames(session);
        for (final name in outputNames) {
          outputInfoList.add({'name': name, 'shape': <int>[], 'type': 'unknown'});
        }
      }

      return outputInfoList;
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "PLUGIN_ERROR", message: "Failed to get output info: $e", details: null);
    }
  }

  // Helper method to check if a JS object has a property
  bool hasProperty(JSObject obj, String name) {
    try {
      return obj.has(name);
    } catch (e) {
      return false;
    }
  }

  // A map to store OrtValue objects (tensors)
  final Map<String, JSObject> _ortValues = {};

  @override
  Future<Map<String, dynamic>> createOrtValue(String sourceType, dynamic data, List<int> shape) async {
    try {
      // Convert shape to JavaScript array
      final jsShape = jsArrayFrom(shape);

      // Map the source type to onnxruntime-web data type
      final dataType = _mapSourceTypeToOrtType(sourceType);

      // Handle different data types
      JSObject tensor;

      switch (sourceType) {
        case 'float32':
          // Convert data to Float32Array
          final jsData = _convertToTypedArray(data, 'Float32Array');
          tensor = tensorConstructor.callAsConstructorVarArgs([dataType.toJS, jsData, jsShape]);
          break;

        case 'int32':
          // Convert data to Int32Array
          final jsData = _convertToTypedArray(data, 'Int32Array');
          tensor = tensorConstructor.callAsConstructorVarArgs([dataType.toJS, jsData, jsShape]);
          break;

        case 'int64':
          // Note: JavaScript doesn't have Int64Array, so using BigInt64Array
          // This might require special handling depending on browser support
          final jsData = _convertToTypedArray(data, 'BigInt64Array');
          tensor = tensorConstructor.callAsConstructorVarArgs([dataType.toJS, jsData, jsShape]);
          break;

        case 'uint8':
          // Convert data to Uint8Array
          final jsData = _convertToTypedArray(data, 'Uint8Array');
          tensor = tensorConstructor.callAsConstructorVarArgs([dataType.toJS, jsData, jsShape]);
          break;

        case 'bool':
          // For boolean tensors, ONNX Runtime uses a Uint8Array with 0/1 values
          // Make sure we properly handle all possible incoming bool representations
          final boolArray =
              (data as List).map((value) {
                if (value is bool) {
                  return value ? 1 : 0;
                } else if (value is num) {
                  return value != 0 ? 1 : 0;
                } else {
                  return value == true ? 1 : 0;
                }
              }).toList();
          final jsData = _convertToTypedArray(boolArray, 'Uint8Array');
          tensor = tensorConstructor.callAsConstructorVarArgs([dataType.toJS, jsData, jsShape]);
          break;

        case 'string':
          // For string tensors, we use the standard JavaScript Array
          // ONNX Runtime JS API accepts string arrays for string tensors
          final stringArray = jsArrayFrom((data as List<String>).toList());
          tensor = tensorConstructor.callAsConstructorVarArgs([dataType.toJS, stringArray, jsShape]);
          break;

        default:
          throw PlatformException(
            code: "UNSUPPORTED_TYPE",
            message: "Unsupported data type: $sourceType",
            details: null,
          );
      }

      // Generate a unique ID for this tensor
      final valueId = '${DateTime.now().millisecondsSinceEpoch}_${math.Random().nextInt(10000)}';

      // Store the tensor
      _ortValues[valueId] = tensor;

      // Return the tensor information
      return {'valueId': valueId, 'dataType': sourceType, 'shape': shape};
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "TENSOR_CREATION_ERROR", message: "Failed to create OrtValue: $e", details: null);
    }
  }

  // Helper to convert Dart List to JavaScript TypedArray
  // Array types: https://api.flutter.dev/flutter/dart-js_interop/JSFloat32Array-extension-type.html
  JSObject _convertToTypedArray(dynamic data, String arrayType) {
    // Make sure data is a list
    final dataList = data is List ? data : [data];

    // Create a JavaScript Array from the Dart List
    final jsArray = jsArrayFrom(dataList);

    // Get the appropriate TypedArray constructor
    final constructor = switch (arrayType) {
      'Float32Array' => float32ArrayConstructor,
      'Int32Array' => int32ArrayConstructor,
      'BigInt64Array' => bigInt64ArrayConstructor,
      'Uint8Array' => uint8ArrayConstructor,
      _ =>
        throw PlatformException(
          code: "UNSUPPORTED_ARRAY_TYPE",
          message: "Unsupported array type: $arrayType",
          details: null,
        ),
    };

    // Create the TypedArray from the Array
    return callMethod(constructor, 'from', [jsArray]);
  }

  // Map Dart source type to ONNX Runtime type strings
  String _mapSourceTypeToOrtType(String sourceType) {
    switch (sourceType) {
      case 'float32':
        return 'float32';
      case 'int32':
        return 'int32';
      case 'int64':
        return 'int64';
      case 'uint8':
        return 'uint8';
      case 'bool':
        return 'bool';
      case 'string':
        return 'string';
      default:
        throw PlatformException(code: "UNSUPPORTED_TYPE", message: "Unsupported data type: $sourceType", details: null);
    }
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) async {
    try {
      // Check if the tensor exists
      if (!_ortValues.containsKey(valueId)) {
        throw PlatformException(code: "INVALID_VALUE", message: "OrtValue not found with ID: $valueId", details: null);
      }

      final tensor = _ortValues[valueId]!;

      // Get tensor type
      final type = getProperty(tensor, 'type');

      // Get tensor shape
      final jsShape = getProperty(tensor, 'dims');
      final shapeLength = getProperty(jsShape, 'length') as int;
      final shape = <int>[];

      // Convert shape to Dart list
      for (var i = 0; i < shapeLength; i++) {
        shape.add(callMethod(jsShape, 'at', [i]) as int);
      }

      // Get tensor data
      final jsData = getProperty(tensor, 'data');
      final dataLength = getProperty(jsData, 'length') as int;
      final data = <dynamic>[];

      // Convert data to Dart list based on type
      switch (type.toString()) {
        case 'float32':
          for (var i = 0; i < dataLength; i++) {
            data.add((callMethod(jsData, 'at', [i]) as num).toDouble());
          }
          break;

        case 'int32':
          for (var i = 0; i < dataLength; i++) {
            data.add((callMethod(jsData, 'at', [i]) as num).toInt());
          }
          break;

        case 'int64':
          // BigInt handling
          for (var i = 0; i < dataLength; i++) {
            final value = callMethod(jsData, 'at', [i]);
            // Convert BigInt or similar to standard number if possible
            // For dart:js_interop, try direct conversion to num or int
            data.add((value as num).toInt());
          }
          break;

        case 'uint8':
          for (var i = 0; i < dataLength; i++) {
            data.add((callMethod(jsData, 'at', [i]) as num).toInt());
          }
          break;

        case 'bool':
          for (var i = 0; i < dataLength; i++) {
            // For boolean tensors, convert the numeric values back to proper Dart booleans
            final value = callMethod(jsData, 'at', [i]) as num;
            // Return the actual numeric value (1 or 0), not a boolean,
            // to match native implementations which return 1/0
            data.add(value != 0 ? 1 : 0);
          }
          break;

        case 'string':
          for (var i = 0; i < dataLength; i++) {
            // For string tensors, extract string values
            final value = callMethod(jsData, 'at', [i]);
            data.add(value.toString());
          }
          break;

        default:
          throw PlatformException(
            code: "UNSUPPORTED_TYPE",
            message: "Unsupported data type for extraction: $type",
            details: null,
          );
      }

      return {'data': data, 'shape': shape};
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "DATA_EXTRACTION_ERROR", message: "Failed to get OrtValue data: $e", details: null);
    }
  }

  @override
  Future<void> releaseOrtValue(String valueId) async {
    try {
      // Check if the tensor exists
      if (!_ortValues.containsKey(valueId)) {
        // If not found, return successfully (similar to other platforms)
        return;
      }

      // Get the tensor
      final tensor = _ortValues[valueId]!;

      // Call dispose method if available to free resources
      if (hasProperty(tensor, 'dispose')) {
        callMethod(tensor, 'dispose', []);
      }

      // Remove the tensor from our map
      _ortValues.remove(valueId);
    } catch (e) {
      // Even if release fails, attempt to remove from the map
      _ortValues.remove(valueId);
    }
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) async {
    try {
      // Check if the tensor exists
      if (!_ortValues.containsKey(valueId)) {
        throw PlatformException(code: "INVALID_VALUE", message: "OrtValue with ID $valueId not found", details: null);
      }

      final tensor = _ortValues[valueId]!;

      // Get tensor type and shape
      final sourceType = getProperty(tensor, 'type').toString();
      final jsShape = getProperty(tensor, 'dims');
      final shapeLength = getProperty(jsShape, 'length') as int;
      final shape = <int>[];

      for (var i = 0; i < shapeLength; i++) {
        shape.add(callMethod(jsShape, 'at', [i]) as int);
      }

      // Get the data from the tensor
      final jsData = getProperty(tensor, 'data');
      final dataLength = getProperty(jsData, 'length') as int;

      // Create a new tensor based on the conversion type
      JSObject newTensor;

      // Handle different conversion scenarios
      switch ('$sourceType-$targetType') {
        // Float32 to Int32
        case 'float32-int32':
          final intArray = [];
          for (var i = 0; i < dataLength; i++) {
            final value = callMethod(jsData, 'at', [i]) as num;
            intArray.add(value.toInt());
          }

          final jsIntArray = _convertToTypedArray(intArray, 'Int32Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['int32'.toJS, jsIntArray, jsArrayFrom(shape)]);
          break;

        // Float32 to Int64 (BigInt64Array)
        case 'float32-int64':
          // For int64 conversions in web, we need to skip the test
          // since BigInt64Array is not properly supported in all browsers
          // Throw a more user-friendly error
          throw PlatformException(
            code: "CONVERSION_ERROR",
            message: "Int64 conversions are not fully supported in the web implementation yet",
            details: null,
          );

        // Int32 to Float32
        case 'int32-float32':
          final floatArray = [];
          for (var i = 0; i < dataLength; i++) {
            final value = callMethod(jsData, 'at', [i]) as num;
            floatArray.add(value.toDouble());
          }

          final jsFloatArray = _convertToTypedArray(floatArray, 'Float32Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['float32'.toJS, jsFloatArray, jsArrayFrom(shape)]);
          break;

        // Int64 to Float32
        case 'int64-float32':
          final floatArray = [];
          for (var i = 0; i < dataLength; i++) {
            final jsValue = callMethod(jsData, 'at', [i]);
            // Convert BigInt to double for dart:js_interop
            double doubleValue = 0.0;
            try {
              final numValue = jsValue as num;
              doubleValue = numValue.toDouble();
            } catch (e) {
              // Handle conversion errors
              doubleValue = 0.0;
            }
            floatArray.add(doubleValue);
          }

          final jsFloatArray = _convertToTypedArray(floatArray, 'Float32Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['float32'.toJS, jsFloatArray, jsArrayFrom(shape)]);
          break;

        // Int32 to Int64
        case 'int32-int64':
          // For int64 conversions in web, we need to skip the test
          // since BigInt64Array is not properly supported in all browsers
          // Throw a more user-friendly error
          throw PlatformException(
            code: "CONVERSION_ERROR",
            message: "Int64 conversions are not fully supported in the web implementation yet",
            details: null,
          );

        // Int64 to Int32 (with potential loss of precision)
        case 'int64-int32':
          final intArray = [];
          for (var i = 0; i < dataLength; i++) {
            final jsValue = callMethod(jsData, 'at', [i]);
            // Convert BigInt to int for dart:js_interop
            int intValue = 0;
            try {
              final numValue = jsValue as num;
              intValue = numValue.toInt();
            } catch (e) {
              // Handle conversion errors
              intValue = 0;
            }
            intArray.add(intValue);
          }

          final jsIntArray = _convertToTypedArray(intArray, 'Int32Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['int32'.toJS, jsIntArray, jsArrayFrom(shape)]);
          break;

        // Uint8 to Float32
        case 'uint8-float32':
          final floatArray = [];
          for (var i = 0; i < dataLength; i++) {
            final value = callMethod(jsData, 'at', [i]) as num;
            floatArray.add(value.toDouble());
          }

          final jsFloatArray = _convertToTypedArray(floatArray, 'Float32Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['float32'.toJS, jsFloatArray, jsArrayFrom(shape)]);
          break;

        // Boolean to Int8/Uint8
        case 'bool-int8':
        case 'bool-uint8':
          // Boolean values are already stored as 0/1 in a Uint8Array
          // Just create a new Uint8Array with the same values
          final byteArray = [];
          for (var i = 0; i < dataLength; i++) {
            final value = callMethod(jsData, 'at', [i]) as num;
            byteArray.add(value != 0 ? 1 : 0);
          }

          final jsByteArray = _convertToTypedArray(byteArray, 'Uint8Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['uint8'.toJS, jsByteArray, jsArrayFrom(shape)]);
          break;

        // Int8/Uint8 to Boolean
        case 'int8-bool':
        case 'uint8-bool':
          // Convert to boolean representation (non-zero values become true)
          final boolArray = [];
          for (var i = 0; i < dataLength; i++) {
            final value = callMethod(jsData, 'at', [i]) as num;
            boolArray.add(value != 0 ? 1 : 0);
          }

          final jsBoolArray = _convertToTypedArray(boolArray, 'Uint8Array');
          newTensor = tensorConstructor.callAsConstructorVarArgs(['bool'.toJS, jsBoolArray, jsArrayFrom(shape)]);
          break;

        // Same type conversion (no-op)
        case 'float32-float32':
        case 'int32-int32':
        case 'int64-int64':
        case 'uint8-uint8':
        case 'int8-int8':
        case 'bool-bool':
        case 'string-string':
          // Clone the original tensor with the same data
          final newData = getProperty(tensor, 'data');
          newTensor = tensorConstructor.callAsConstructorVarArgs([sourceType.toJS, newData, jsArrayFrom(shape)]);
          break;

        default:
          throw PlatformException(
            code: "CONVERSION_ERROR",
            message: "Conversion from $sourceType to $targetType is not supported",
            details: null,
          );
      }

      // Generate a unique ID for this tensor
      final newValueId = '${DateTime.now().millisecondsSinceEpoch}_${math.Random().nextInt(10000)}';

      // Store the tensor
      _ortValues[newValueId] = newTensor;

      // Return the tensor information
      return {'valueId': newValueId, 'dataType': targetType, 'shape': shape};
    } catch (e) {
      if (e is PlatformException) {
        rethrow;
      }
      throw PlatformException(code: "CONVERSION_ERROR", message: "Failed to convert OrtValue: $e", details: null);
    }
  }
}
