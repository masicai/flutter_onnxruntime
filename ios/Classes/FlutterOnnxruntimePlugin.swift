import Flutter
import UIKit
import onnxruntime_objc
import Foundation

enum OrtError: Error {
    case flutterError(FlutterError)
}

// swiftlint:disable:next type_body_length
public class FlutterOnnxruntimePlugin: NSObject, FlutterPlugin {
  private var sessions = [String: ORTSession]()
  private var env: ORTEnv?

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "flutter_onnxruntime", binaryMessenger: registrar.messenger())
    let instance = FlutterOnnxruntimePlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  // swiftlint:disable:next cyclomatic_complexity
  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    if env == nil {
      do {
        env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
      } catch {
        result(FlutterError(code: "ENV_INIT_FAILED", message: error.localizedDescription, details: nil))
        return
      }
    }

    switch call.method {
    case "getPlatformVersion":
      result("iOS " + UIDevice.current.systemVersion)

    /** Create a new session

      Create a new session from a model file path.

      Reference: https://onnxruntime.ai/docs/api/objectivec/Classes/ORTSession.html
    */
    case "createSession":
      handleCreateSession(call: call, result: result)
    case "runInference":
      handleRunInference(call: call, result: result)
    case "closeSession":
      handleCloseSession(call: call, result: result)
    case "getMetadata":
      handleGetMetadata(call: call, result: result)
    case "getInputInfo":
      handleGetInputInfo(call: call, result: result)
    case "getOutputInfo":
      handleGetOutputInfo(call: call, result: result)
    case "createOrtValue":
      handleCreateOrtValue(call, result: result)
    case "convertOrtValue":
      handleConvertOrtValue(call, result: result)
    case "moveOrtValueToDevice":
      handleMoveOrtValueToDevice(call, result: result)
    case "getOrtValueData":
      handleGetOrtValueData(call, result: result)
    case "releaseOrtValue":
      handleReleaseOrtValue(call, result: result)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func handleCreateSession(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Model path is required", details: nil))
      return
    }

    do {
      let sessionOptions = try ORTSessionOptions()

      if let options = args["sessionOptions"] as? [String: Any] {
        if let intraOpNumThreads = options["intraOpNumThreads"] as? Int {
          try sessionOptions.setIntraOpNumThreads(Int32(intraOpNumThreads))
        }

        // if let interOpNumThreads = options["interOpNumThreads"] as? Int {
        //   try sessionOptions.setInterOpNumThreads(Int32(interOpNumThreads))
        // }

        // if let enableCpuMemArena = options["enableCpuMemArena"] as? Bool {
        //   sessionOptions.enableCPUMemArena = enableCpuMemArena
        // }
      }

      // Check if file exists
      let fileManager = FileManager.default
      if !fileManager.fileExists(atPath: modelPath) {
        result(FlutterError(code: "FILE_NOT_FOUND", message: "Model file not found at path: \(modelPath)", details: nil))
        return
      }

      // Create session from file path
      guard let safeEnv = env else {
        result(FlutterError(code: "ENV_NOT_INITIALIZED", message: "ONNX Runtime environment not initialized", details: nil))
        return
      }

      let session = try ORTSession(env: safeEnv, modelPath: modelPath, sessionOptions: sessionOptions)
      let sessionId = UUID().uuidString
      sessions[sessionId] = session

      // Get input and output names
      var inputNames: [String] = []
      var outputNames: [String] = []

      // Get input names
      if let inputNodeNames = try? session.inputNames() {
        inputNames = inputNodeNames
      }

      // Get output names
      if let outputNodeNames = try? session.outputNames() {
        outputNames = outputNodeNames
      }

      let responseMap: [String: Any] = [
        "sessionId": sessionId,
        "inputNames": inputNames,
        "outputNames": outputNames
      ]

      result(responseMap)
    } catch {
      result(FlutterError(code: "SESSION_CREATION_FAILED", message: error.localizedDescription, details: nil))
    }
  }

  private func handleRunInference(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sessionId = args["sessionId"] as? String,
          let inputs = args["inputs"] as? [String: Any] else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }

    // Get run options if provided
    // TODO: process and pass the runOptions to session.run()
    let runOptions = args["runOptions"] as? [String: Any] ?? [:]

    do {
      // Get session
      guard let session = sessions[sessionId] else {
        throw OrtError.flutterError(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
      }

      // Process inputs - validate OrtValue references directly here
      var ortInputs: [String: ORTValue] = [:]

      for (name, value) in inputs {
        // Only process OrtValue references (sent as dictionary with valueId)
        if let valueDict = value as? [String: Any], let valueId = valueDict["valueId"] as? String {
          if let existingValue = ortValues[valueId] {
            ortInputs[name] = existingValue
          } else {
            throw OrtError.flutterError(FlutterError(code: "INVALID_ORT_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
          }
        } else {
          throw OrtError.flutterError(FlutterError(code: "INVALID_INPUT_FORMAT",
            message: "Input for '\(name)' must be an OrtValue reference with valueId",
            details: nil))
        }
      }

      // Get output names
      let outputNames = try session.outputNames()

      // Run inference with prepared output containers
      let outputs = try session.run(withInputs: ortInputs, outputNames: Set(outputNames), runOptions: nil)

      // store outputs in ortValues dictionary and return metadata in Flutter format
      var flutterOutputs: [String: Any] = [:]
      for (outputName, outputTensor) in outputs {
        let valueId = UUID().uuidString
        ortValues[valueId] = outputTensor
        let tensorInfo = try outputTensor.tensorTypeAndShapeInfo()
        // set an array of [valueId, dataType, shape]
        let shape = try tensorInfo.shape.map { Int($0) }
        // get the element type as string
        let typeName = _getDataTypeName(from: tensorInfo.elementType)
        flutterOutputs[outputName] = [valueId, typeName, shape]
      }
      // Return result
      result(flutterOutputs)
    } catch let error as FlutterError {
      result(error)
    } catch {
      result(FlutterError(code: "RUN_INFERENCE_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func handleCloseSession(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sessionId = args["sessionId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
      return
    }

    if sessions.removeValue(forKey: sessionId) != nil {
      result(nil)
    } else {
      result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
    }
  }

  private func handleGetMetadata(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sessionId = args["sessionId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
      return
    }

    guard let session = sessions[sessionId] else {
      result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
      return
    }

    // Note: 06/04/2025 on v1.21.0 session.getMetadata() is not supported in onnxruntime-objc
    // do {
    //   let modelMetadata = try session.getMetadata()

    //   let metadataMap: [String: Any] = [
    //     "producerName": modelMetadata.producerName ?? "",
    //     "graphName": modelMetadata.graphName ?? "",
    //     "domain": modelMetadata.domain ?? "",
    //     "description": modelMetadata.description ?? "",
    //     "version": modelMetadata.version,
    //     "customMetadataMap": modelMetadata.customMetadata ?? [:]
    //   ]

    //   result(metadataMap)
    // } catch {
    //   result(FlutterError(code: "METADATA_ERROR", message: error.localizedDescription, details: nil))
    // }

    // return empty map
    result([:])
  }

  private func handleGetInputInfo(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sessionId = args["sessionId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
      return
    }

    guard let session = sessions[sessionId] else {
      result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
      return
    }

    do {
      var nodeInfoList: [[String: Any]] = []

      let inputNames = try session.inputNames()
      // Note: 06/04/2025 on v1.21.0 session.getInputInfo() is not supported in onnxruntime-objc
      // let inputInfoMap = try session.getInputInfo()

      for name in inputNames {
        let infoMap: [String: Any] = ["name": name]
        nodeInfoList.append(infoMap)
      }

      result(nodeInfoList)
    } catch {
      result(FlutterError(code: "INPUT_INFO_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func handleGetOutputInfo(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sessionId = args["sessionId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
      return
    }

    guard let session = sessions[sessionId] else {
      result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
      return
    }

    do {
      var nodeInfoList: [[String: Any]] = []

      let outputNames = try session.outputNames()
      // Note: 06/04/2025 on v1.21.0 session.getOutputInfo() is not supported in onnxruntime-objc
      // let outputInfoMap = try session.getOutputInfo()

      for name in outputNames {
        let infoMap: [String: Any] = ["name": name]
        nodeInfoList.append(infoMap)
      }

      result(nodeInfoList)
    } catch {
      result(FlutterError(code: "OUTPUT_INFO_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  // MARK: - OrtValue Management

  private var ortValues: [String: ORTValue] = [:]

  // swiftlint:disable:next cyclomatic_complexity
  private func handleCreateOrtValue(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sourceType = args["sourceType"] as? String,
          let data = args["data"],
          let shape = args["shape"] as? [Int] else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }

    // Convert shape to NSNumber array for ORTValue
    let shapeNumbers = shape.map { NSNumber(value: $0) }

    do {
      // Create tensor based on source data type
      var tensor: ORTValue

      switch sourceType {
      case "float32":
        if let floatArray = data as? [Float] {
          // Create float tensor
          let data = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .float, shape: shapeNumbers)
        } else if let doubleArray = data as? [Double] {
          // Convert double to float
          let floatArray = doubleArray.map { Float($0) }
          let data = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .float, shape: shapeNumbers)
        } else if let anyArray = data as? [Any] {
          // Try to convert Any array to Float array
          let floatArray = try anyArray.map { value -> Float in
            if let number = value as? NSNumber {
              return number.floatValue
            } else {
              throw OrtError.flutterError(FlutterError(code: "CONVERSION_ERROR", message: "Cannot convert \(type(of: value)) to Float", details: nil))
            }
          }
          let data = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .float, shape: shapeNumbers)
        } else if let typedData = data as? FlutterStandardTypedData {
          // Handle FlutterStandardTypedData
          if typedData.data.count % 4 == 0 {
            let mutableData = NSMutableData(data: typedData.data)
            tensor = try ORTValue(tensorData: mutableData, elementType: .float, shape: shapeNumbers)
          } else if typedData.data.count % 8 == 0 {
            // Could be Float64 data, convert to Float32
            let float64Count = typedData.data.count / 8

            var float32Array = [Float](repeating: 0.0, count: float64Count)

            // Extract Float64 values and convert to Float32
            typedData.data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
              let float64Buffer = buffer.bindMemory(to: Float64.self)
              for index in 0..<float64Count {
                float32Array[index] = Float(float64Buffer[index])
              }
            }

            let float32Data = NSMutableData(bytes: float32Array, length: float32Array.count * MemoryLayout<Float>.stride)
            tensor = try ORTValue(tensorData: float32Data, elementType: .float, shape: shapeNumbers)
          } else {
            result(FlutterError(code: "INVALID_DATA_TYPE",
                               message: "Data size \(typedData.data.count) is not consistent with Float32 or Float64 data",
                               details: nil))
            return
          }
        } else {
          result(FlutterError(code: "INVALID_DATA",
                             message: "Data must be a list of numbers for float32 type. Received: \(type(of: data))",
                             details: nil))
          return
        }

      case "int32":
        if let intArray = data as? [Int32] {
          let data = NSMutableData(bytes: intArray, length: intArray.count * MemoryLayout<Int32>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .int32, shape: shapeNumbers)
        } else if let intArray = data as? [Int] {
          let int32Array = intArray.map { Int32($0) }
          let data = NSMutableData(bytes: int32Array, length: int32Array.count * MemoryLayout<Int32>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .int32, shape: shapeNumbers)
        } else if let typedData = data as? FlutterStandardTypedData {
          // Handle FlutterStandardTypedData for Int32
          if typedData.data.count % 4 == 0 {
            let int32Count = typedData.data.count / 4
            var int32Array = [Int32](repeating: 0, count: int32Count)

            typedData.data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
              let int32Buffer = buffer.bindMemory(to: Int32.self)
              for index in 0..<int32Count {
                int32Array[index] = int32Buffer[index]
              }
            }

            let int32Data = NSMutableData(bytes: int32Array, length: int32Array.count * MemoryLayout<Int32>.stride)
            tensor = try ORTValue(tensorData: int32Data, elementType: .int32, shape: shapeNumbers)
          } else {
            result(FlutterError(code: "INVALID_DATA_TYPE",
                               message: "Data size \(typedData.data.count) is not consistent with Int32 data",
                               details: nil))
            return
          }
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for int32 type", details: nil))
          return
        }

      case "int64":
        if let longArray = data as? [Int64] {
          let data = NSMutableData(bytes: longArray, length: longArray.count * MemoryLayout<Int64>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .int64, shape: shapeNumbers)
        } else if let intArray = data as? [Int] {
          let int64Array = intArray.map { Int64($0) }
          let data = NSMutableData(bytes: int64Array, length: int64Array.count * MemoryLayout<Int64>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .int64, shape: shapeNumbers)
        } else if let typedData = data as? FlutterStandardTypedData {
          // Handle FlutterStandardTypedData for Int64
          if typedData.data.count % 8 == 0 {
            let int64Count = typedData.data.count / 8
            var int64Array = [Int64](repeating: 0, count: int64Count)

            typedData.data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
              let int64Buffer = buffer.bindMemory(to: Int64.self)
              for index in 0..<int64Count {
                int64Array[index] = int64Buffer[index]
              }
            }

            let int64Data = NSMutableData(bytes: int64Array, length: int64Array.count * MemoryLayout<Int64>.stride)
            tensor = try ORTValue(tensorData: int64Data, elementType: .int64, shape: shapeNumbers)
          } else {
            result(FlutterError(code: "INVALID_DATA_TYPE",
                               message: "Data size \(typedData.data.count) is not consistent with Int64 data",
                               details: nil))
            return
          }
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for int64 type", details: nil))
          return
        }

      case "uint8":
        if let uintArray = data as? [UInt8] {
          let data = NSMutableData(bytes: uintArray, length: uintArray.count * MemoryLayout<UInt8>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .uInt8, shape: shapeNumbers)
        } else if let intArray = data as? [Int] {
          let uintArray = intArray.map { UInt8($0) }
          let data = NSMutableData(bytes: uintArray, length: uintArray.count * MemoryLayout<UInt8>.stride)
          tensor = try ORTValue(tensorData: data, elementType: .uInt8, shape: shapeNumbers)
        } else if let typedData = data as? FlutterStandardTypedData {
          // Handle FlutterStandardTypedData for UInt8 - directly use the data for byte arrays
          let uint8Data = NSMutableData(data: typedData.data)
          tensor = try ORTValue(tensorData: uint8Data, elementType: .uInt8, shape: shapeNumbers)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for uint8 type", details: nil))
          return
        }

      case "bool":
        if let boolArray = data as? [Bool] {
          // Convert bool array to UInt8 array (1 for true, 0 for false)
          let uint8Array = boolArray.map { $0 ? UInt8(1) : UInt8(0) }
          let data = NSMutableData(bytes: uint8Array, length: uint8Array.count * MemoryLayout<UInt8>.stride)
          // Use uint8 type since bool is not available in ORTTensorElementDataType
          tensor = try ORTValue(tensorData: data, elementType: .uInt8, shape: shapeNumbers)
        } else if let typedData = data as? FlutterStandardTypedData {
          // For bool values, assume they are stored as bytes, where non-zero is true
          let uint8Data = NSMutableData(data: typedData.data)
          tensor = try ORTValue(tensorData: uint8Data, elementType: .uInt8, shape: shapeNumbers)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of booleans for bool type", details: nil))
          return
        }

      default:
        result(FlutterError(code: "UNSUPPORTED_TYPE", message: "Unsupported source data type: \(sourceType)", details: nil))
        return
      }

      // Generate unique ID for the tensor
      let valueId = UUID().uuidString
      ortValues[valueId] = tensor

      // Return tensor information
      let tensorInfo: [String: Any] = [
        "valueId": valueId,
        "dataType": sourceType,
        "shape": shape,
        "device": "cpu"
      ]

      result(tensorInfo)
    } catch {
      result(FlutterError(code: "TENSOR_CREATION_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func handleConvertOrtValue(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String,
          let targetType = args["targetType"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }

    guard let tensor = ortValues[valueId] else {
      result(FlutterError(code: "INVALID_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
      return
    }

    do {
      // Get tensor information
      let tensorInfo = try tensor.tensorTypeAndShapeInfo()
      let shape = try tensorInfo.shape.map { Int(truncating: $0) }

      // In ORTValue, direct type conversion isn't available
      // We would need to extract data and create a new tensor
      // For now, just return the same tensor with a new ID

      // Store the tensor with a new ID
      let newValueId = UUID().uuidString
      ortValues[newValueId] = tensor

      // Return tensor information
      let resultInfo: [String: Any] = [
        "valueId": newValueId,
        "dataType": targetType,
        "shape": shape,
        "device": "cpu" // ORTValue only supports CPU in this implementation
      ]

      result(resultInfo)
    } catch {
      result(FlutterError(code: "CONVERSION_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func handleMoveOrtValueToDevice(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String,
          let targetDevice = args["targetDevice"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }

    guard let tensor = ortValues[valueId] else {
      result(FlutterError(code: "INVALID_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
      return
    }

    // Currently, we only support CPU device in this implementation
    if targetDevice != "cpu" {
      result(FlutterError(code: "UNSUPPORTED_DEVICE", message: "Only CPU device is supported in this implementation", details: nil))
      return
    }

    do {
      // Get tensor information
      let tensorInfo = try tensor.tensorTypeAndShapeInfo()
      let shape = try tensorInfo.shape.map { Int(truncating: $0) }

      // Return tensor information (no actual device transfer needed as we're staying on CPU)
      let resultInfo: [String: Any] = [
        "valueId": valueId,
        "dataType": _getDataTypeName(from: tensorInfo.elementType),
        "shape": shape,
        "device": "cpu"
      ]

      result(resultInfo)
    } catch {
      result(FlutterError(code: "DEVICE_TRANSFER_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  // swiftlint:disable:next cyclomatic_complexity
  private func handleGetOrtValueData(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing valueId", details: nil))
      return
    }

    guard let tensor = ortValues[valueId] else {
      result(FlutterError(code: "INVALID_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
      return
    }

    do {
      // Get tensor information
      let tensorInfo = try tensor.tensorTypeAndShapeInfo()
      let shape = try tensorInfo.shape.map { Int(truncating: $0) }
      let elementCount = shape.reduce(1, *)
      var data: Any
      let dataType = _getDataTypeName(from: tensorInfo.elementType)

      // Extract data based on the tensor's native type
      switch tensorInfo.elementType {
      case .float:
        let dataPtr = try tensor.tensorData()
        let floatPtr = dataPtr.bytes.bindMemory(to: Float.self, capacity: elementCount)
        let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: elementCount)
        data = Array(floatBuffer)

      case .int32:
        let dataPtr = try tensor.tensorData()
        let intPtr = dataPtr.bytes.bindMemory(to: Int32.self, capacity: elementCount)
        let intBuffer = UnsafeBufferPointer(start: intPtr, count: elementCount)
        data = Array(intBuffer)

      case .int64:
        let dataPtr = try tensor.tensorData()
        let int64Ptr = dataPtr.bytes.bindMemory(to: Int64.self, capacity: elementCount)
        let int64Buffer = UnsafeBufferPointer(start: int64Ptr, count: elementCount)
        data = Array(int64Buffer)

      case .uInt8:
        let dataPtr = try tensor.tensorData()
        let uint8Ptr = dataPtr.bytes.bindMemory(to: UInt8.self, capacity: elementCount)
        let uint8Buffer = UnsafeBufferPointer(start: uint8Ptr, count: elementCount)
        data = Array(uint8Buffer)

      case .int8:
        let dataPtr = try tensor.tensorData()
        let int8Ptr = dataPtr.bytes.bindMemory(to: Int8.self, capacity: elementCount)
        let int8Buffer = UnsafeBufferPointer(start: int8Ptr, count: elementCount)
        data = Array(int8Buffer)

      default:
        // Try to extract as float for unsupported types
        let dataPtr = try tensor.tensorData()
        let floatPtr = dataPtr.bytes.bindMemory(to: Float.self, capacity: elementCount)
        let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: elementCount)
        data = Array(floatBuffer)
      }

      // Also include the data type in the result
      let resultMap: [String: Any] = [
        "data": data,
        "shape": shape,
        "dataType": dataType
      ]

      result(resultMap)
    } catch {
      result(FlutterError(code: "DATA_EXTRACTION_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func handleReleaseOrtValue(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing value ID", details: nil))
      return
    }

    // Remove and release tensor
    ortValues.removeValue(forKey: valueId)

    result(nil)
  }

  // Helper function to convert ORTTensorElementDataType to string
  private func _getDataTypeName(from type: ORTTensorElementDataType) -> String {
    switch type {
    case .float: return "float32"
    case .int32: return "int32"
    case .int64: return "int64"
    case .uInt8: return "uint8"
    case .int8: return "int8"
    case .string: return "string"
    // ORTTensorElementDataType doesn't have a bool type, we use uint8 for boolean data
    default: return "unknown"
    }
  }
}
