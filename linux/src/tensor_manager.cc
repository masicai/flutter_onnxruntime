// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "tensor_manager.h"
#include "value_conversion.h"

TensorManager::TensorManager()
    : next_tensor_id_(1), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

TensorManager::~TensorManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  tensors_.clear();
  tensor_types_.clear();
  tensor_shapes_.clear();
}

std::string TensorManager::generateTensorId() { return "tensor_" + std::to_string(next_tensor_id_++); }

std::string TensorManager::createFloat32Tensor(const std::vector<float> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // make a more robust copy, avoid the delocation of the original data
    float *tensor_data = new float[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<float>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    // Following RAII principles, use std::make_unique to tie the OrtValue lifetime to the pointer
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "float32";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    throw;
  }
}

std::string TensorManager::createInt32Tensor(const std::vector<int32_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Make a robust copy of the data
    int32_t *tensor_data = new int32_t[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<int32_t>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "int32";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    throw;
  }
}

std::string TensorManager::createInt64Tensor(const std::vector<int64_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Make a robust copy of the data
    int64_t *tensor_data = new int64_t[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "int64";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    throw;
  }
}

std::string TensorManager::createUint8Tensor(const std::vector<uint8_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Make a robust copy of the data
    uint8_t *tensor_data = new uint8_t[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<uint8_t>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "uint8";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    throw;
  }
}

std::string TensorManager::createBoolTensor(const std::vector<bool> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Create a regular array for the boolean data (std::vector<bool> is specialized and can't be used directly)
    bool *tensor_data = new bool[data.size()];
    for (size_t i = 0; i < data.size(); i++) {
      tensor_data[i] = data[i];
    }
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<bool>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "bool";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    throw;
  }
}

FlValue *TensorManager::getTensorData(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the tensor exists
  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);

  if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
    // Tensor not found
    return fl_value_new_null();
  }

  // Create result map
  g_autoptr(FlValue) result = fl_value_new_map();

  try {
    // Get tensor type
    const std::string &tensor_type = type_it->second;

    // Get tensor shape
    const std::vector<int64_t> &shape = shape_it->second;

    // Convert shape to FlValue
    FlValue *shape_list = fl_value_new_list();
    for (const auto &dim : shape) {
      fl_value_append_take(shape_list, fl_value_new_int(static_cast<int64_t>(dim)));
    }

    // Set shape and type in result
    fl_value_set_string_take(result, "shape", shape_list);
    fl_value_set_string_take(result, "dataType", fl_value_new_string(tensor_type.c_str()));

    // Handle different tensor types
    if (tensor_type == "float32") {
      // Get float data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      float *tensor_data = tensor->GetTensorMutableData<float>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<float> data_vec(tensor_data, tensor_data + elem_count);
      FlValue *data_list = vector_to_fl_value(data_vec);

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "int32") {
      // Get int32 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      int32_t *tensor_data = tensor->GetTensorMutableData<int32_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<int32_t> data_vec(tensor_data, tensor_data + elem_count);
      // Convert to int for Flutter
      std::vector<int> int_data_vec(data_vec.begin(), data_vec.end());
      FlValue *data_list = vector_to_fl_value(int_data_vec);

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "int64") {
      // Get int64 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      int64_t *tensor_data = tensor->GetTensorMutableData<int64_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<int64_t> data_vec(tensor_data, tensor_data + elem_count);
      FlValue *data_list = fl_value_new_list();
      for (const auto &val : data_vec) {
        fl_value_append_take(data_list, fl_value_new_int(val));
      }

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "uint8") {
      // Get uint8 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      uint8_t *tensor_data = tensor->GetTensorMutableData<uint8_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<uint8_t> data_vec(tensor_data, tensor_data + elem_count);
      FlValue *data_list = fl_value_new_list();
      for (const auto &val : data_vec) {
        fl_value_append_take(data_list, fl_value_new_int(val));
      }

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "bool") {
      // Get bool data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      bool *tensor_data = tensor->GetTensorMutableData<bool>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values - convert bool to int for Flutter compatibility
      FlValue *data_list = fl_value_new_list();
      for (size_t i = 0; i < elem_count; i++) {
        fl_value_append_take(data_list, fl_value_new_int(tensor_data[i] ? 1 : 0));
      }

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else {
      // Unsupported tensor type
      throw std::runtime_error("Unsupported tensor type: " + tensor_type);
    }
  } catch (const Ort::Exception &e) {
    throw std::runtime_error(e.what());
  }

  return fl_value_ref(result);
}

bool TensorManager::releaseTensor(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);

  if (tensor_it == tensors_.end()) {
    return false;
  }

  // Remove tensor, type, and shape
  tensors_.erase(tensor_it);
  if (type_it != tensor_types_.end()) {
    tensor_types_.erase(type_it);
  }
  if (shape_it != tensor_shapes_.end()) {
    tensor_shapes_.erase(shape_it);
  }

  return true;
}

Ort::Value *TensorManager::getTensor(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = tensors_.find(tensor_id);
  if (it == tensors_.end()) {
    return nullptr;
  }

  return it->second.get();
}

void TensorManager::storeTensor(const std::string &tensor_id, Ort::Value &&tensor) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Store the tensor
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));

    // Get tensor info to store type and shape
    Ort::TensorTypeAndShapeInfo tensor_info = tensors_[tensor_id]->GetTensorTypeAndShapeInfo();

    // Get and store the tensor shape
    auto shape = tensor_info.GetShape();
    tensor_shapes_[tensor_id] = shape;

    // Get and store the tensor type
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    tensor_types_[tensor_id] = get_element_type_string(element_type);
  } catch (const std::exception &e) {
    // Handle exception - maybe log it
  }
}

std::string TensorManager::getTensorType(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return tensor_types_.at(tensor_id);
}

std::vector<int64_t> TensorManager::getTensorShape(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return tensor_shapes_.at(tensor_id);
}

const char *TensorManager::get_element_type_string(ONNXTensorElementDataType element_type) {
  switch (element_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return "int8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return "uint16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return "int16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    return "string";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return "bool";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "float16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "double";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return "uint32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return "uint64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    return "complex64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    return "complex128";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return "bfloat16";
  default:
    return "unknown";
  }
}

std::string TensorManager::convertTensor(const std::string &tensor_id, const std::string &target_type) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the tensor exists
  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);

  if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
    throw std::runtime_error("Tensor not found");
  }

  const std::string &source_type = type_it->second;

  // If the target type is the same as the source type, just clone the tensor
  if (source_type == target_type) {
    // Create a new tensor ID
    std::string new_tensor_id = generateTensorId();

    // Clone the tensor
    Ort::Value *tensor = tensor_it->second.get();
    Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
    size_t elem_count = tensor_info.GetElementCount();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Create a new tensor based on the data type
    if (source_type == "float32") {
      float *data = tensor->GetTensorMutableData<float>();
      float *new_data = new float[elem_count];
      std::memcpy(new_data, data, elem_count * sizeof(float));
      auto new_tensor = Ort::Value::CreateTensor<float>(memory_info_, new_data, elem_count, shape.data(), shape.size());
      tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    } else if (source_type == "int32") {
      int32_t *data = tensor->GetTensorMutableData<int32_t>();
      int32_t *new_data = new int32_t[elem_count];
      std::memcpy(new_data, data, elem_count * sizeof(int32_t));
      auto new_tensor =
          Ort::Value::CreateTensor<int32_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
      tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    } else if (source_type == "int64") {
      int64_t *data = tensor->GetTensorMutableData<int64_t>();
      int64_t *new_data = new int64_t[elem_count];
      std::memcpy(new_data, data, elem_count * sizeof(int64_t));
      auto new_tensor =
          Ort::Value::CreateTensor<int64_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
      tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    } else if (source_type == "uint8") {
      uint8_t *data = tensor->GetTensorMutableData<uint8_t>();
      uint8_t *new_data = new uint8_t[elem_count];
      std::memcpy(new_data, data, elem_count * sizeof(uint8_t));
      auto new_tensor =
          Ort::Value::CreateTensor<uint8_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
      tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    } else if (source_type == "bool") {
      bool *data = tensor->GetTensorMutableData<bool>();
      bool *new_data = new bool[elem_count];
      std::memcpy(new_data, data, elem_count * sizeof(bool));
      auto new_tensor = Ort::Value::CreateTensor<bool>(memory_info_, new_data, elem_count, shape.data(), shape.size());
      tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    }

    // Store the type and shape
    tensor_types_[new_tensor_id] = source_type;
    tensor_shapes_[new_tensor_id] = shape;

    return new_tensor_id;
  }

  // Convert based on the source type
  if (source_type == "float32") {
    return convertFloat32To(tensor_id, target_type);
  } else if (source_type == "int32") {
    return convertInt32To(tensor_id, target_type);
  } else if (source_type == "int64") {
    return convertInt64To(tensor_id, target_type);
  } else if (source_type == "uint8") {
    return convertUint8To(tensor_id, target_type);
  } else if (source_type == "bool") {
    return convertBoolTo(tensor_id, target_type);
  }

  throw std::runtime_error("Unsupported type: " + source_type);
}

std::string TensorManager::convertFloat32To(const std::string &tensor_id, const std::string &target_type) {
  // Get the tensor
  Ort::Value *tensor = tensors_[tensor_id].get();
  Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
  size_t elem_count = tensor_info.GetElementCount();
  std::vector<int64_t> shape = tensor_info.GetShape();
  float *data = tensor->GetTensorMutableData<float>();

  // Create a new tensor ID
  std::string new_tensor_id = generateTensorId();

  // Convert to the target type
  if (target_type == "int32") {
    // Convert float32 to int32
    int32_t *new_data = new int32_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Round float to int
      new_data[i] = static_cast<int32_t>(data[i] + (data[i] >= 0 ? 0.5f : -0.5f));
    }
    auto new_tensor = Ort::Value::CreateTensor<int32_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int32";
  } else if (target_type == "int64") {
    // Convert float32 to int64
    int64_t *new_data = new int64_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Round float to int64
      new_data[i] = static_cast<int64_t>(data[i] + (data[i] >= 0 ? 0.5f : -0.5f));
    }
    auto new_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int64";
  } else if (target_type == "uint8") {
    // Convert float32 to uint8
    uint8_t *new_data = new uint8_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Clamp between 0 and 255
      float val = data[i] < 0 ? 0 : (data[i] > 255 ? 255 : data[i] + 0.5f);
      new_data[i] = static_cast<uint8_t>(val);
    }
    auto new_tensor = Ort::Value::CreateTensor<uint8_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "uint8";
  } else if (target_type == "bool") {
    // Convert float32 to bool
    bool *new_data = new bool[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] != 0.0f;
    }
    auto new_tensor = Ort::Value::CreateTensor<bool>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "bool";
  } else {
    throw std::runtime_error("Unsupported type: " + target_type);
  }

  // Store the shape
  tensor_shapes_[new_tensor_id] = shape;

  return new_tensor_id;
}

std::string TensorManager::convertInt32To(const std::string &tensor_id, const std::string &target_type) {
  // Get the tensor
  Ort::Value *tensor = tensors_[tensor_id].get();
  Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
  size_t elem_count = tensor_info.GetElementCount();
  std::vector<int64_t> shape = tensor_info.GetShape();
  int32_t *data = tensor->GetTensorMutableData<int32_t>();

  // Create a new tensor ID
  std::string new_tensor_id = generateTensorId();

  // Convert to the target type
  if (target_type == "float32") {
    // Convert int32 to float32
    float *new_data = new float[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = static_cast<float>(data[i]);
    }
    auto new_tensor = Ort::Value::CreateTensor<float>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "float32";
  } else if (target_type == "int64") {
    // Convert int32 to int64
    int64_t *new_data = new int64_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = static_cast<int64_t>(data[i]);
    }
    auto new_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int64";
  } else if (target_type == "uint8") {
    // Convert int32 to uint8
    uint8_t *new_data = new uint8_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Clamp between 0 and 255
      int32_t val = data[i] < 0 ? 0 : (data[i] > 255 ? 255 : data[i]);
      new_data[i] = static_cast<uint8_t>(val);
    }
    auto new_tensor = Ort::Value::CreateTensor<uint8_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "uint8";
  } else if (target_type == "bool") {
    // Convert int32 to bool
    bool *new_data = new bool[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] != 0;
    }
    auto new_tensor = Ort::Value::CreateTensor<bool>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "bool";
  } else {
    throw std::runtime_error("Unsupported type: " + target_type);
  }

  // Store the shape
  tensor_shapes_[new_tensor_id] = shape;

  return new_tensor_id;
}

std::string TensorManager::convertInt64To(const std::string &tensor_id, const std::string &target_type) {
  // Get the tensor
  Ort::Value *tensor = tensors_[tensor_id].get();
  Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
  size_t elem_count = tensor_info.GetElementCount();
  std::vector<int64_t> shape = tensor_info.GetShape();
  int64_t *data = tensor->GetTensorMutableData<int64_t>();

  // Create a new tensor ID
  std::string new_tensor_id = generateTensorId();

  // Convert to the target type
  if (target_type == "float32") {
    // Convert int64 to float32
    float *new_data = new float[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Note: potential precision loss for large int64 values
      new_data[i] = static_cast<float>(data[i]);
    }
    auto new_tensor = Ort::Value::CreateTensor<float>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "float32";
  } else if (target_type == "int32") {
    // Convert int64 to int32
    int32_t *new_data = new int32_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Clamp to int32 range to prevent overflow
      int64_t val = data[i];
      if (val > INT32_MAX)
        val = INT32_MAX;
      if (val < INT32_MIN)
        val = INT32_MIN;
      new_data[i] = static_cast<int32_t>(val);
    }
    auto new_tensor = Ort::Value::CreateTensor<int32_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int32";
  } else if (target_type == "uint8") {
    // Convert int64 to uint8
    uint8_t *new_data = new uint8_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      // Clamp between 0 and 255
      int64_t val = data[i] < 0 ? 0 : (data[i] > 255 ? 255 : data[i]);
      new_data[i] = static_cast<uint8_t>(val);
    }
    auto new_tensor = Ort::Value::CreateTensor<uint8_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "uint8";
  } else if (target_type == "bool") {
    // Convert int64 to bool
    bool *new_data = new bool[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] != 0;
    }
    auto new_tensor = Ort::Value::CreateTensor<bool>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "bool";
  } else {
    throw std::runtime_error("Unsupported type: " + target_type);
  }

  // Store the shape
  tensor_shapes_[new_tensor_id] = shape;

  return new_tensor_id;
}

std::string TensorManager::convertUint8To(const std::string &tensor_id, const std::string &target_type) {
  // Get the tensor
  Ort::Value *tensor = tensors_[tensor_id].get();
  Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
  size_t elem_count = tensor_info.GetElementCount();
  std::vector<int64_t> shape = tensor_info.GetShape();
  uint8_t *data = tensor->GetTensorMutableData<uint8_t>();

  // Create a new tensor ID
  std::string new_tensor_id = generateTensorId();

  // Convert to the target type
  if (target_type == "float32") {
    // Convert uint8 to float32
    float *new_data = new float[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = static_cast<float>(data[i]);
    }
    auto new_tensor = Ort::Value::CreateTensor<float>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "float32";
  } else if (target_type == "int32") {
    // Convert uint8 to int32
    int32_t *new_data = new int32_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = static_cast<int32_t>(data[i]);
    }
    auto new_tensor = Ort::Value::CreateTensor<int32_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int32";
  } else if (target_type == "int64") {
    // Convert uint8 to int64
    int64_t *new_data = new int64_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = static_cast<int64_t>(data[i]);
    }
    auto new_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int64";
  } else if (target_type == "bool") {
    // Convert uint8 to bool
    bool *new_data = new bool[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] != 0;
    }
    auto new_tensor = Ort::Value::CreateTensor<bool>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "bool";
  } else {
    throw std::runtime_error("Unsupported type: " + target_type);
  }

  // Store the shape
  tensor_shapes_[new_tensor_id] = shape;

  return new_tensor_id;
}

std::string TensorManager::convertBoolTo(const std::string &tensor_id, const std::string &target_type) {
  // Get the tensor
  Ort::Value *tensor = tensors_[tensor_id].get();
  Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
  size_t elem_count = tensor_info.GetElementCount();
  std::vector<int64_t> shape = tensor_info.GetShape();
  bool *data = tensor->GetTensorMutableData<bool>();

  // Create a new tensor ID
  std::string new_tensor_id = generateTensorId();

  // Convert to the target type
  if (target_type == "float32") {
    // Convert bool to float32
    float *new_data = new float[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] ? 1.0f : 0.0f;
    }
    auto new_tensor = Ort::Value::CreateTensor<float>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "float32";
  } else if (target_type == "int32") {
    // Convert bool to int32
    int32_t *new_data = new int32_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] ? 1 : 0;
    }
    auto new_tensor = Ort::Value::CreateTensor<int32_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int32";
  } else if (target_type == "int64") {
    // Convert bool to int64
    int64_t *new_data = new int64_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] ? 1 : 0;
    }
    auto new_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "int64";
  } else if (target_type == "uint8") {
    // Convert bool to uint8
    uint8_t *new_data = new uint8_t[elem_count];
    for (size_t i = 0; i < elem_count; i++) {
      new_data[i] = data[i] ? 1 : 0;
    }
    auto new_tensor = Ort::Value::CreateTensor<uint8_t>(memory_info_, new_data, elem_count, shape.data(), shape.size());
    tensors_[new_tensor_id] = std::make_unique<Ort::Value>(std::move(new_tensor));
    tensor_types_[new_tensor_id] = "uint8";
  } else {
    throw std::runtime_error("Unsupported type: " + target_type);
  }

  // Store the shape
  tensor_shapes_[new_tensor_id] = shape;

  return new_tensor_id;
}
