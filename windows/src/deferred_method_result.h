// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FLUTTER_ONNXRUNTIME_DEFERRED_METHOD_RESULT_H_
#define FLUTTER_ONNXRUNTIME_DEFERRED_METHOD_RESULT_H_

#include <flutter/encodable_value.h>
#include <flutter/method_result.h>

#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace flutter_onnxruntime {

/// A MethodResult used while a handler runs on the background worker.
///
/// The handler still calls Success/Error/NotImplemented exactly as before, but
/// this implementation records the outcome and hands it to a delivery callback
/// instead of replying to the engine. The callback (see
/// FlutterOnnxruntimePlugin::DispatchToWorker) forwards the outcome to the real
/// result on the platform thread, which is required for a method-channel reply.
class DeferredMethodResult final
    : public flutter::MethodResult<flutter::EncodableValue> {
 public:
  // Receives the outcome as a callable that takes the real result. The
  // delivery is responsible for running that callable on the platform thread.
  using Delivery = std::function<void(
      std::function<void(flutter::MethodResult<flutter::EncodableValue>&)>)>;

  explicit DeferredMethodResult(Delivery deliver)
      : deliver_(std::move(deliver)) {}

 protected:
  void SuccessInternal(const flutter::EncodableValue* result) override {
    const bool has_value = result != nullptr;
    flutter::EncodableValue value =
        has_value ? *result : flutter::EncodableValue();
    deliver_([has_value, value = std::move(value)](
                 flutter::MethodResult<flutter::EncodableValue>& target) mutable {
      if (has_value) {
        target.Success(value);
      } else {
        target.Success();
      }
    });
  }

  void ErrorInternal(const std::string& error_code,
                     const std::string& error_message,
                     const flutter::EncodableValue* error_details) override {
    std::optional<flutter::EncodableValue> details;
    if (error_details != nullptr) {
      details = *error_details;
    }
    deliver_([error_code, error_message, details = std::move(details)](
                 flutter::MethodResult<flutter::EncodableValue>& target) {
      if (details.has_value()) {
        target.Error(error_code, error_message, *details);
      } else {
        target.Error(error_code, error_message);
      }
    });
  }

  void NotImplementedInternal() override {
    deliver_([](flutter::MethodResult<flutter::EncodableValue>& target) {
      target.NotImplemented();
    });
  }

 private:
  Delivery deliver_;
};

}  // namespace flutter_onnxruntime

#endif  // FLUTTER_ONNXRUNTIME_DEFERRED_METHOD_RESULT_H_
