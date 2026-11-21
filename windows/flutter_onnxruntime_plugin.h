// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FLUTTER_PLUGIN_FLUTTER_ONNXRUNTIME_PLUGIN_H_
#define FLUTTER_PLUGIN_FLUTTER_ONNXRUNTIME_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <functional>
#include <memory>
#include <string>

namespace flutter_onnxruntime {

// Runs native work off the platform thread; defined in Windows-only sources.
class PlatformThreadPump;

// Forward declaration
class FlutterOnnxruntimePluginImpl;

class FlutterOnnxruntimePlugin : public flutter::Plugin {
public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  FlutterOnnxruntimePlugin();

  virtual ~FlutterOnnxruntimePlugin();

  // Disallow copy and assign.
  FlutterOnnxruntimePlugin(const FlutterOnnxruntimePlugin &) = delete;
  FlutterOnnxruntimePlugin &operator=(const FlutterOnnxruntimePlugin &) = delete;

private:
  using PluginMethodResult =
      flutter::MethodResult<flutter::EncodableValue>;
  using PluginMethodCall = flutter::MethodCall<flutter::EncodableValue>;
  using PluginHandler = void (FlutterOnnxruntimePlugin::*)(
      const PluginMethodCall &, std::unique_ptr<PluginMethodResult>);

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                        std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // Runs |handler| on the background worker and delivers its result on the
  // platform thread. ONNX work must not run on the platform thread: that is
  // the thread pumping the Win32 message loop, so it also stalls the window.
  void DispatchToWorker(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result,
      PluginHandler handler);

  // Session management method handlers
  void HandleCreateSession(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                           std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetAvailableProviders(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                                   std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleRunInference(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                          std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleCloseSession(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                          std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetMetadata(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                         std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetInputInfo(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                          std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetOutputInfo(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                           std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // OrtValue method handlers
  void HandleCreateOrtValue(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                            std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleConvertOrtValue(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                             std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetOrtValueData(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                             std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleReleaseOrtValue(const flutter::MethodCall<flutter::EncodableValue> &method_call,
                             std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // Private implementation
  std::unique_ptr<FlutterOnnxruntimePluginImpl> impl_;

  // Executes ONNX calls off the platform thread.
  std::unique_ptr<PlatformThreadPump> pump_;
};

} // namespace flutter_onnxruntime

#endif // FLUTTER_PLUGIN_FLUTTER_ONNXRUNTIME_PLUGIN_H_