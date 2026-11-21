// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FLUTTER_ONNXRUNTIME_PLATFORM_THREAD_PUMP_H_
#define FLUTTER_ONNXRUNTIME_PLATFORM_THREAD_PUMP_H_

#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace flutter_onnxruntime {

/// Runs work on one background thread and hands results back to the platform
/// thread through a message-only window.
///
/// The Windows embedder dispatches plugin method calls on the platform thread,
/// which also runs the application's Win32 message loop. ONNX Runtime calls
/// take long enough that running them there stalls window input and repainting,
/// which is why the Android, iOS, and macOS plugin implementations already run
/// inference on a background task queue. This class provides the same behavior
/// on Windows: the plugin queues its native work with [PostTask], and every
/// result is delivered through [PostCompletion], whose callback is invoked on
/// the platform thread by the message loop.
class PlatformThreadPump {
 public:
  PlatformThreadPump();
  ~PlatformThreadPump();

  PlatformThreadPump(const PlatformThreadPump&) = delete;
  PlatformThreadPump& operator=(const PlatformThreadPump&) = delete;

  /// Starts the worker thread and creates the message-only window.
  /// Must be called on the platform thread.
  void Start();

  /// True when the worker and window are available. A plugin whose pump failed
  /// to start must keep working, so it falls back to inline execution.
  bool IsRunning() const { return window_.load() != nullptr; }

  /// Runs queued tasks to completion, joins the worker, and destroys the
  /// window. Must be called on the platform thread.
  void Stop();

  /// Queues |task| on the worker thread. Tasks run in submission order.
  /// Dropped when the pump has been stopped.
  void PostTask(std::function<void()> task);

  /// Queues |completion| to run on the platform thread. May be called from
  /// any thread.
  void PostCompletion(std::function<void()> completion);

 private:
  static LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wparam,
                                     LPARAM lparam);

  void RunWorker();
  void DrainCompletions();

  std::atomic<HWND> window_{nullptr};
  std::atomic<UINT> completion_message_{0};

  std::thread worker_;
  std::mutex task_mutex_;
  std::condition_variable task_ready_;
  std::deque<std::function<void()>> tasks_;
  bool stopping_ = false;

  std::mutex completion_mutex_;
  std::deque<std::function<void()>> completions_;
};

}  // namespace flutter_onnxruntime

#endif  // FLUTTER_ONNXRUNTIME_PLATFORM_THREAD_PUMP_H_
