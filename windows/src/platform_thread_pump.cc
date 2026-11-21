// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "platform_thread_pump.h"

#include <utility>

namespace flutter_onnxruntime {

namespace {

constexpr wchar_t kPumpWindowClass[] = L"FlutterOnnxRuntimePlatformPump";
constexpr wchar_t kCompletionMessageName[] =
    L"FlutterOnnxRuntime.Completion";
// Used only if RegisterWindowMessageW fails, which would be highly unusual.
// WM_APP-based ids are valid for PostMessage and cannot collide with the ones
// Windows defines.
constexpr UINT kFallbackCompletionMessage = WM_APP + 0x4F4E;  // 'ON'

HINSTANCE ModuleHandleForProc(WNDPROC proc) {
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(proc), &module)) {
    return module;
  }
  return GetModuleHandleW(nullptr);
}

}  // namespace

PlatformThreadPump::PlatformThreadPump() = default;

PlatformThreadPump::~PlatformThreadPump() { Stop(); }

void PlatformThreadPump::Start() {
  if (window_.load() != nullptr) {
    return;
  }

  UINT message = RegisterWindowMessageW(kCompletionMessageName);
  completion_message_.store(message == 0 ? kFallbackCompletionMessage : message);

  WNDCLASSEXW window_class = {};
  window_class.cbSize = sizeof(window_class);
  window_class.lpfnWndProc = &PlatformThreadPump::WindowProc;
  window_class.hInstance = ModuleHandleForProc(&PlatformThreadPump::WindowProc);
  window_class.lpszClassName = kPumpWindowClass;
  // A second engine in the same process reuses the class; that is expected.
  RegisterClassExW(&window_class);

  HWND window = CreateWindowExW(0, kPumpWindowClass, L"", 0, 0, 0, 0, 0,
                                HWND_MESSAGE, nullptr, window_class.hInstance,
                                this);
  if (window != nullptr) {
    SetWindowLongPtrW(window, GWLP_USERDATA,
                      reinterpret_cast<LONG_PTR>(this));
  }
  window_.store(window);

  worker_ = std::thread(&PlatformThreadPump::RunWorker, this);
}

void PlatformThreadPump::Stop() {
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
    // Do not start queued work that the shutdown has already made pointless.
    // The task already running still completes: it holds manager state that
    // must not be freed underneath it.
    tasks_.clear();
  }
  task_ready_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }

  HWND window = window_.exchange(nullptr);
  if (window != nullptr) {
    DestroyWindow(window);
  }
  // No message loop remains to run them.
  std::lock_guard<std::mutex> lock(completion_mutex_);
  completions_.clear();
}

void PlatformThreadPump::PostTask(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (stopping_) {
      return;
    }
    tasks_.push_back(std::move(task));
  }
  task_ready_.notify_one();
}

void PlatformThreadPump::PostCompletion(std::function<void()> completion) {
  HWND window = window_.load();
  UINT message = completion_message_.load();
  if (window == nullptr || message == 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    completions_.push_back(std::move(completion));
  }
  // Wakes the platform thread's GetMessage loop, which dispatches the
  // completion on that thread.
  PostMessageW(window, message, 0, 0);
}

void PlatformThreadPump::RunWorker() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_ready_.wait(lock,
                       [this] { return stopping_ || !tasks_.empty(); });
      if (tasks_.empty()) {
        // Stop() waits for queued tasks, so this only happens once they are
        // all done.
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop_front();
    }
    if (task) {
      task();
    }
  }
}

void PlatformThreadPump::DrainCompletions() {
  std::deque<std::function<void()>> completions;
  {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    completions.swap(completions_);
  }
  for (std::function<void()>& completion : completions) {
    if (!completion) {
      continue;
    }
    try {
      completion();
    } catch (...) {
      // A reply that throws must not unwind into the window procedure.
    }
  }
}

// static
LRESULT CALLBACK PlatformThreadPump::WindowProc(HWND window, UINT message,
                                                WPARAM wparam, LPARAM lparam) {
  auto* pump = reinterpret_cast<PlatformThreadPump*>(
      GetWindowLongPtrW(window, GWLP_USERDATA));
  if (pump != nullptr &&
      message == pump->completion_message_.load()) {
    pump->DrainCompletions();
    return 0;
  }
  return DefWindowProcW(window, message, wparam, lparam);
}

}  // namespace flutter_onnxruntime
