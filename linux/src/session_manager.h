// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef SESSION_MANAGER_H
#define SESSION_MANAGER_H

#include <map>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Session information structure
struct SessionInfo {
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

// Session Manager Class
class SessionManager {
public:
  SessionManager();
  ~SessionManager();

  // Create a new session from a model file path
  std::string createSession(const char *model_path, void *options);

  // Get a session by ID
  Ort::Session *getSession(const std::string &session_id);

  // Close and remove a session
  bool closeSession(const std::string &session_id);

  // Get session info
  SessionInfo *getSessionInfo(const std::string &session_id);

  // Check if a session exists
  bool hasSession(const std::string &session_id);

  // Get input names for a session
  std::vector<std::string> getInputNames(const std::string &session_id);

  // Get output names for a session
  std::vector<std::string> getOutputNames(const std::string &session_id);

private:
  // Generate a unique session ID
  std::string generateSessionId();

  // Map of session IDs to session info
  std::map<std::string, SessionInfo> sessions_;

  // Counter for generating unique session IDs
  int next_session_id_;

  // Mutex for thread safety
  std::mutex mutex_;

  // ONNX Runtime environment
  Ort::Env env_;
};

#endif // SESSION_MANAGER_H