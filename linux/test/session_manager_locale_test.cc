// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for issue #73 — see the ScopedCLocale comment in
// src/session_manager.cc for why the locale guard exists.
//
// This test must own the first Ort::Env of its process: the ONNX operator
// schemas, where the affected constants are parsed, are registered once per
// process, so a process that already holds one cannot reproduce the regression.
// Hence its own test executable (see linux/CMakeLists.txt), and hence exactly
// one TEST in this file — a second one would register the schemas first and
// silently defuse the other.

#include <gtest/gtest.h>

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "src/session_manager.h"

namespace {

// Minimal ONNX model (ir_version 8, opset 14): input "x" float[1,8] -> HardSwish -> output "y"
// float[1,8]. Embedded rather than loaded from example/assets/models/hardswish_model.onnx (the same
// bytes, used by the integration suite) so the native test binary stays hermetic — it has no Flutter
// asset bundle. Regenerate both with:
//
//   import onnx
//   from onnx import TensorProto, helper
//   graph = helper.make_graph(
//       [helper.make_node("HardSwish", ["x"], ["y"])], "only_HardSwish",
//       [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
//       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])])
//   model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
//   model.ir_version = 8
//   onnx.save(model, "hardswish.onnx")
const unsigned char kHardSwishModel[] = {
    0x08, 0x08, 0x3a, 0x4d, 0x0a, 0x11, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x09, 0x48, 0x61, 0x72, 0x64,
    0x53, 0x77, 0x69, 0x73, 0x68, 0x12, 0x0e, 0x6f, 0x6e, 0x6c, 0x79, 0x5f, 0x48, 0x61, 0x72, 0x64, 0x53, 0x77,
    0x69, 0x73, 0x68, 0x5a, 0x13, 0x0a, 0x01, 0x78, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02,
    0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x62, 0x13, 0x0a, 0x01, 0x79, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12,
    0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0e,
};

// Set to a non-zero value to turn the "no comma-decimal locale installed" skip into a failure, so CI
// cannot pass this test by silently skipping it.
const char *const kRequireLocaleEnvVar = "FLUTTER_ONNXRUNTIME_REQUIRE_COMMA_LOCALE";

// Switches LC_NUMERIC to a comma-decimal locale, mimicking what GTK does when the user's system
// locale is e.g. German. Restores the previous locale on destruction. `name()` is empty if no
// comma-decimal locale is installed.
class CommaDecimalLocale {
public:
  CommaDecimalLocale() {
    const char *previous = setlocale(LC_NUMERIC, nullptr);
    previous_ = previous ? previous : "C";
    // glibc normalises the codeset at lookup, so the ".utf8" spellings would be dead entries.
    for (const char *candidate : {"de_DE.UTF-8", "fr_FR.UTF-8"}) {
      if (setlocale(LC_NUMERIC, candidate) != nullptr && std::string(localeconv()->decimal_point) == ",") {
        name_ = candidate;
        return;
      }
    }
  }
  ~CommaDecimalLocale() { setlocale(LC_NUMERIC, previous_.c_str()); }

  const std::string &name() const { return name_; }

private:
  std::string previous_;
  std::string name_;
};

// Pid-suffixed so concurrent runs don't race on one path, and removed on destruction so a run that
// fails an ASSERT partway through doesn't leave the file behind.
class ScopedModelFile {
public:
  ScopedModelFile() : path_(testing::TempDir() + "hardswish_issue73_" + std::to_string(getpid()) + ".onnx") {}
  ~ScopedModelFile() { std::remove(path_.c_str()); }

  const std::string &path() const { return path_; }

private:
  std::string path_;
};

} // namespace

TEST(SessionManagerLocaleTest, HardSwishIsCorrectUnderCommaDecimalLocale) {
  CommaDecimalLocale locale;
  if (locale.name().empty()) {
    const char *hint = "No comma-decimal locale is installed, so the issue #73 regression cannot be "
                       "exercised. Generate one with `sudo locale-gen de_DE.UTF-8`.";
    const char *require = std::getenv(kRequireLocaleEnvVar);
    ASSERT_TRUE(require == nullptr || std::string(require) == "0")
        << hint << " (" << kRequireLocaleEnvVar << " is set, so skipping is not allowed.)";
    GTEST_SKIP() << hint;
  }
  SCOPED_TRACE("comma-decimal locale: " + locale.name());

  // SessionManager loads models from a path, so write the embedded model to a temp file.
  ScopedModelFile model;
  const std::string &model_path = model.path();
  {
    std::ofstream model_file(model_path, std::ios::binary);
    model_file.write(reinterpret_cast<const char *>(kHardSwishModel), sizeof(kHardSwishModel));
    // Close before checking: write errors on a full/read-only temp dir only surface at flush time.
    model_file.close();
    ASSERT_TRUE(model_file.good()) << "failed to write " << model_path;
  }

  // Creates the Ort::Env (and thereby registers the ONNX schemas) with the comma-decimal locale
  // active — the exact situation of issue #73.
  SessionManager session_manager;

  // Positive control: the comma-decimal locale must still be active on this thread. If it is not,
  // either ScopedCLocale failed to restore the thread's locale, or the locale was never in effect and
  // the assertions below would pass vacuously.
  ASSERT_EQ(std::string(localeconv()->decimal_point), ",")
      << "comma-decimal locale not active after SessionManager construction";

  std::string session_id = session_manager.createSession(model_path.c_str(), nullptr);
  ASSERT_TRUE(session_manager.hasSession(session_id));

  std::vector<float> input = {-4.0f, -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f};
  std::vector<int64_t> shape = {1, static_cast<int64_t>(input.size())};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.push_back(
      Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), shape.data(), shape.size()));

  std::vector<Ort::Value> outputs = session_manager.runInference(session_id, inputs, {"x"});
  ASSERT_EQ(outputs.size(), 1u);
  // GetTensorData does no validation, so pin the type and element count before indexing.
  auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(output_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(output_info.GetElementCount(), input.size());
  const float *result = outputs[0].GetTensorData<float>();

  // HardSwish(x) = x * clamp(x / 6 + 0.5, 0, 1), as reported in issue #73. The broken
  // locale-dependent parse yields x * clamp(0 * x + 0, 0, 1) = +/-0 for every element.
  const float expected[] = {0.0f, 0.0f, -1.0f / 3.0f, -5.0f / 24.0f, 0.0f, 7.0f / 24.0f, 2.0f / 3.0f, 3.0f};
  ASSERT_EQ(input.size(), sizeof(expected) / sizeof(expected[0]));
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "at index " << i;
  }
}
