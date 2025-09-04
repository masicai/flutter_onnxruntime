## 1.5.1
* Upgrade ONNX Runtime to version 1.22.0 for 16KB Page Size support in Android (#39)
* Fix Linux build issues in CMake and X11 display server (#37)
* Upgrade Kotlin version to 2.1.0 (#38)

## 1.4.3
* Fix an issue with input name-value mismatch in Windows (#28 #29)
* Add more integration tests for input name and order consistency
* Add integration tests for int64 input tensor

## 1.4.2
* Remove unnecessary warnings in CMake for the Linux build
* Improve README and troubleshooting documentation

## 1.4.1
* Support string tensors in all platforms
* Reinforce structure and behavior consistency between Linux and Windows implementations
* Minor bug fixes and documentation updates

## 1.4.0
* Support Windows platform 🎉🎉🎉
* Refactor SessionManager in Linux for cleaner architecture
* Add Azure to provider list
* Improve the CMake build stability in Linux

## 1.3.0
* Support Web platform 🎉🎉🎉
* Add integration tests for Web in both local and CI

## 1.2.3
* Standardize Execution Provider names across all platforms
* Fix a bug in get_metadata method for Linux

## 1.2.2
* Improve example documentation

## 1.2.0
* Returning a multi-dimensional list for tensor data extraction
* Support return a flat list

## 1.1.0
* Standardize error codes and error messages from native
* Remove auto disposal of input tensors after inference in Kotlin
* Refactor C++ to return standard Platform Exception and impose standard error handling
* Add back the example to package as required by pub.dev; this will increase the package size unnecessarily

## 1.0.0
* Support running inference with an ONNX model on Android, iOS, Linux, and macOS
* Support ONNX Runtime version 1.21.0
