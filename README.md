<img src="flutter_onnxruntime.png" alt="flutter_onnxruntime" align="center"/>
<p align="center">
<a href="https://pub.dev/packages/flutter_onnxruntime" alt="Flutter ONNX Runtime on pub.dev">
        <img src="https://img.shields.io/pub/v/flutter_onnxruntime.svg" height="25" /></a>
<a href="https://pub.dev/packages/flutter_onnxruntime" alt="Flutter ONNX Runtime monthly downloads">
        <img src="https://img.shields.io/pub/dm/flutter_onnxruntime.svg" height="25" /></a>
</p>

# flutter_onnxruntime

Native Wrapper Flutter Plugin for ONNX Runtime

*Supported ONNX Runtime versions:*

| Platform | ONNX Runtime | Source |
|----------|:------------:|--------|
| Android | **1.28.0** | Maven `com.microsoft.onnxruntime:onnxruntime-android` |
| iOS, macOS | **1.28.0** | Swift Package Manager (recommended), or CocoaPods `onnxruntime-objc` |
| Linux, Windows | **1.28.3** | Official GitHub release, downloaded at build time|
| Web | chosen by your app | `onnxruntime-web` script tag in `web/index.html`|


*Note:* For Android build, you need to upgrade your `flutter_onnxruntime` to version `>=1.5.1` to satisfy the [16 KB Google Play compatibility requirement](https://android-developers.googleblog.com/2025/05/prepare-play-apps-for-devices-with-16kb-page-size.html).

## 🌟 Why This Project?

`flutter_onnxruntime` is a lightweight plugin that provides native wrappers for running ONNX Runtime on multiple platforms.

      📦 No Pre-built Libraries
      Libraries are fetched directly from official repositories during installation, ensuring they are always up-to-date!

      🛡️ Memory Safety
      All memory management is handled in native code, reducing the risk of memory leaks.

      🔄 Easy Upgrades
      Stay current with the latest ONNX Runtime releases without the hassle of maintaining complex generated FFI wrappers.

## 🚀 Getting Started

### Installation

Add the following dependency to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_onnxruntime: ^1.8.1
```

### Quick Start

Example of running an addition model:
```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

// create inference session
final ort = OnnxRuntime();
final session = await ort.createSessionFromAsset('assets/models/addition_model.onnx');

// specify input with data and shape
final inputs = {
   'A': await OrtValue.fromList([1, 1, 1], [3]),
   'B': await OrtValue.fromList([2, 2, 2], [3])
}

// start the inference
final outputs = await session.run(inputs);

// print output data
print(await outputs['C']!.asList());
```

To get started with the Flutter ONNX Runtime plugin, see the [API Usage Guide](doc/api_usage.md).

## 🧪 Examples

### [Simple Addition Model](example/)

A simple model with only one operator (Add) that takes two inputs and produces one output.

Run this example with:
```bash
cd example
flutter pub get
flutter run
```

### [Image Classification Model](https://github.com/masicai/flutter-onnxruntime-examples)

A more complex model that takes an image as input and classifies it into one of the predefined categories.

Clone [this repository](https://github.com/masicai/flutter-onnxruntime-examples) and run the example following the repo's guidelines.

## 📊 Component Overview

| Component | Description |
|-----------|-------------|
| OnnxRuntime | Main entry point for creating sessions and configuring global options |
| OrtSession | Represents a loaded ML model for running inference |
| OrtValue | Represents tensor data for inputs and outputs |
| OrtSessionOptions | Configuration options for session creation |
| OrtRunOptions | Configuration options for inference execution |

## 🚧 Implementation Status

| Feature | Android | iOS | Linux | macOS | Windows | Web |
|---------|:-------:|:---:|:-----:|:-----:|:-------:|:---: |
| CPU Inference | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| EP<sup>1</sup> Configuration | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Input/Output names | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Data Type Conversion | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inference on Emulator | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Input/Output Info | ✅ | ❌* | ✅ | ❌* | ✅ | ✅ |
| Model Metadata | ✅ | ❌* | ✅ | ❌* | ✅ | ❌* |
| FP16 Support | ✅ | ✅ | ✍️ | ✅ | ✍️ | ✍️ |

✅: Completed

❌: Not supported

🚧: Ongoing

✍️: Planned

`*`: Retrieving model metadata and input/output info is not available for Swift and Javascript API.

<sup>1</sup>: Execution Providers (EP) are hardware accelerated inference interface for AI inference (e.g., CPU, GPU, NPU, TPU, etc.) 

## 📋 Required development setup

### Android

Android build requires `proguard-rules.pro` inside your Android project at `android/app/` with the following content:
  ```
  -keep class ai.onnxruntime.** { *; }
  ```
or running the below command from your terminal:

  ```bash
  echo "-keep class ai.onnxruntime.** { *; }" > android/app/proguard-rules.pro
  ```

Refer to [troubleshooting.md](doc/troubleshooting.md) for more information.

### iOS and macOS

ONNX Runtime requires at least iOS 16 and macOS 14. Open `ios/Runner.xcworkspace` or `macos/Runner.xcworkspace` in Xcode, then in `Runner` → `General` set `Minimum Deployments` to `16.0` (iOS) or `14.0` (macOS).

Use [Swift Package Manager](https://docs.flutter.dev/packages-and-plugins/swift-package-manager/for-app-developers) (SPM). It needs no Podfile changes. Flutter 3.44 and later turn it on by default. On Flutter 3.24 to 3.41, turn it on with:

```bash
flutter config --enable-swift-package-manager
```

CocoaPods still works, but we no longer recommend it:
- Flutter has made SPM the default.
- CocoaPods trunk becomes read-only on 2 December 2026.
- A clean `pod install` currently fails because Microsoft's download host is down (see the [Microsoft download outage](doc/troubleshooting.md#microsoft-download-outage-november-2026)).

If you stay on CocoaPods, see [CocoaPods setup](doc/troubleshooting.md#cocoapods-setup) for the Podfile changes.

### Linux and Windows

The build downloads ONNX Runtime 1.28.3 from the official GitHub release. Those downloads are currently down, so clean builds fail (see the [Microsoft download outage](doc/troubleshooting.md#microsoft-download-outage-november-2026)). To use another version, set `ONNXRUNTIME_VERSION` as a plain variable in your app's `linux/CMakeLists.txt` or `windows/CMakeLists.txt`, anywhere before the `include(flutter/generated_plugins.cmake)` line:

```cmake
set(ONNXRUNTIME_VERSION "1.22.0")
```

Passing `-DONNXRUNTIME_VERSION=1.22.0` to CMake works too. With `set(... CACHE ...)` instead, changing the version later needs `flutter clean`, because CMake never overwrites an existing cache entry.


## 🛠️ Troubleshooting

For troubleshooting, see the [troubleshooting.md](doc/troubleshooting.md) file.

## 🤝 Contributing
Contributions to the Flutter ONNX Runtime plugin are welcome. Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## 📚 Documentation
Find more information in the [documentation](doc/).
