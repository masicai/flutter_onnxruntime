# Troubleshooting

Common issues and their solutions.

## Microsoft download outage (November 2026)

In November 2026, Microsoft stopped serving ONNX Runtime from two places: the `microsoft/onnxruntime` GitHub repository and its release files (around 17 November), and `download.onnxruntime.ai` (around 20 November). A clean build that downloads ONNX Runtime from either place fails. Machines that still have the files cached keep building. Those caches are the CocoaPods and SwiftPM caches under `~/Library/Caches`, and on Linux and Windows the build directory, which `flutter clean` deletes. We will remove this section once Microsoft restores the downloads.

| Build | Status | What to do |
|---|---|---|
| iOS / macOS with Swift Package Manager | Fixed in 1.9.1 | Upgrade to `flutter_onnxruntime` 1.9.1 or later |
| iOS / macOS with CocoaPods | A clean `pod install` fails | Switch to Swift Package Manager |
| Linux / Windows | Clean builds fail | No workaround yet, see [#80](https://github.com/masicai/flutter_onnxruntime/issues/80) |
| Android, Web | Not affected | None needed |

### CocoaPods: `pod install` fails on `onnxruntime-c`
```
[!] Error installing onnxruntime-c
curl: (60) SSL: no alternative certificate subject name matches target host name 'download.onnxruntime.ai'
```
The `onnxruntime-c` and `onnxruntime-objc` pods download their binaries from `download.onnxruntime.ai`, which now presents the wrong TLS certificate and returns 404 for every file. This affects every plugin version. Switch to Swift Package Manager as described in the [README](../README.md#ios-and-macos). The plugin's Swift package doesn't use these pods. If other plugins in your app only support CocoaPods, Flutter still installs them with CocoaPods.

### Swift Package Manager: `onnxruntime_extensions` fails to download
```
failed downloading 'https://download.onnxruntime.ai/pod-archive-onnxruntime-extensions-c-0.13.0.zip' which is required by binary target 'onnxruntime_extensions'
```
`flutter_onnxruntime` 1.9.0 and earlier pin an ONNX Runtime Swift package whose unused `onnxruntime_extensions` target downloads from `download.onnxruntime.ai`. SwiftPM downloads every binary target of a package, used or not. Upgrade to 1.9.1 or later, which pins a package without that target ([#83](https://github.com/masicai/flutter_onnxruntime/issues/83)).

### Linux and Windows: the ONNX Runtime download fails
The build downloads ONNX Runtime from `https://github.com/microsoft/onnxruntime/releases/download/...`, which returns 404. [#80](https://github.com/masicai/flutter_onnxruntime/issues/80) reports it on Windows as:
```
url is not available: https://github.com/microsoft/onnxruntime/releases/download/v1.28.3/onnxruntime-win-x64-1.28.3.zip
```
Setting `ONNXRUNTIME_VERSION` doesn't help, because every version comes from the same place. Until there is a fix, keep the build directory of a machine that has built successfully (don't run `flutter clean` there), and follow [#80](https://github.com/masicai/flutter_onnxruntime/issues/80).

## Known issues in ONNX Runtime 1.28

### KleidiAI memory growth on SME-capable ARM64 devices
[microsoft/onnxruntime#29538](https://github.com/microsoft/onnxruntime/issues/29538), unfixed upstream. On devices with SME/SME2 (Apple M4 and later, A19-class iPhones, SME2 Android SoCs), each inference thread keeps KleidiAI MatMul buffers sized for the largest input shape it has seen, so memory grows with varying input shapes and is never returned. The convolution part of this regression ([#66](https://github.com/masicai/flutter_onnxruntime/issues/66)) is fixed in 1.28. If memory still grows, disable KleidiAI, which is slower:
```dart
OrtSessionOptions(sessionConfigs: {'mlas.disable_kleidiai': '1'})
```

## Android
* `OrtProvider.ARM_NN` fails with `INVALID_PROVIDER`: the ArmNN execution provider was removed in ONNX Runtime 1.25. Use `XNNPACK`, `NNAPI` or `CPU` instead.
* `JNI DETECTED ERROR IN APPLICATION: mid == null`
    For Android consumers using the library with R8-minimized builds, currently you need to add the following line to your `proguard-rules.pro` inside your Android project at `android/app/` ([reference](https://onnxruntime.ai/docs/build/android.html#note-proguard-rules-for-r8-minimization-android-app-builds-to-work))
    ```
    -keep class ai.onnxruntime.** { *; }
    ```
    or run the below bash command from the project root:
    ```bash
    echo "-keep class ai.onnxruntime.** { *; }" > android/app/proguard-rules.pro
    ```
* `...Module was compiled with an incompatible version of Kotlin. The binary version of its metadata is 2.1.0, expected version is 1.8.0`: update your Kotlin version to 2.1.0 in `android/settings.gradle.kts`:
    ```kotlin
    plugins {
        id("org.jetbrains.kotlin.android") version "2.1.0" apply false
    }
    ```

## iOS and macOS
* "error: compiling for macOS 10.14, but module 'flutter_onnxruntime' has a minimum deployment target of macOS 14.0" (or the iOS equivalent): the app's deployment target is below the plugin's minimum (iOS 16, macOS 14). Open `ios/Runner.xcworkspace` or `macos/Runner.xcworkspace` in Xcode, then in `Runner` → `General` set `Minimum Deployments` to `16.0` (iOS) or `14.0` (macOS).
* `RuntimeException` while running Reshape node with "input_shape_size == size was false"
    If you are using an ORT optimized model, it's possible that there is some certain nodes that is not supported by ORT. Try using the original ONNX model (without ORT optimization) to see if the issue persists.

### CocoaPods setup
Swift Package Manager is the recommended setup and needs none of this (see the [README](../README.md#ios-and-macos)). With CocoaPods, also read the [Microsoft download outage](#microsoft-download-outage-november-2026) above.

* Set the minimum platform and static linkage in the Podfile. In `ios/Podfile`:
    ```ruby
    platform :ios, '16.0'

    # existing code ...

    target 'Runner' do
      use_frameworks! :linkage => :static
    ```
    In `macos/Podfile`:
    ```ruby
    platform :osx, '14.0'

    # existing code ...

    target 'Runner' do
      use_frameworks! :linkage => :static
    ```
    Without static linkage, `pod install` fails with "The 'Pods-Runner' target has transitive dependencies that include statically linked binaries: (onnxruntime-objc and onnxruntime-c)".
* `CocoaPods could not find compatible versions for pod "onnxruntime-objc"` or `CocoaPods's specs repository is too out-of-date to satisfy dependencies`:
    This usually happens when you have an older version of `onnxruntime-objc` installed in your local CocoaPods repository, for example right after upgrading the plugin to a new ONNX Runtime version. Try running the following command to update your local CocoaPods repository:
    ```
    cd ios/
    pod update onnxruntime-objc
    ```

## Linux
* When running with ONNX Runtime 1.21.0, you may see reference counting warnings related to FlValue objects. These don't prevent the app from running but may be addressed in future updates.
