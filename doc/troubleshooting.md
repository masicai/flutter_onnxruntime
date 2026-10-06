# Troubleshooting

Common issues and their solutions.

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

## iOS
* Target minimum version: iOS 16
    * Open `ios/Podfile` and change the target minimum version to 16.0
        ```pod
        platform :ios, '16.0'
        ```
* "The 'Pods-Runner' target has transitive dependencies that include statically linked binaries: (onnxruntime-objc and onnxruntime-c)". In `Podfile` change:
    ```pod
    target 'Runner' do
    use_frameworks! :linkage => :static
    ```
* `RuntimeException` while running Reshape node with "input_shape_size == size was false"
    If you are using an ORT optimized model, it's possible that there is some certain nodes that is not supported by ORT. Try using the original ONNX model (without ORT optimization) to see if the issue persists.
* `CocoaPods could not find compatible versions for pod "onnxruntime-objc"` or `CocoaPods's specs repository is too out-of-date to satisfy dependencies`:
    This usually happens when you have an older version of `onnxruntime-objc` installed in your local CocoaPods repository, for example right after upgrading the plugin to a new ONNX Runtime version. Try running the following command to update your local CocoaPods repository:
    ```
    cd ios/
    pod update onnxruntime-objc
    ```

## macOS
* Target minimum version: MacOS 14
    * Open `macos/Podfile` and change the target minimum version to 14.0
        ```pod
        platform :osx, '14.0'
        ```
    * "error: compiling for macOS 10.14, but module 'flutter_onnxruntime' has a minimum deployment target of macOS 14.0".
        * In terminal, cd to the `macos` directory and run the XCode to open the project:
            ```
            open Runner.xcworkspace
            ```
        * In `Runner` -> `General`, change `Minimum Deployments` to `14.0`.
* "The 'Pods-Runner' target has transitive dependencies that include statically linked binaries: (onnxruntime-objc and onnxruntime-c)". In `Podfile` change:
    ```
    target 'Runner' do
    use_frameworks! :linkage => :static
    ```


## Linux
* When running with ONNX Runtime 1.21.0, you may see reference counting warnings related to FlValue objects. These don't prevent the app from running but may be addressed in future updates.
