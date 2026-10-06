// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "flutter_onnxruntime",
  platforms: [
    .macOS("14.0")
  ],
  products: [
    // The Flutter tooling requires the library product name to be the
    // dasherized plugin name (it generates a dependency on "flutter-onnxruntime").
    .library(name: "flutter-onnxruntime", targets: ["flutter_onnxruntime"])
  ],
  dependencies: [
    // Pinned exactly so the vendored internal headers in
    // Sources/flutter_onnxruntime_objc/vendor/ always match the resolved package.
    // masicai fork pinning ORT 1.28.0 (fork tag 1.28.0). Microsoft's SPM repo has
    // no 1.28.x tag, and its framework-format binary gets embedded by Xcode, which
    // App Store Connect rejects (ITMS-90208, issue #71). The fork ships the same
    // binary repackaged as a library-format xcframework, so keep it until upstream
    // fixes the artifact format.
    .package(url: "https://github.com/masicai/onnxruntime-swift-package-manager", exact: "1.28.0")
  ],
  targets: [
    // Swift target. `import FlutterMacOS` resolves implicitly through the
    // Flutter tooling; do NOT declare a Flutter framework dependency here
    // (the FlutterFramework local package only exists on Flutter master).
    .target(
      name: "flutter_onnxruntime",
      dependencies: [
        "flutter_onnxruntime_objc",
        .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
      ],
      resources: [
        .process("PrivacyInfo.xcprivacy")
      ]
    ),
    // ObjC++ target: SwiftPM does not support mixed-language targets, so the
    // float16 C++ bridge and the messenger workaround live in their own module.
    .target(
      name: "flutter_onnxruntime_objc",
      dependencies: [
        .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
      ],
      cxxSettings: [
        // Make the vendored cxx_api.h resolve the ORT C/C++ API headers from
        // the binary xcframework ("onnxruntime/..." prefixed paths).
        .define("SPM_BUILD"),
        .headerSearchPath("vendor")
      ]
    )
  ],
  cxxLanguageStandard: .cxx17
)
