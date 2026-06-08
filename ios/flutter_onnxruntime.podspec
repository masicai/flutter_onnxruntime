#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_onnxruntime.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_onnxruntime'
  s.version          = '0.0.1'
  s.summary          = 'Flutter plugin for ONNX Runtime'
  s.description      = <<-DESC
Flutter plugin for running ONNX models with the native ONNX Runtime, supporting both CocoaPods and Swift Package Manager.
                       DESC
  s.homepage         = 'https://github.com/masicai/flutter_onnxruntime'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'MASIC AI' => 'contact@masicai.com' }
  s.source           = { :path => '.' }
  # Sources live in the Swift Package Manager layout and are shared by both build systems.
  s.source_files = 'flutter_onnxruntime/Sources/**/*.{swift,h,mm}'
  # The vendored ORT internal headers are C++; keep them out of the umbrella
  # header / module map (they cannot compile as Objective-C).
  s.project_header_files = 'flutter_onnxruntime/Sources/flutter_onnxruntime_objc/vendor/*.h'
  s.dependency 'Flutter'
  s.platform = :ios, '16.0'
  s.dependency 'onnxruntime-objc', '1.24.2'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'HEADER_SEARCH_PATHS' => '"${PODS_ROOT}/onnxruntime-objc/objectivec" "${PODS_ROOT}/onnxruntime-objc/objectivec/include"'
  }
  s.swift_version = '5.0'

  s.resource_bundles = {'flutter_onnxruntime_privacy' => ['flutter_onnxruntime/Sources/flutter_onnxruntime/PrivacyInfo.xcprivacy']}
end
