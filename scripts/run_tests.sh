#!/bin/bash

# Script to run unit and integration tests for flutter_onnxruntime
# Place this in the root directory of the project

# Exit on error
set -e

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Running flutter_onnxruntime tests...${NC}"

# Run unit tests
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Running unit tests...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
flutter test test/unit

echo -e "${BLUE}Running integration tests on all available platforms...${NC}"

cd example

# Function to run tests on a specific device
run_test_on_device() {
    local DEVICE=$1
    local PLATFORM_NAME=$2
    
    echo -e "${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Running tests on ${PLATFORM_NAME} device: ${DEVICE}${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"
    flutter test integration_test/onnxruntime_integration_test.dart -d "$DEVICE"
}

# Get all available devices
AVAILABLE_DEVICES=$(flutter devices)
echo -e "${YELLOW}Available devices:${NC}"
echo "$AVAILABLE_DEVICES"
echo ""

# Test on Android devices
ANDROID_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "android" | awk '{print $1}')
if [ -n "$ANDROID_DEVICES" ]; then
    # Take the first Android device
    ANDROID_DEVICE=$(echo "$ANDROID_DEVICES" | head -n 1)
    run_test_on_device "$ANDROID_DEVICE" "Android"
else
    echo -e "${RED}No Android devices found. Skipping Android tests.${NC}"
fi

# Test on iOS devices
IOS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "ios|iPhone|iPad" | awk '{print $1}')
if [ -n "$IOS_DEVICES" ]; then
    # Take the first iOS device
    IOS_DEVICE=$(echo "$IOS_DEVICES" | head -n 1)
    run_test_on_device "$IOS_DEVICE" "iOS"
else
    echo -e "${RED}No iOS devices found. Skipping iOS tests.${NC}"
fi

# Test on Linux desktop
LINUX_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "linux" | awk '{print $1}')
if [ -n "$LINUX_DEVICES" ]; then
    # Take the first Linux device (usually just "linux")
    LINUX_DEVICE=$(echo "$LINUX_DEVICES" | head -n 1)
    run_test_on_device "$LINUX_DEVICE" "Linux"
else
    echo -e "${RED}Linux desktop not available. Skipping Linux tests.${NC}"
fi

# Test on macOS desktop
MACOS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "macos" | awk '{print $1}')
if [ -n "$MACOS_DEVICES" ]; then
    # Take the first macOS device (usually just "macos")
    MACOS_DEVICE=$(echo "$MACOS_DEVICES" | head -n 1)
    run_test_on_device "$MACOS_DEVICE" "macOS"
else
    echo -e "${RED}macOS desktop not available. Skipping macOS tests.${NC}"
fi

echo -e "${GREEN}All tests completed!${NC}"