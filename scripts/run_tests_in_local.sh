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

# Define directories for chromedriver
CHROMEDRIVER_DIR="$HOME/.cache/flutter_onnxruntime/chromedriver"
CHROMEDRIVER="$CHROMEDRIVER_DIR/chromedriver"

# Function to run tests on a specific device
run_integration_test_on_device() {
    local DEVICE=$1
    local PLATFORM_NAME=$2

    echo -e "${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Running integration tests on ${PLATFORM_NAME} device: ${DEVICE}${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"
    flutter test integration_test/all_tests.dart -d "$DEVICE"
}

# Function to check if Chrome is installed
check_chrome_installed() {
    if command -v google-chrome &> /dev/null || command -v google-chrome-stable &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to get Chrome version
get_chrome_version() {
    if command -v google-chrome &> /dev/null; then
        google-chrome --version | awk '{print $3}' | cut -d. -f1-3
    elif command -v google-chrome-stable &> /dev/null; then
        google-chrome-stable --version | awk '{print $3}' | cut -d. -f1-3
    else
        echo "unknown"
    fi
}

# Function to setup ChromeDriver
setup_chromedriver() {
    local CHROME_VERSION=$(get_chrome_version)

    # Create directory if it doesn't exist
    mkdir -p "$CHROMEDRIVER_DIR"

    # Check if we already have the correct chromedriver version
    if [ -f "$CHROMEDRIVER" ] && [ -f "$CHROMEDRIVER_DIR/version.txt" ]; then
        local SAVED_VERSION=$(cat "$CHROMEDRIVER_DIR/version.txt")
        if [ "$SAVED_VERSION" == "$CHROME_VERSION" ]; then
            echo -e "${GREEN}ChromeDriver is already installed for Chrome version $CHROME_VERSION${NC}"
            return 0
        fi
    fi

    echo -e "${YELLOW}Setting up ChromeDriver for Chrome version $CHROME_VERSION${NC}"

    # Install Python and webdriver-manager if not already installed
    if ! command -v pip3 &> /dev/null; then
        echo -e "${YELLOW}Installing Python and pip${NC}"
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip
    fi

    # Install webdriver-manager if not already installed
    if ! pip3 list | grep -q webdriver-manager; then
        echo -e "${YELLOW}Installing webdriver-manager${NC}"
        pip3 install webdriver-manager
    fi

    # Download the compatible ChromeDriver
    echo -e "${YELLOW}Downloading ChromeDriver for Chrome version $CHROME_VERSION${NC}"
    CHROMEDRIVER_PATH=$(python3 -c "from webdriver_manager.chrome import ChromeDriverManager; print(ChromeDriverManager().install())")

    # Copy to our directory
    cp "$CHROMEDRIVER_PATH" "$CHROMEDRIVER"
    echo "$CHROME_VERSION" > "$CHROMEDRIVER_DIR/version.txt"

    echo -e "${GREEN}ChromeDriver installed at: $CHROMEDRIVER${NC}"
    return 0
}

# Function to check and kill any existing ChromeDriver processes
check_and_kill_chromedriver() {
    echo -e "${YELLOW}Checking for existing ChromeDriver processes...${NC}"
    local EXISTING_CHROMEDRIVER_PIDS=$(pgrep -f "chromedriver")

    if [ -n "$EXISTING_CHROMEDRIVER_PIDS" ]; then
        echo -e "${YELLOW}Found existing ChromeDriver processes with PIDs: $EXISTING_CHROMEDRIVER_PIDS${NC}"
        echo -e "${YELLOW}Killing existing ChromeDriver processes...${NC}"
        pkill -f "chromedriver" || true
        sleep 1
    else
        echo -e "${GREEN}No existing ChromeDriver processes found.${NC}"
    fi
}

# Function to run web tests
run_web_tests() {
    # Check if Chrome is installed
    if ! check_chrome_installed; then
        echo -e "${RED}Chrome browser not found. Skipping web tests.${NC}"
        return 1
    fi

    # Setup ChromeDriver
    if ! setup_chromedriver; then
        echo -e "${RED}Failed to setup ChromeDriver. Skipping web tests.${NC}"
        return 1
    fi

    # Check and kill any existing ChromeDriver processes
    check_and_kill_chromedriver

    echo -e "${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Running integration tests on Web platform${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"

    # Start ChromeDriver on port 4444
    "$CHROMEDRIVER" --port=4444 &
    CHROMEDRIVER_PID=$!

    # Give ChromeDriver time to start
    sleep 3

    # Run the integration tests
    local TEST_RESULT=0

    flutter drive \
        --driver=test_driver/integration_test.dart \
        --target=integration_test/all_tests.dart \
        -d web-server \
        --web-port=8080 \
        --browser-name=chrome || TEST_RESULT=$?

    # Make sure to kill our ChromeDriver process
    if [ -n "$CHROMEDRIVER_PID" ]; then
        echo -e "${YELLOW}Stopping ChromeDriver (PID: $CHROMEDRIVER_PID)...${NC}"
        kill $CHROMEDRIVER_PID 2>/dev/null || true
    fi

    # Also do a final check to make sure no chromedriver processes are left
    check_and_kill_chromedriver

    if [ $TEST_RESULT -ne 0 ]; then
        echo -e "${RED}Web tests failed with exit code: $TEST_RESULT${NC}"
        return $TEST_RESULT
    fi

    echo -e "${GREEN}Web tests completed successfully!${NC}"
    return 0
}

# Function to run Android tests
run_android_tests() {
    # Test on Android devices
    ANDROID_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "android" | awk '{print $1}')
    if [ -n "$ANDROID_DEVICES" ]; then
        # Take the first Android device
        ANDROID_DEVICE=$(echo "$ANDROID_DEVICES" | head -n 1)
        run_integration_test_on_device "$ANDROID_DEVICE" "Android"
    else
        echo -e "${RED}No Android devices found. Skipping Android tests.${NC}"
    fi
}

# Function to run iOS tests
run_ios_tests() {
    # Test on iOS devices
    IOS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "ios|iPhone|iPad" | awk '{print $1}')
    if [ -n "$IOS_DEVICES" ]; then
        # Take the first iOS device
        IOS_DEVICE=$(echo "$IOS_DEVICES" | head -n 1)
        run_integration_test_on_device "$IOS_DEVICE" "iOS"
    else
        echo -e "${RED}No iOS devices found. Skipping iOS tests.${NC}"
    fi
}

# Function to run the native Linux unit tests (ctest), built by the Linux integration test run above.
run_linux_native_tests() {
    echo -e "${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Running native unit tests (ctest)${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"
    # The issue #73 locale test skips unless a comma-decimal locale is installed; to exercise it,
    # run `sudo locale-gen de_DE.UTF-8` (CI additionally sets FLUTTER_ONNXRUNTIME_REQUIRE_COMMA_LOCALE=1
    # so that the skip becomes a failure there).
    ctest --test-dir build/linux/x64/debug/plugins/flutter_onnxruntime --output-on-failure --no-tests=error
}

# Function to run Linux tests
run_linux_tests() {
    # Test on Linux desktop
    LINUX_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "linux" | awk '{print $1}')
    if [ -n "$LINUX_DEVICES" ]; then
        # Take the first Linux device (usually just "linux")
        LINUX_DEVICE=$(echo "$LINUX_DEVICES" | head -n 1)
        run_integration_test_on_device "$LINUX_DEVICE" "Linux"
        run_linux_native_tests
    else
        echo -e "${RED}Linux desktop not available. Skipping Linux tests.${NC}"
    fi
}

# Function to run macOS tests
run_macos_tests() {
    # Test on macOS desktop
    MACOS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "macos" | awk '{print $1}')
    if [ -n "$MACOS_DEVICES" ]; then
        # Take the first macOS device (usually just "macos")
        MACOS_DEVICE=$(echo "$MACOS_DEVICES" | head -n 1)
        run_integration_test_on_device "$MACOS_DEVICE" "macOS"
    else
        echo -e "${RED}macOS desktop not available. Skipping macOS tests.${NC}"
    fi
}

# Function to run Windows tests
run_window_tests() {
    # Test on Windows desktop
    WINDOWS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "windows" | awk '{print $1}')
    if [ -n "$WINDOWS_DEVICES" ]; then
        # Take the first Windows device (usually just "windows")
        WINDOWS_DEVICE=$(echo "$WINDOWS_DEVICES" | head -n 1)
        run_integration_test_on_device "$WINDOWS_DEVICE" "Windows"
    else
        echo -e "${RED}Windows desktop not available. Skipping Windows tests.${NC}"
    fi
}

# Function to run all tests
run_all_tests() {
    run_android_tests
    run_ios_tests
    run_linux_tests
    run_macos_tests
    run_window_tests
    run_web_tests
}

# Run unit tests
echo -e "${BLUE}Running flutter_onnxruntime tests...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Running unit tests...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
flutter test test/unit

echo -e "${BLUE}Running integration tests...${NC}"

cd example

# Get all available devices
AVAILABLE_DEVICES=$(flutter devices)
echo -e "${YELLOW}Available devices:${NC}"
echo "$AVAILABLE_DEVICES"
echo ""

# Display menu for selecting test platform
echo -e "${BLUE}Select a platform to run integration tests:${NC}"
echo "1) Run on all available devices"
echo "2) Run on Android"
echo "3) Run on iOS"
echo "4) Run on Linux"
echo "5) Run on macOS"
echo "6) Run on Web"
echo "7) Run on Windows"
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        run_all_tests
        ;;
    2)
        run_android_tests
        ;;
    3)
        run_ios_tests
        ;;
    4)
        run_linux_tests
        ;;
    5)
        run_macos_tests
        ;;
    6)
        run_web_tests
        ;;
    7)
        run_window_tests
        ;;
    *)
        echo -e "${RED}Invalid option. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}All tests completed!${NC}"
