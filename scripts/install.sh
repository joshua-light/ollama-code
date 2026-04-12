#!/usr/bin/env bash
set -euo pipefail

BINARY_NAME="ollama-code"

# Resolve the repo root (scripts/ lives one level down).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detect OS and set install directory.
OS="$(uname -s)"
case "$OS" in
    Linux)  INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}" ;;
    Darwin) INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}" ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Check dependencies.
if ! command -v cargo &>/dev/null; then
    echo "Error: 'cargo' is not installed."
    echo "Install Rust: https://rustup.rs"
    exit 1
fi

# Build the release binary.
echo "Building $BINARY_NAME..."
cargo build --release --manifest-path "$REPO_DIR/Cargo.toml"

# Install the binary.
BINARY_PATH="$REPO_DIR/target/release/$BINARY_NAME"
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: build failed, binary not found at $BINARY_PATH"
    exit 1
fi

echo "Installing $BINARY_NAME to $INSTALL_DIR..."
if [ -w "$INSTALL_DIR" ]; then
    cp "$BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"
else
    echo "(requires sudo)"
    sudo cp "$BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"
    sudo chmod +x "$INSTALL_DIR/$BINARY_NAME"
fi

echo "Done! Installed $BINARY_NAME to $INSTALL_DIR/$BINARY_NAME"
echo ""
echo "Make sure Ollama is running (ollama serve) and you have a model pulled."
echo "Run '$BINARY_NAME' to start."
