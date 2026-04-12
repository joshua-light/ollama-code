$ErrorActionPreference = "Stop"

$BinaryName = "ollama-code"

# Resolve the repo root (scripts/ lives one level down).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir

# Check dependencies.
if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
    Write-Error "'cargo' is not installed. Install Rust: https://rustup.rs"
    exit 1
}

# Set install directory.
$InstallDir = if ($env:INSTALL_DIR) { $env:INSTALL_DIR } else { "$env:USERPROFILE\.cargo\bin" }

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# Build the release binary.
Write-Host "Building $BinaryName..."
cargo build --release --manifest-path "$RepoDir\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

# Install the binary.
$BinaryPath = "$RepoDir\target\release\$BinaryName.exe"
if (-not (Test-Path $BinaryPath)) {
    throw "Build failed, binary not found at $BinaryPath"
}

Write-Host "Installing $BinaryName to $InstallDir..."
Copy-Item $BinaryPath "$InstallDir\$BinaryName.exe" -Force

Write-Host ""
Write-Host "Done! Installed $BinaryName to $InstallDir\$BinaryName.exe"
Write-Host ""
Write-Host "Make sure Ollama is running (ollama serve) and you have a model pulled."
Write-Host "Run '$BinaryName' to start."

# Check if install dir is in PATH.
if ($env:PATH -notlike "*$InstallDir*") {
    Write-Host ""
    Write-Host "Warning: '$InstallDir' is not in your PATH."
    Write-Host "Add it with: `$env:PATH += `";$InstallDir`""
}
