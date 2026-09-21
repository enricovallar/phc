#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking for existing Conda installation ==="
if command -v conda &>/dev/null; then
    echo "Conda is already installed at: $(which conda)"
    conda --version
    exit 0
fi

if [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Miniconda found at $HOME/miniconda3/bin/conda. Initializing shell..."
    "$HOME/miniconda3/bin/conda" init bash
    exit 0
fi

echo "=== Downloading and Installing Miniconda ==="
INSTALL_DIR="$HOME/miniconda3"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
TEMP_INSTALLER="/tmp/miniconda_installer.sh"

echo "Downloading installer from $INSTALLER_URL..."
curl -fsSL "$INSTALLER_URL" -o "$TEMP_INSTALLER"

echo "Installing Miniconda to $INSTALL_DIR..."
bash "$TEMP_INSTALLER" -b -u -p "$INSTALL_DIR"
rm -f "$TEMP_INSTALLER"

echo "Initializing Conda for bash..."
"$INSTALL_DIR/bin/conda" init bash

echo "=== Miniconda installation complete! ==="
echo "Run 'source ~/.bashrc' or open a new terminal to use 'conda'."
