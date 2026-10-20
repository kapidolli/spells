#!/usr/bin/env bash
# Sets up Spells from this checkout for the signed-in user on Linux (KDE Plasma first):
# a Python 3.12 venv, the engines and models, the udev rule, and a menu entry.
# Only the udev rule needs sudo. Run it again after pulling to refresh everything.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"

missing=()
for tool in uv wl-copy wl-paste cmake ninja g++ glslc; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
    echo "Missing tools: ${missing[*]}"
    echo "On Fedora: sudo dnf install uv wl-clipboard cmake ninja-build gcc-c++ glslc vulkan-headers"
    exit 1
fi

uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]" jeepney

.venv/bin/python build/linux_engines.py

rules=/etc/udev/rules.d/70-spells.rules
if ! cmp -s build/linux/70-spells.rules "$rules"; then
    echo "Installing $rules (sudo) so Spells can read the keyboard and send the paste shortcut"
    sudo install -m 0644 build/linux/70-spells.rules "$rules"
    sudo udevadm control --reload
    sudo udevadm trigger --subsystem-match=input --action=change
    sudo udevadm trigger --subsystem-match=misc --action=change
fi

icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$icons" "$apps"
install -m 0644 build/brand/spells-256.png "$icons/spells.png"
cat >"$apps/spells.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Spells
Comment=Offline voice dictation
Exec="$repo/.venv/bin/python" -m spells
Icon=spells
Terminal=false
Categories=Utility;Accessibility;
StartupNotify=false
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" || true

echo "Done. Start Spells from the menu, then hold Ctrl+Super and speak."
