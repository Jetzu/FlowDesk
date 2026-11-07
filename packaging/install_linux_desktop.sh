#!/usr/bin/env bash
# Add FlowDesk to the Linux application menu (per user, no root needed).
#   packaging/install_linux_desktop.sh            # source checkout: runs ./flowdesk.sh
#   packaging/install_linux_desktop.sh /opt/FlowDesk/FlowDesk   # a built app
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXEC="${1:-$(dirname "$HERE")/flowdesk.sh}"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
mkdir -p "$APPS" "$ICONS"
cp "$HERE/icon.png" "$ICONS/flowdesk.png"
sed -e "s|^Exec=.*|Exec=\"$EXEC\" %F|" "$HERE/flowdesk.desktop" > "$APPS/flowdesk.desktop"
chmod +x "$APPS/flowdesk.desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "${ICONS%/256x256/apps}" || true
echo "Installed $APPS/flowdesk.desktop (Exec=$EXEC)"
