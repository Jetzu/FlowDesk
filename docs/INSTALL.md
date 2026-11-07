# Installing FlowDesk

FlowDesk runs on Windows 10/11, Linux (X11 or Wayland) and macOS 11+. You can
either **download a built app** (no Python needed) or **run it from source**.

- [Download a built app](#download-a-built-app)
- [Windows from source](#windows)
- [Linux from source](#linux)
- [macOS from source](#macos)
- [Building the standalone app yourself](#building-the-standalone-app-yourself)
- [Troubleshooting](#troubleshooting)

---

## Download a built app

Every tagged version (`v*`) is built automatically on GitHub Actions for all three
platforms. Open the repository's **Releases** page and download the zip for your
system:

| System  | File                   | Start it with                       |
|---------|------------------------|-------------------------------------|
| Windows | `FlowDesk-windows.zip` | `FlowDesk\FlowDesk.exe`             |
| Linux   | `FlowDesk-linux.zip`   | `FlowDesk/FlowDesk`                 |
| macOS   | `FlowDesk-macos.zip`   | `FlowDesk.app`                      |

1. Unzip it to a folder you can write to (Windows: e.g. `%LOCALAPPDATA%\Programs\FlowDesk`;
   Linux: e.g. `~/Applications/FlowDesk`; macOS: drag `FlowDesk.app` to *Applications*).
   Keep the whole folder together: the executable needs the files next to it.
2. Start it:
   - **Windows:** double-click `FlowDesk.exe`. The builds are not code-signed, so
     SmartScreen may say *"Windows protected your PC"*: click **More info → Run anyway**.
     Right-click `FlowDesk.exe` → *Pin to Start / taskbar* for quick access.
   - **Linux:** `chmod +x FlowDesk/FlowDesk` if needed, then run it. Install the
     [Qt system libraries](#linux) first if it fails to start. To add a menu entry:
     `packaging/install_linux_desktop.sh /path/to/FlowDesk/FlowDesk` (from a source checkout).
   - **macOS:** the app is not notarized. The first time, **right-click → Open → Open**
     (or run `xattr -dr com.apple.quarantine /Applications/FlowDesk.app`).

---

## Windows

1. Install **Python 3.12** from <https://www.python.org/downloads/windows/>
   (the "Windows installer (64-bit)"). On the first installer page tick
   **"Add python.exe to PATH"**; the `py` launcher is installed as well.
2. Get the source: *Code → Download ZIP* on GitHub and unzip it, or
   `git clone <repository-url>` in a terminal.
3. Double-click **`flowdesk.bat`** in the FlowDesk folder. On the first run it creates
   the virtual environment `.venv` and installs the dependencies (a few minutes),
   then starts FlowDesk without a console window. Later starts are immediate.
   - To see error messages, run it from *Command Prompt* with
     `set FLOWDESK_CONSOLE=1` followed by `flowdesk.bat`.
   - You can drag `.fdw`, `.wsp` or `.fcs` files onto `flowdesk.bat`.

Manual alternative (Command Prompt or PowerShell, in the FlowDesk folder):

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m flowdesk
```

`pip install -e .` also creates `.venv\Scripts\flowdesk-gui.exe`, which starts
FlowDesk without a console (useful for a desktop shortcut). If
[uv](https://docs.astral.sh/uv/) is installed, `flowdesk.bat` uses it automatically.

PowerShell may refuse to run `Activate.ps1`; you don't need to activate the
environment - call `.venv\Scripts\python` directly as shown.

---

## Linux

Tested on Ubuntu 22.04/24.04 and Debian 12; other distributions need the equivalent
packages.

1. Python and the Qt runtime libraries (PySide6 ships Qt itself but needs these from
   the system):

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip \
       libxcb-cursor0 libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 fontconfig
   ```

   Fedora: `sudo dnf install python3 xcb-util-cursor mesa-libEGL mesa-libGL libxkbcommon-x11 dbus-libs fontconfig`.
   Arch: `sudo pacman -S python xcb-util-cursor libglvnd libxkbcommon-x11 dbus fontconfig`.

   Python 3.11 or newer is required; on older distributions install 3.12 with
   [uv](https://docs.astral.sh/uv/) (`uv python install 3.12`), which `flowdesk.sh` uses
   automatically when present.

2. Get the source and start it:

   ```bash
   git clone <repository-url> FlowDesk
   cd FlowDesk
   ./flowdesk.sh          # first run creates .venv and installs the dependencies
   ```

   Manual alternative: `python3 -m venv .venv && .venv/bin/python -m pip install -e . && .venv/bin/python -m flowdesk`.

3. Optional - application menu entry with icon:

   ```bash
   packaging/install_linux_desktop.sh
   ```

   (installs `~/.local/share/applications/flowdesk.desktop` and the icon).

**Wayland:** FlowDesk uses Qt's native Wayland backend when available. If the window
has no decorations, menus appear in the wrong place or it fails to start with a
Wayland error, use XWayland instead:

```bash
QT_QPA_PLATFORM=xcb ./flowdesk.sh
```

If Qt reports *"Could not load the Qt platform plugin xcb"*, a library from step 1 is
missing; `QT_DEBUG_PLUGINS=1 ./flowdesk.sh` names it.

---

## macOS

1. Install Python 3.12 from <https://www.python.org/downloads/macos/> or with
   Homebrew (`brew install python@3.12`) or uv (`brew install uv`).
2. Get the source and start it:

   ```bash
   git clone <repository-url> FlowDesk
   cd FlowDesk
   ./flowdesk.sh
   ```

   or double-click **`FlowDesk.command`** in Finder (the first time: right-click → Open).

---

## Building the standalone app yourself

On the platform you want to build for (PyInstaller does not cross-compile):

```bash
python -m pip install -e ".[packaging]"     # or let build.py install PyInstaller
python packaging/build.py --zip
```

Output: `dist/FlowDesk.app` (macOS), `dist/FlowDesk/FlowDesk.exe` (Windows),
`dist/FlowDesk/FlowDesk` (Linux), plus `dist/FlowDesk-<os>.zip`. The macOS build is
about 130 MB. `packaging/make_icons.py` regenerates `icon.png`/`.ico`/`.icns` from the
built-in artwork (the `.icns` needs macOS' `iconutil`). Pushing a tag `v*` runs
`.github/workflows/release.yml`, which builds all three and attaches the zips to the
GitHub release.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Text or icons tiny / huge on a scaled display | FlowDesk follows the system scale (fractional 125 %/150 % supported). Override with `QT_SCALE_FACTOR=1.25`. |
| Windows taskbar shows the Python icon | Start via `flowdesk.bat`, `flowdesk-gui.exe` or the built `FlowDesk.exe` (they set the FlowDesk app id). |
| `No module named flowdesk` | Run with the environment's Python (`.venv/bin/python -m flowdesk` / `.venv\Scripts\python -m flowdesk`), or reinstall with `pip install -e .`. |
| Settings | Stored by Qt: registry `HKCU\Software\FlowDesk` (Windows), `~/.config/FlowDesk/FlowDesk.conf` (Linux), `~/Library/Preferences/com.flowdesk.FlowDesk.plist` (macOS). |
| Workspaces moved to another computer | `.fdw` files store each FCS path plus a path relative to the workspace; keep the FCS files next to (or below) the workspace, or FlowDesk asks for the folder. |
