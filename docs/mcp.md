# Claude Code, Codex and Claude Desktop with FlowDesk

FlowDesk ships an MCP server (`flowdesk-mcp`) and an in-app bridge. An AI assistant then works on the **same window you are
looking at**: it reads the workspace, looks at histograms and plots (a PNG plus numbers), gates, compensates, clusters,
measures and exports. Every change is an undo step in FlowDesk, and what you do in the window is reported back to the
assistant.

```
Claude Code / Codex / Claude Desktop  <-stdio->  flowdesk-mcp  <-localhost + token->  bridge inside FlowDesk
```

## Set up

```bash
pip install -e ".[agent]"            # from a source checkout (adds the `mcp` package)
flowdesk-mcp-install                 # registers FlowDesk with every agent it finds
flowdesk-mcp-install --check         # what is detected and registered
flowdesk-mcp-install --dry-run       # what would change
flowdesk-mcp-install --uninstall
```

The installer detects Claude Code (`~/.claude.json`), Codex (`~/.codex/config.toml`) and Claude Desktop, backs up each
config (`*.flowdesk-backup`) and only touches its own `flowdesk` entry. Name clients explicitly with `--claude`, `--codex`,
`--claude-desktop`. In the app: **Tools ▸ Agent ▸ Connect Claude / Codex…** does the same.

From the packaged app (macOS DMG) no Python is needed: the entry is the app's own executable,
`/Applications/FlowDesk.app/Contents/MacOS/FlowDesk --mcp`. Re-run the installer after moving the app.
(On Windows the packaged app is a windowed program without console streams; use the pip install there.)

Then, in Claude Code or Codex:

> Open the FlowDesk test set, gate lymphocytes and single cells, then CD4 and CD8 inside CD3, and give me the CD4/CD8
> ratio per donor and condition.

If no FlowDesk is running the server starts one. `FLOWDESK_AUTOLAUNCH=0` switches that off.

## Agent Chat inside FlowDesk

**Tools ▸ Agent ▸ Agent Chat** (Ctrl+Shift+A) docks a chat panel that runs your installed Claude Code or Codex CLI against
this window. Providers whose CLI is not installed are greyed out; model and effort are chosen with `/model` or the button at
the bottom right. Slash commands: `/model`, `/clear`, `/claude`, `/codex`, `/skills`, `/stop`, `/help`. Enter sends,
Shift+Enter inserts a newline, Esc interrupts.

The panel starts the CLI with `FLOWDESK_PID` set, so its MCP server attaches to exactly this window even when several are
open. The CLI is found on `PATH`, in the Claude desktop app's bundled Claude Code, the VS Code / Cursor extension,
`~/.claude/local`, or (Codex) the ChatGPT app; override with `FLOWDESK_CLAUDE_EXE` / `FLOWDESK_CODEX_EXE`. Both CLIs use your
own account and plan.

## Headless

`flowdesk --agent-headless [workspace.fdw | files...]` runs FlowDesk without a window (servers, batch jobs); `launch(mode="headless")`
does the same from the agent. Everything except what needs a window works, and plots still render.

## Tools

| Group | Tools |
|---|---|
| Connection | `status`, `launch`, `connect`, `quit_flowdesk` |
| State | `workspace`, `sample_info`, `populations`, `open_workspace`, `new_workspace`, `save_workspace`, `add_samples`, `remove_samples`, `set_keyword`, `create_group`, `delete_group` |
| Gating | `create_gate` (rectangle, range, polygon, ellipse, boolean), `edit_gate`, `rename_population`, `delete_population`, `set_scale` |
| Looking | `histogram` (bins, percentiles, peaks, valley threshold), `plot` (PNG + density map, axes, gates with counts), `statistics` |
| Analysis | `compensation` (list / compute / apply / assign), `dimension_reduction` (t-SNE, UMAP, PCA), `cluster` (FlowSOM, k-means, Phenograph), `compare_populations`, `cell_cycle`, `proliferation`, `data_cleaning` |
| Output | `export_events` (FCS / CSV), `export_statistics`, `export_flowjo_workspace` |
| Control | `undo`, `redo`, `python`, `events`, `log` |
| Skills | `skills`, `skill`, `save_skill` |

Conventions: samples are named like their file (loosely spelled, or an id); populations are paths such as
`/Lymphocytes/Single cells/CD3+` or a unique name; parameters can be given by stain (`CD4` resolves to the compensated
detector); gate values are raw axis units; frequencies are percent. Nothing is saved until `save_workspace`.

## Skills

`flowdesk/agent/skills/*.md` are how-to notes the agent reads before a non-trivial job: gating strategy, compensation, batch
analysis, troubleshooting gates, dimensionality reduction, quality control, opening FlowJo workspaces, the Python API.
`save_skill` stores your own in `~/.flowdesk/skills`.

## Safety

* The bridge listens on `127.0.0.1` only, requires a per-session token (kept in a `0600` discovery file under
  `~/.flowdesk/instances`), and can be switched off in **Tools ▸ Agent ▸ Agent Bridge** (or `FLOWDESK_BRIDGE=0`).
* The assistant cannot close your window without `confirm=true`, and not at all while there are unsaved changes unless it
  is told to discard them. `open_workspace` and `new_workspace` refuse to replace unsaved work the same way.
* Everything an assistant changes goes through the undo stack. Data sent to the assistant's provider is governed by that
  provider's terms: do not point an assistant at data you may not share (see [LEGAL.md](../LEGAL.md)).

## Environment variables

| Variable | Meaning |
|---|---|
| `FLOWDESK_PID` | pin the server to one window (set by Agent Chat) |
| `FLOWDESK_NAME` | name of a window (also `connect(name=...)`) |
| `FLOWDESK_PORT` | fixed bridge port |
| `FLOWDESK_BRIDGE` | `0` switches the bridge off, `1` forces it on |
| `FLOWDESK_AUTOLAUNCH` | `0`: never start FlowDesk from the server |
| `FLOWDESK_EXE` | FlowDesk executable to launch |
| `FLOWDESK_HOME` | state folder, default `~/.flowdesk` |
| `FLOWDESK_OUTPUT_DIR` | where plots are saved |
| `FLOWDESK_CLAUDE_EXE`, `FLOWDESK_CODEX_EXE` | CLI locations for Agent Chat |
