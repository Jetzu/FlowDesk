"""FlowDesk for agents: an MCP server, an in-app bridge and an Agent Chat panel.

    flowdesk.agent.server      the MCP server (``flowdesk-mcp``): tools for Claude Code, Codex and Claude Desktop
    flowdesk.agent.bridgecore  the localhost bridge that lives inside the running application
    flowdesk.agent.ops         everything an agent can do, as plain functions on a workspace controller
    flowdesk.agent.hostapp     wires the bridge to a controller (window or headless) and runs ops on the GUI thread
    flowdesk.agent.chat_ui     the Agent Chat dock (runs your installed Claude Code or Codex CLI)
    flowdesk.agent.install     registers the server with Claude Code, Codex and Claude Desktop

The design follows the molmcp servers for PyMOL and ChimeraX and the ChemDraw server: the server never owns the
data, it talks to a bridge inside the application, so an agent works on the same window the user is looking at.
"""
from __future__ import annotations

APP = "flowdesk"
SERVER_NAME = "flowdesk"
PID_ENV = "FLOWDESK_PID"  # set by the in-app chat so the server attaches to exactly that window
