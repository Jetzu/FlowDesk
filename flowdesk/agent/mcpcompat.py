"""Import shim for the MCP Python SDK: v2 renamed FastMCP to MCPServer."""
try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer as FastMCP  # type: ignore
    MCP_MAJOR = 2
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP, Image  # type: ignore
    MCP_MAJOR = 1

__all__ = ["FastMCP", "Image", "MCP_MAJOR"]
