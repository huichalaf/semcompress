"""Entry point for `python -m semcompress`.

Starts the MCP server for Claude Code integration.
Requires the [mcp] extra: pip install semcompress[mcp]
"""

import sys


def main():
    try:
        from semcompress.mcp_server import main as run_server
    except ImportError:
        print(
            "Error: MCP dependencies not installed.\n"
            "Install them with: pip install semcompress[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Starting semcompress MCP server (stdio transport)...", file=sys.stderr)
    run_server()


if __name__ == "__main__":
    main()
