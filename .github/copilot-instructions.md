## Copilot notes for this repo

- Prefer **uv** for Python operations.
  - Use `uv sync` to install dependencies from `pyproject.toml`/`uv.lock`.
  - Use `uv run ...` to run scripts/tests in the managed environment.
- Prefer **uvx** for running the published entrypoint from git.
  - Example: `uvx --from git+https://github.com/venetanji/comfyui-mcp-server comfyui-mcp-server`

### MCP transport defaults

- The default transport is **stdio** (best for MCP client integrations).
- HTTP transport is opt-in via `--transport streamable-http` or `MCP_TRANSPORT=streamable-http`.
