# ComfyUI MCP Server

A lightweight Python-based MCP (Model Context Protocol) server that interfaces with a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance to generate images programmatically via AI agent requests.

## Overview

This project enables AI agents to send generation requests to ComfyUI using the MCP protocol over WebSocket. It supports:
- Flexible workflow selection (e.g., the bundled `generate_image.json` and `generate_song.json`).
- Dynamic parameters (text prompts, tags, lyrics, dimensions, etc.) inferred from workflow placeholders.
- Automatic asset URL routing—image workflows return PNG/JPEG URLs, audio workflows return MP3 URLs.

## Prerequisites

- **Python 3.10+**
- **ComfyUI**: Installed and running locally (e.g., on `localhost:8188`).
- **Dependencies**: `requests`, `websockets`, `mcp` (install via pip).

## Setup

1. **Clone the Repository**:
   git clone <your-repo-url>
   cd comfyui-mcp-server

2. **Install Dependencies**:

   pip install requests websockets mcp


3. **Start ComfyUI**:
- Install ComfyUI (see [ComfyUI docs](https://github.com/comfyanonymous/ComfyUI)).
- Run it on port 8188:
  ```
  cd <ComfyUI_dir>
  python main.py --port 8188
  ```

4. **Prepare Workflows**:
- Place API-format workflow files (e.g., `generate_image.json`, `generate_song.json`, or your own) in the `workflows/` directory.
- Export workflows from ComfyUI’s UI with “Save (API Format)” (enable dev mode in settings).

## Usage

1. **Run the MCP Server**:
   python server.py

- Listens on `ws://localhost:9000`.

2. **Test with the Client**:
   python client.py

- Sends a sample request: `"a dog wearing sunglasses"` with `512x512` using `sd_xl_base_1.0.safetensors`.
- Output example:
  ```
  Response from server:
  {
    "image_url": "http://localhost:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output"
  }
  ```

- Modify `client.py`’s `payload` to change `prompt`, `width`, `height`, `workflow_id`, or model-specific settings.
- Example:
  ```
  "params": json.dumps({
      "prompt": "a cat in space",
      "width": 768,
      "height": 768,
      "workflow_id": "generate_image",
      "model": "v1-5-pruned-emaonly.ckpt"
  })
  ```

### Bundled example workflows

- `generate_image.json`: Minimal Stable Diffusion 1.5 image sampler that exposes `prompt`, `width`, `height`, and `model` parameters. Produces PNG URLs.
- `generate_song.json`: AceStep audio text-to-song workflow that exposes `tags` and `lyrics` parameters and returns an MP3 URL.

Add additional API-format workflows following the placeholder convention below to expose new MCP tools automatically.

### Workflow-backed MCP tools

- Any workflow JSON placed in `workflows/` that contains placeholders such as `PARAM_PROMPT`, `PARAM_TAGS`, or `PARAM_LYRICS` is exposed automatically as an MCP tool.
- Placeholders live inside node inputs and follow the convention `PARAM_<TYPE?>_<NAME>` where `<TYPE?>` is optional. Supported type hints: `STR`, `STRING`, `TEXT`, `INT`, `FLOAT`, and `BOOL`.
- Example: `"tags": "PARAM_TAGS"` creates a `tags: str` argument, while `"steps": "PARAM_INT_STEPS"` becomes an `int` argument.
- The tool name defaults to the workflow filename (normalized to snake_case). Rename the JSON file if you want a friendlier MCP tool name.
- Outputs are inferred heuristically: workflows that contain audio nodes return audio URLs, otherwise image URLs are returned.
- Add more workflows and they will show up without extra Python changes, provided they use the placeholder convention above.

## Project Structure

- `server.py`: MCP server with WebSocket transport and lifecycle support.
- `comfyui_client.py`: Interfaces with ComfyUI’s API, handles workflow queuing.
- `client.py`: Test client for sending MCP requests.
- `workflows/`: Directory for API-format workflow JSON files.

## Notes

- Ensure your chosen `model` (e.g., `v1-5-pruned-emaonly.ckpt`) exists in `<ComfyUI_dir>/models/checkpoints/`.
- The MCP SDK lacks native WebSocket transport; this uses a custom implementation.
- For custom workflows, adjust node IDs in `comfyui_client.py`’s `DEFAULT_MAPPING` if needed.

## Contributing

Feel free to submit issues or PRs to enhance flexibility (e.g., dynamic node mapping, progress streaming).

## License

Apache License