import argparse
import base64
import copy
import inspect
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult
from fastmcp.server.context import Context
from comfyui_client import ComfyUIClient
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP_Server")

PLACEHOLDER_PREFIX = "PARAM_"
PLACEHOLDER_TYPE_HINTS = {
    "STR": str,
    "STRING": str,
    "TEXT": str,
    "INT": int,
    "FLOAT": float,
    "BOOL": bool,
}
PLACEHOLDER_DESCRIPTIONS = {
    "prompt": "Main text prompt used inside the workflow.",
    "width": "Output image width in pixels.",
    "height": "Output image height in pixels.",
    "image_filename": "Filename of the source image to load (from ComfyUI outputs).",
    "tags": "Comma-separated descriptive tags for the audio model.",
    "lyrics": "Full lyric text that should drive the audio generation.",
}
DEFAULT_OUTPUT_KEYS = ("images", "image", "gifs", "gif", "files")
AUDIO_OUTPUT_KEYS = ("audio", "audios", "sound", "files")
WORKFLOW_DIR = Path(__file__).parent / "workflows"
DEFAULT_COMFY_URL = os.getenv("COMFY_URL", "http://thor:8188")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "9000"))
OUTPUT_FOLDER: Optional[Path] = None


@dataclass
class WorkflowParameter:
    name: str
    placeholder: str
    annotation: type
    description: str
    bindings: list[Tuple[str, str]] = field(default_factory=list)
    required: bool = True
    is_image_input: bool = False


@dataclass
class WorkflowToolDefinition:
    workflow_id: str
    tool_name: str
    description: str
    template: Dict[str, Any]
    parameters: "OrderedDict[str, WorkflowParameter]"
    output_preferences: Sequence[str]


class WorkflowManager:
    def __init__(self, workflows_dir: Path):
        self.workflows_dir = workflows_dir
        self._tool_names: set[str] = set()
        self.tool_definitions = self._load_workflows()

    def _load_workflows(self):
        definitions: list[WorkflowToolDefinition] = []
        if not self.workflows_dir.exists():
            logger.info("Workflow directory %s does not exist yet", self.workflows_dir)
            return definitions

        for workflow_path in sorted(self.workflows_dir.glob("*.json")):
            try:
                with open(workflow_path, "r", encoding="utf-8") as handle:
                    workflow = json.load(handle)
            except json.JSONDecodeError as exc:
                logger.error("Skipping workflow %s due to JSON error: %s", workflow_path.name, exc)
                continue

            parameters = self._extract_parameters(workflow)
            if not parameters:
                logger.info(
                    "Workflow %s has no %s placeholders; skipping auto-tool registration",
                    workflow_path.name,
                    PLACEHOLDER_PREFIX,
                )
                continue

            tool_name = self._dedupe_tool_name(self._derive_tool_name(workflow_path.stem))
            definition = WorkflowToolDefinition(
                workflow_id=workflow_path.stem,
                tool_name=tool_name,
                description=self._derive_description(workflow_path.stem),
                template=workflow,
                parameters=parameters,
                output_preferences=self._guess_output_preferences(workflow),
            )
            logger.info(
                "Prepared workflow tool '%s' from %s with params %s",
                tool_name,
                workflow_path.name,
                list(parameters.keys()),
            )
            definitions.append(definition)

        return definitions

    def render_workflow(self, definition: WorkflowToolDefinition, provided_params: Dict[str, Any], comfyui_client, output_folder: Optional[Path] = None):
        workflow = copy.deepcopy(definition.template)
        for param in definition.parameters.values():
            if param.required and param.name not in provided_params:
                raise ValueError(f"Missing required parameter '{param.name}'")
            raw_value = provided_params[param.name]
            
            # Handle automatic image upload for LoadImage nodes
            if param.is_image_input:
                from pathlib import Path
                file_path = Path(raw_value)
                
                # If not an absolute path or doesn't exist, try looking in output folder
                if not file_path.is_absolute() or not file_path.exists():
                    if output_folder:
                        candidate = output_folder / raw_value
                        if candidate.exists():
                            file_path = candidate
                            logger.info(f"Found image in output folder: {file_path}")
                
                if not file_path.exists():
                    raise FileNotFoundError(f"Image file not found: {raw_value} (also checked in output folder: {output_folder})")
                
                logger.info(f"Auto-uploading image for parameter '{param.name}': {file_path}")
                upload_result = comfyui_client.upload_image(str(file_path))
                # Use just the filename that ComfyUI assigned
                coerced_value = upload_result["asset"]["filename"]
                logger.info(f"Image uploaded as: {coerced_value}")
            else:
                coerced_value = self._coerce_value(raw_value, param.annotation)
            
            for node_id, input_name in param.bindings:
                workflow[node_id]["inputs"][input_name] = coerced_value
        return workflow

    def _extract_parameters(self, workflow: Dict[str, Any]):
        parameters: "OrderedDict[str, WorkflowParameter]" = OrderedDict()
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            node_class = node.get("class_type", "")
            is_load_image = node_class == "LoadImage"
            
            for input_name, value in inputs.items():
                parsed = self._parse_placeholder(value)
                if not parsed:
                    continue
                param_name, annotation, placeholder_value = parsed
                description = PLACEHOLDER_DESCRIPTIONS.get(
                    param_name, f"Value for '{param_name}'."
                )
                parameter = parameters.get(param_name)
                if not parameter:
                    parameter = WorkflowParameter(
                        name=param_name,
                        placeholder=placeholder_value,
                        annotation=annotation,
                        description=description,
                        is_image_input=is_load_image,
                    )
                    parameters[param_name] = parameter
                parameter.bindings.append((node_id, input_name))
        return parameters

    def _parse_placeholder(self, value):
        if not isinstance(value, str) or not value.startswith(PLACEHOLDER_PREFIX):
            return None
        token = value[len(PLACEHOLDER_PREFIX) :]
        annotation = str
        if "_" in token:
            type_candidate, remainder = token.split("_", 1)
            type_hint = PLACEHOLDER_TYPE_HINTS.get(type_candidate.upper())
            if type_hint:
                annotation = type_hint
                token = remainder
        param_name = self._normalize_name(token)
        return param_name, annotation, value

    def _normalize_name(self, raw: str):
        cleaned = [
            (char.lower() if char.isalnum() else "_")
            for char in raw.strip()
        ]
        normalized = "".join(cleaned).strip("_")
        return normalized or "param"

    def _derive_tool_name(self, stem: str):
        return self._normalize_name(stem)

    def _dedupe_tool_name(self, base_name: str):
        name = base_name or "workflow_tool"
        if name not in self._tool_names:
            self._tool_names.add(name)
            return name
        suffix = 2
        while f"{name}_{suffix}" in self._tool_names:
            suffix += 1
        deduped = f"{name}_{suffix}"
        self._tool_names.add(deduped)
        return deduped

    def _derive_description(self, stem: str):
        readable = stem.replace("_", " ").replace("-", " ").strip()
        readable = readable if readable else stem
        return f"Execute the '{readable}' ComfyUI workflow."

    def _guess_output_preferences(self, workflow: Dict[str, Any]):
        for node in workflow.values():
            class_type = str(node.get("class_type", "")).lower()
            if "audio" in class_type:
                return AUDIO_OUTPUT_KEYS
        return DEFAULT_OUTPUT_KEYS

    def _coerce_value(self, value: Any, annotation: type):
        if annotation is str:
            return str(value)
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)
        if annotation is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)
        return value

# Global ComfyUI client (fallback since context isn’t available)
comfyui_client = ComfyUIClient(DEFAULT_COMFY_URL)
workflow_manager = WorkflowManager(WORKFLOW_DIR)

# Define application context (for future use)
class AppContext:
    def __init__(self, comfyui_client: ComfyUIClient):
        self.comfyui_client = comfyui_client

# Lifespan management (placeholder for future context support)
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    logger.info("Starting MCP server lifecycle...")
    try:
        # Startup: Could add ComfyUI health check here in the future
        logger.info("ComfyUI client initialized globally")
        yield AppContext(comfyui_client=comfyui_client)
    finally:
        # Shutdown: Cleanup (if needed)
        logger.info("Shutting down MCP server")

# Initialize FastMCP with lifespan
mcp = FastMCP(
    "ComfyUI_MCP_Server",
    lifespan=app_lifespan
)


def _save_file_from_url(url: str, filename: str, output_folder: Path) -> str:
    """Download file from URL and save to output folder. Returns the saved file path."""
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / filename
        
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
        
        # Write file and ensure it's fully flushed to disk
        with open(output_path, "wb") as f:
            f.write(response.content)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        logger.info(f"Saved file to: {output_path}")
        return str(output_path)
    except Exception as exc:
        logger.error(f"Failed to save file: {exc}")
        raise


@mcp.tool(name="list_client_roots", description="List the roots (resources) declared by the connected client.")
async def list_client_roots(ctx: Context):
    """Query the client for its declared roots and log them"""
    try:
        # Check if context has list_roots method
        if not hasattr(ctx, 'list_roots'):
            message = "The connected client does not support the roots capability. Roots are a client-side feature where clients declare directories/resources they want the server to be aware of."
            logger.warning(message)
            return CallToolResult(
                content=[{"type": "text", "text": message}],
                structuredContent={"supported": False}
            )
        
        # Use context's built-in list_roots() method
        roots = await ctx.list_roots()
        
        if not roots:
            message = "Client supports roots but has not declared any"
            logger.info(message)
            return CallToolResult(
                content=[{"type": "text", "text": message}],
                structuredContent={"roots": [], "supported": True}
            )
        
        # Log and format roots
        logger.info(f"Client roots ({len(roots)} total):")
        root_list = []
        for root in roots:
            uri = str(root.uri)
            name = root.name if hasattr(root, 'name') else None
            logger.info(f"  - {name}: {uri}" if name else f"  - {uri}")
            root_list.append({"uri": uri, "name": name} if name else {"uri": uri})
        
        message = f"Found {len(roots)} root(s):\n" + "\n".join(
            [f"• {r.get('name', 'Unnamed')}: {r['uri']}" if r.get('name') else f"• {r['uri']}" for r in root_list]
        )
        
        return CallToolResult(
            content=[{"type": "text", "text": message}],
            structuredContent={"roots": root_list, "count": len(roots), "supported": True}
        )
        
    except Exception as exc:
        # Handle "Method not found" specifically
        error_str = str(exc)
        if "Method not found" in error_str or "not found" in error_str.lower():
            message = "The connected client does not support the roots capability. This is normal - roots are an optional MCP feature that clients can choose to implement."
            logger.info(message)
            return CallToolResult(
                content=[{"type": "text", "text": message}],
                structuredContent={"supported": False, "error": error_str}
            )
        
        error_msg = f"Failed to query client roots: {exc}"
        logger.error(error_msg)
        return CallToolResult(
            content=[{"type": "text", "text": error_msg}],
            structuredContent={"error": error_str},
            isError=True
        )



def _register_workflow_tool(definition: WorkflowToolDefinition):
    def _tool_impl(*args, **kwargs):
        bound = _tool_impl.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            workflow = workflow_manager.render_workflow(definition, dict(bound.arguments), comfyui_client, OUTPUT_FOLDER)
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=definition.output_preferences,
            )
            asset = result.get("asset")
            if not asset:
                return {"error": "No asset returned by ComfyUI."}
            filename = asset.get("filename", "")
            asset_url = result.get("asset_url", "")
            mime_type = comfyui_client.guess_mime_type(filename)
            kind = "image" if mime_type.startswith("image/") else "audio"
            
            # Save file if output folder is configured
            saved_path = None
            if OUTPUT_FOLDER:
                try:
                    saved_path = _save_file_from_url(asset_url, filename, OUTPUT_FOLDER)
                except Exception as save_exc:
                    logger.warning(f"Failed to save file: {save_exc}")
            
            # Build response content
            content = []
            if saved_path:
                message = f"Workflow completed successfully.\nGenerated {kind}: {filename}\nSaved to: {saved_path}"
                content.append({"type": "text", "text": message})
                
                # Add image/audio data if file was saved locally
                # Add small delay to ensure file system has fully committed the write
                try:
                    time.sleep(0.1)  # 100ms delay to prevent race condition
                    with open(saved_path, "rb") as f:
                        file_data = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": kind,
                        "data": file_data,
                        "mimeType": mime_type,
                    })
                except Exception as read_exc:
                    logger.warning(f"Failed to read saved file for embedding: {read_exc}")
            else:
                markdown_hint = f"![Image]({asset_url})" if kind == "image" else f"[Audio]({asset_url})"
                message = f"Workflow completed successfully.\nGenerated {kind}: {filename}\nAsset URL: {asset_url}\n\nTo display, use: {markdown_hint}"
                content.append({"type": "text", "text": message})
            
            # Keep structuredContent minimal to avoid stack overflow
            structured = {
                "filename": filename,
                "mimeType": mime_type,
                "type": kind,
            }
            if saved_path:
                structured["savedPath"] = saved_path
            else:
                structured["assetUrl"] = asset_url
            
            return CallToolResult(
                content=content,
                structuredContent=structured
            )
        except Exception as exc:
            logger.exception("Workflow '%s' failed", definition.workflow_id)
            error_payload = {
                "error": str(exc),
                "workflow_id": definition.workflow_id,
                "tool": definition.tool_name,
            }
            return CallToolResult(content=[{"type": "text", "text": str(exc)}], structuredContent=error_payload, isError=True)

    parameters = []
    annotations: Dict[str, Any] = {}
    for param in definition.parameters.values():
        parameter = inspect.Parameter(
            name=param.name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=param.annotation,
        )
        parameters.append(parameter)
        annotations[param.name] = param.annotation
    annotations["return"] = dict
    _tool_impl.__signature__ = inspect.Signature(parameters, return_annotation=dict)
    _tool_impl.__annotations__ = annotations
    _tool_impl.__name__ = f"tool_{definition.tool_name}"
    _tool_impl.__doc__ = definition.description
    mcp.tool(name=definition.tool_name, description=definition.description)(_tool_impl)
    logger.info(
        "Registered MCP tool '%s' for workflow '%s'",
        definition.tool_name,
        definition.workflow_id,
    )


if workflow_manager.tool_definitions:
    for tool_definition in workflow_manager.tool_definitions:
        _register_workflow_tool(tool_definition)
else:
    logger.info(
        "No workflow placeholders found in %s; add %s markers to enable auto tools",
        WORKFLOW_DIR,
        PLACEHOLDER_PREFIX,
    )


def main():
    parser = argparse.ArgumentParser(description="ComfyUI MCP Server")
    parser.add_argument(
        "--comfy-url",
        default=DEFAULT_COMFY_URL,
        help="ComfyUI base URL (or set COMFY_URL).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for the MCP server (or set MCP_PORT).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Folder to save generated files. If not specified, files will not be saved locally.",
    )
    args = parser.parse_args()

    global comfyui_client, OUTPUT_FOLDER
    if args.comfy_url != comfyui_client.base_url:
        logger.info("Using ComfyUI URL: %s", args.comfy_url)
        comfyui_client = ComfyUIClient(args.comfy_url)

    if args.port != mcp.settings.port:
        logger.info("Using MCP server port: %s", args.port)
        mcp.settings.port = args.port
    
    if args.output_folder:
        OUTPUT_FOLDER = Path(args.output_folder)
        logger.info("Saving generated files to: %s", OUTPUT_FOLDER)

    mcp.run(transport="streamable-http")



if __name__ == "__main__":
    main()
