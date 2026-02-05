import requests
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

DEFAULT_MAPPING = {
    "prompt": ("6", "text"),
    "width": ("5", "width"),
    "height": ("5", "height"),
    "model": ("4", "ckpt_name")
}

class ComfyUIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        """Fetch list of available checkpoint models from ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple")
            if response.status_code != 200:
                logger.warning("Failed to fetch model list; using default handling")
                return []
            data = response.json()
            models = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.warning(f"Error fetching models: {e}")
            return []

    def generate_image(self, prompt, width, height, workflow_id="basic_api_test", model=None):
        try:
            workflow_file = f"workflows/{workflow_id}.json"
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            params = {"prompt": prompt, "width": width, "height": height}
            if model:
                # Validate or correct model name
                if model.endswith("'"):  # Strip accidental quote
                    model = model.rstrip("'")
                    logger.info(f"Corrected model name: {model}")
                if self.available_models and model not in self.available_models:
                    raise Exception(f"Model '{model}' not in available models: {self.available_models}")
                params["model"] = model

            for param_key, value in params.items():
                if param_key in DEFAULT_MAPPING:
                    node_id, input_key = DEFAULT_MAPPING[param_key]
                    if node_id not in workflow:
                        raise Exception(f"Node {node_id} not found in workflow {workflow_id}")
                    workflow[node_id]["inputs"][input_key] = value

            result = self.run_custom_workflow(
                workflow,
                preferred_output_keys=("images", "image", "gifs", "gif")
            )
            logger.info(f"Generated image URL: {result['asset_url']}")
            return result["asset_url"]

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")

    def run_custom_workflow(self, workflow: Dict[str, Any], preferred_output_keys: Sequence[str] | None = None, max_attempts: int = 1200):
        if preferred_output_keys is None:
            preferred_output_keys = ("images", "image", "gifs", "gif", "audio", "audios", "files")

        prompt_id = self._queue_workflow(workflow)
        entry = self._wait_for_prompt(prompt_id, max_attempts=max_attempts)
        outputs = entry.get("outputs", {}) if isinstance(entry, dict) else entry
        asset = self._extract_first_asset(entry, preferred_output_keys)
        asset_url = self._build_asset_url(asset)
        return {
            "asset_url": asset_url,
            "asset": asset,
            "prompt_id": prompt_id,
            "raw_outputs": entry
        }

    def _queue_workflow(self, workflow: Dict[str, Any]):
        logger.info("Submitting workflow to ComfyUI...")
        response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
        if response.status_code != 200:
            raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")
        prompt_id = response.json()["prompt_id"]
        logger.info(f"Queued workflow with prompt_id: {prompt_id}")
        return prompt_id

    def _wait_for_prompt(self, prompt_id: str, max_attempts: int = 1200):
        for attempt in range(max_attempts):
            response = requests.get(f"{self.base_url}/history/{prompt_id}")
            if response.status_code != 200:
                logger.warning("History endpoint returned %s on attempt %s", response.status_code, attempt + 1)
            else:
                history = response.json()
                if history.get(prompt_id):
                    entry = history[prompt_id]
                    outputs = entry.get("outputs", {}) if isinstance(entry, dict) else {}
                    status = entry.get("status", {}) if isinstance(entry, dict) else {}
                    if outputs:
                        logger.info("Workflow outputs: %s", json.dumps(outputs, indent=2))
                        return entry
                    if status:
                        logger.info("Workflow status: %s", json.dumps(status, indent=2))
                        status_flag = status.get("status") if isinstance(status, dict) else None
                        status_str = status.get("status_str") if isinstance(status, dict) else None
                        completed = status.get("completed") if isinstance(status, dict) else None
                        if completed is True or status_flag == "completed" or status_str == "completed":
                            logger.info("Workflow marked completed but outputs empty; returning entry.")
                            return entry
                    logger.info("Workflow outputs not ready yet; waiting...")
            time.sleep(1)
        raise Exception(f"Workflow {prompt_id} didn’t complete within {max_attempts} seconds")

    def _extract_first_asset(self, payload: Dict[str, Any], preferred_output_keys: Sequence[str]):
        outputs = payload.get("outputs", payload) if isinstance(payload, dict) else payload
        if not isinstance(outputs, dict):
            outputs = {}
        for node_output in outputs.values():
            for key in preferred_output_keys:
                assets = node_output.get(key)
                if assets:
                    return assets[0]
            ui_block = node_output.get("ui")
            if isinstance(ui_block, dict):
                for ui_value in ui_block.values():
                    if isinstance(ui_value, list) and ui_value:
                        first = ui_value[0]
                        if isinstance(first, dict) and "filename" in first:
                            return first
        # Fallback: scan for any list of dicts containing a filename
        for node_output in outputs.values():
            for value in node_output.values():
                if isinstance(value, list) and value:
                    first = value[0]
                    if isinstance(first, dict) and "filename" in first:
                        return first
        # Fallback: deep search for any dict with filename
        def find_asset(obj):
            if isinstance(obj, dict):
                if "filename" in obj:
                    return obj
                for child in obj.values():
                    found = find_asset(child)
                    if found:
                        return found
            elif isinstance(obj, list):
                for child in obj:
                    found = find_asset(child)
                    if found:
                        return found
            return None

        found_asset = find_asset(payload)
        if found_asset:
            return found_asset
        logger.warning(
            "No outputs matched preferred keys %s. Full outputs: %s",
            preferred_output_keys,
            json.dumps(payload, indent=2),
        )
        raise Exception(f"No outputs matched preferred keys: {preferred_output_keys}")

    def _build_asset_url(self, asset: Dict[str, Any]):
        filename = asset["filename"]
        subfolder = asset.get("subfolder", "")
        output_type = asset.get("type", "output")
        return f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={output_type}"

    def build_asset_url(self, filename: str, subfolder: str = "", output_type: str = "output"):
        return f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={output_type}"

    def upload_image(self, file_path: str, subfolder: str = "", overwrite: bool = False):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        data = {
            "subfolder": subfolder,
            "overwrite": "true" if overwrite else "false",
        }
        with path.open("rb") as handle:
            files = {"image": (path.name, handle, "application/octet-stream")}
            response = requests.post(f"{self.base_url}/upload/image", files=files, data=data)
        if response.status_code != 200:
            raise Exception(f"Image upload failed: {response.status_code} - {response.text}")
        
        # ComfyUI returns: {'name': 'filename.png', 'subfolder': 'images', 'type': 'input'}
        result = response.json()
        original_filename = path.name
        filename = result["name"]
        resolved_subfolder = result.get("subfolder", subfolder)
        output_type = result.get("type", "input")
        asset_url = self.build_asset_url(filename, resolved_subfolder, output_type)
        
        return {
            "asset_url": asset_url,
            "asset": {
                "filename": filename,
                "subfolder": resolved_subfolder,
                "type": output_type,
                "originalFileName": original_filename,
            },
            "raw_response": result
        }

    def guess_mime_type(self, filename: str):
        lowered = filename.lower()
        if lowered.endswith(".png"):
            return "image/png"
        if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
            return "image/jpeg"
        if lowered.endswith(".webp"):
            return "image/webp"
        if lowered.endswith(".gif"):
            return "image/gif"
        if lowered.endswith(".mp3"):
            return "audio/mpeg"
        if lowered.endswith(".wav"):
            return "audio/wav"
        if lowered.endswith(".ogg"):
            return "audio/ogg"
        return "application/octet-stream"