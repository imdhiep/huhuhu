"""
Example: Custom Inference with Qwen3.6
This script shows how to use the VLM pipeline directly in Python.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor
from src.model_loader import ModelLoader
from src.inference import InferenceEngine
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize components
print("Loading model... (this may take a few minutes)")
model_loader = ModelLoader(
    model_name=config["model"]["model_name"],
    device=config["model"]["device"],
    torch_dtype=config["model"]["torch_dtype"],
    use_flash_attention=config["model"]["use_flash_attention"]
)
model_loader.load_model()

video_processor = VideoProcessor(
    target_fps=config["video"]["target_fps"],
    max_frames=config["video"]["max_frames"],
    resolution=config["video"]["resolution"]
)

inference = InferenceEngine(model_loader, video_processor, config)

# Run inference
video_path = "data/meeting.mp4"  # Thay bằng video của bạn
question = "Describe what people are doing in detail."

result = inference.run_inference(
    video_path=video_path,
    question=question,
    include_cot=True
)

print("\n" + "="*80)
print("RESULT:")
print("="*80)
print(f"Video: {result['video_path']}")
print(f"Frames: {result['num_frames']}")
print(f"\nReasoning:\n{result['reasoning_steps']}")
print(f"\nFinal Answer:\n{result['final_answer']}")
