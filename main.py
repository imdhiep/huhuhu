"""
Main Entry Point: VLM Video Reasoning with Qwen3-VL-2B-Instruct
Supports inference, dataset building, and batch processing.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor
from src.model_loader import ModelLoader
from src.inference import InferenceEngine
from src.dataset_builder import DatasetBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict) -> ModelLoader:
    """Initialize and load model."""
    model_config = config["model"]
    logger.info(f"Loading model: {model_config['model_name']}")

    model_loader = ModelLoader(
        model_name=model_config["model_name"],
        device=model_config.get("device", "cuda"),
        torch_dtype=model_config.get("torch_dtype", "bfloat16"),
        use_flash_attention=model_config.get("use_flash_attention", True),
        cache_dir="models/cache"
    )

    model_loader.load_model()
    return model_loader


def cmd_inference(args, config):
    """Run single video inference."""
    model_loader = setup_model(config)
    video_processor = VideoProcessor(
        target_fps=config["video"]["target_fps"],
        max_frames=config["video"]["max_frames"],
        resolution=config["video"]["resolution"],
        sampling_strategy=config["video"]["sampling_strategy"]
    )
    inference_engine = InferenceEngine(model_loader, video_processor, config)

    result = inference_engine.run_inference(
        video_path=args.video,
        question=args.question,
        include_cot=not args.no_cot
    )

    print("\n" + "="*80)
    print("INFERENCE RESULT")
    print("="*80)
    print(f"Video: {result['video_path']}")
    print(f"Question: {result['question']}")
    print(f"Frames processed: {result['num_frames']}")
    print("\n--- Reasoning ---")
    print(result.get("reasoning_steps", "N/A"))
    print("\n--- Final Answer ---")
    print(result["final_answer"])
    print("="*80)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to {output_path}")


def cmd_build_dataset(args, config):
    """Build dataset from videos in folder."""
    model_loader = setup_model(config)
    video_processor = VideoProcessor(
        target_fps=config["video"]["target_fps"],
        max_frames=config["video"]["max_frames"],
        resolution=config["video"]["resolution"],
        sampling_strategy=config["video"]["sampling_strategy"]
    )
    inference_engine = InferenceEngine(model_loader, video_processor, config)

    # Find all MP4 videos
    video_folder = Path(config["data"]["video_folder"])
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(video_folder.glob(f"*{ext}"))

    video_paths = [str(v) for v in video_paths]
    video_paths.sort()

    if not video_paths:
        logger.error(f"No videos found in {video_folder}")
        return

    logger.info(f"Found {len(video_paths)} videos")

    # Custom questions if provided
    question_templates = None
    if args.questions:
        question_templates = args.questions.split("|")

    builder = DatasetBuilder(
        output_path=config["data"]["output_json"],
        include_temporal=config.get("dataset_builder", {}).get("include_temporal", True)
    )

    dataset = builder.build_dataset(
        video_paths=video_paths,
        inference_engine=inference_engine,
        question_templates=question_templates,
        num_questions_per_video=args.num_questions
    )

    logger.info(f"Dataset built: {len(dataset)} entries")


def cmd_batch(args, config):
    """Run batch inference on multiple videos."""
    model_loader = setup_model(config)
    video_processor = VideoProcessor(
        target_fps=config["video"]["target_fps"],
        max_frames=config["video"]["max_frames"],
        resolution=config["video"]["resolution"],
        sampling_strategy=config["video"]["sampling_strategy"]
    )
    inference_engine = InferenceEngine(model_loader, video_processor, config)

    # Get video paths
    if args.video_folder:
        video_folder = Path(args.video_folder)
        video_paths = [str(v) for v in video_folder.glob("*.mp4")]
        video_paths.sort()
    elif args.video_list:
        with open(args.video_list, "r") as f:
            video_paths = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Must provide either --video-folder or --video-list")
        return

    results = inference_engine.run_batch_inference(
        video_paths=video_paths,
        question=args.question,
        output_file=args.output
    )

    logger.info(f"Processed {len(results)} videos")


def main():
    parser = argparse.ArgumentParser(
        description="VLM Video Reasoning with Qwen3-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video inference
  python main.py inference --video data/meeting.mp4 --question "Describe the actions"

  # Build dataset from all videos in data/
  python main.py build-dataset --num-questions 3

  # Batch inference
  python main.py batch --video-folder data/ --question "What are people doing?"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inference command
    inf_parser = subparsers.add_parser("inference", help="Run inference on single video")
    inf_parser.add_argument("--video", type=str, required=True, help="Path to MP4 video")
    inf_parser.add_argument("--question", type=str, required=True, help="Question about the video")
    inf_parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought")
    inf_parser.add_argument("--output", type=str, help="Save result to JSON file")

    # Build dataset command
    build_parser = subparsers.add_parser("build-dataset", help="Build dataset from videos")
    build_parser.add_argument("--num-questions", type=int, default=3, help="Questions per video")
    build_parser.add_argument("--questions", type=str, help="Custom questions separated by |")
    build_parser.add_argument("--output", type=str, help="Custom output path")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch inference on multiple videos")
    batch_parser.add_argument("--video-folder", type=str, help="Folder containing MP4 videos")
    batch_parser.add_argument("--video-list", type=str, help="Text file with video paths")
    batch_parser.add_argument("--question", type=str, required=True, help="Question for all videos")
    batch_parser.add_argument("--output", type=str, default="outputs/batch_results.json", help="Output JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load config
    config = load_config()

    # Override config with args
    if args.output and args.command in ["build-dataset", "batch"]:
        config["data"]["output_json"] = args.output

    # Execute command
    if args.command == "inference":
        cmd_inference(args, config)
    elif args.command == "build-dataset":
        cmd_build_dataset(args, config)
    elif args.command == "batch":
        cmd_batch(args, config)


if __name__ == "__main__":
    main()
