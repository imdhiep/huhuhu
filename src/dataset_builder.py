"""
Dataset Builder: Build VLM dataset in ShareGPT4Video format with reasoning.
Generates JSON dataset from videos with Qwen3.6 reasoning.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build dataset in ShareGPT4Video format with reasoning chains."""

    def __init__(
        self,
        output_path: str = "outputs/dataset.json",
        include_temporal: bool = True,
        include_bbox: bool = False
    ):
        """
        Initialize dataset builder.

        Args:
            output_path: Path to save JSON dataset
            include_temporal: Include timestamp info in video_uid
            include_bbox: Include bounding box annotations
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.include_temporal = include_temporal
        self.include_bbox = include_bbox

    def create_entry(
        self,
        video_path: str,
        conversations: List[Dict],
        reasoning_per_turn: List[Dict],
        source_split: str = "custom",
        source_line: int = 0
    ) -> Dict:
        """
        Create a single dataset entry.

        Args:
            video_path: Path to video file
            conversations: List of conversation turns [{"from": "human"/"gpt", "value": "text"}]
            reasoning_per_turn: List of reasoning [{"turn": n, "reasoning": "..."}]
            source_split: Data source split
            source_line: Line number in source

        Returns:
            Dataset entry dict
        """
        video_uid = Path(video_path).stem

        entry = {
            "id": source_line,
            "video": str(Path(video_path).absolute()),
            "video_uid": video_uid,
            "source_split": source_split,
            "source_line": source_line,
            "generated_conversations": conversations,
            "reasoning_per_turn": reasoning_per_turn
        }

        if self.include_temporal:
            entry["video_start_time"] = "00:00:00"
            entry["video_end_time"] = "00:00:00"

        return entry

    def build_dataset(
        self,
        video_paths: List[str],
        inference_engine,
        question_templates: Optional[List[str]] = None,
        num_questions_per_video: int = 3,
        save_interval: int = 10
    ) -> List[Dict]:
        """
        Build dataset from list of videos.

        Args:
            video_paths: List of video file paths
            inference_engine: InferenceEngine instance
            question_templates: Custom question templates
            num_questions_per_video: Number of Q&A pairs per video
            save_interval: Save every N videos

        Returns:
            List of dataset entries
        """
        if question_templates is None:
            question_templates = [
                "Describe what is happening in the video from start to end.",
                "What actions are being performed? Describe step by step.",
                "Analyze the interactions between people in the video.",
                "What is the main activity and who are the participants?",
                "Describe the sequence of events in chronological order."
            ]

        dataset = []

        for idx, video_path in enumerate(video_paths):
            logger.info(f"Processing video {idx+1}/{len(video_paths)}: {video_path}")

            try:
                # Generate conversations for this video
                entry = self._process_single_video(
                    video_path,
                    inference_engine,
                    question_templates[:num_questions_per_video],
                    idx
                )
                dataset.append(entry)

                # Periodic save
                if (idx + 1) % save_interval == 0:
                    self._save_partial(dataset)

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                # Add error entry
                error_entry = {
                    "id": idx,
                    "video": str(Path(video_path).absolute()),
                    "video_uid": Path(video_path).stem,
                    "error": str(e),
                    "generated_conversations": [],
                    "reasoning_per_turn": []
                }
                dataset.append(error_entry)

        # Final save
        self._save_final(dataset)
        logger.info(f"Dataset built with {len(dataset)} entries")

        return dataset

    def _process_single_video(
        self,
        video_path: str,
        inference_engine,
        questions: List[str],
        entry_id: int
    ) -> Dict:
        """Process one video and generate Q&A pairs."""
        conversations = []
        reasoning_per_turn = []

        for turn_idx, question in enumerate(questions):
            logger.info(f"  Turn {turn_idx+1}: {question[:50]}...")

            # Run inference
            result = inference_engine.run_inference(
                video_path=video_path,
                question=question,
                include_cot=True
            )

            # Add human turn
            conversations.append({
                "from": "human",
                "value": f"<video>\n{question}"
            })

            # Add GPT turn
            conversations.append({
                "from": "gpt",
                "value": result["final_answer"]
            })

            # Add reasoning
            reasoning_text = result.get("reasoning_steps", "") or result.get("full_response", "")
            reasoning_per_turn.append({
                "turn": turn_idx,
                "reasoning": reasoning_text
            })

        return self.create_entry(
            video_path=video_path,
            conversations=conversations,
            reasoning_per_turn=reasoning_per_turn,
            source_split="custom",
            source_line=entry_id
        )

    def _save_partial(self, dataset: List[Dict]):
        """Save partial dataset."""
        temp_path = self.output_path.with_suffix(".partial.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved partial dataset to {temp_path}")

    def _save_final(self, dataset: List[Dict]):
        """Save final dataset."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Dataset saved to {self.output_path}")

    def validate_entry(self, entry: Dict) -> bool:
        """
        Validate a dataset entry.

        Args:
            entry: Dataset entry dict

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["id", "video", "generated_conversations", "reasoning_per_turn"]
        for field in required_fields:
            if field not in entry:
                logger.error(f"Missing field: {field}")
                return False

        # Check conversations format
        for conv in entry["generated_conversations"]:
            if "from" not in conv or "value" not in conv:
                return False
            if conv["from"] not in ["human", "gpt"]:
                return False

        return True
