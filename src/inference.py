"""
Inference Engine: Run VLM inference on videos with Chain-of-Thought prompting.
Supports Qwen3-VL-2B-Instruct model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

import numpy as np
from PIL import Image

from src.video_processor import VideoProcessor
from src.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Run VLM inference on videos with reasoning."""

    def __init__(
        self,
        model_loader: ModelLoader,
        video_processor: VideoProcessor,
        config: Dict
    ):
        self.model_loader = model_loader
        self.video_processor = video_processor
        self.config = config

        # Prompt templates
        self.prompts = config.get("prompting", {}).get("question_templates", {})
        self.cot_instruction = config.get("prompting", {}).get("cot_instruction", "")

    def run_inference(
        self,
        video_path: Union[str, Path],
        question: str,
        include_cot: bool = True
    ) -> Dict:
        """
        Run inference on a video with a question.

        Args:
            video_path: Path to MP4 video
            question: Text query about the video
            include_cot: Whether to include chain-of-thought in response

        Returns:
            Dictionary with reasoning steps and final answer
        """
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Question: {question}")

        # Step 1: Extract frames
        frames = self.video_processor.extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video")

        # Step 2: Convert frames to PIL Images
        pil_images = [Image.fromarray(frame) for frame in frames]

        # Step 3: Build messages with CoT
        messages = self._build_messages(pil_images, question, include_cot)

        # Step 4: Generate response
        logger.info("Running model inference...")
        response = self.model_loader.generate(
            messages=messages,
            max_new_tokens=self.config["model"]["max_new_tokens"],
            temperature=self.config["model"]["temperature"],
            top_p=self.config["model"]["top_p"]
        )

        # Step 5: Parse response
        result = self._parse_response(response, include_cot)

        # Add metadata
        result["video_path"] = str(video_path)
        result["question"] = question
        result["num_frames"] = len(frames)
        result["timestamp"] = datetime.now().isoformat()

        return result

    def _build_messages(
        self,
        images: List[Image.Image],
        question: str,
        include_cot: bool
    ) -> List[Dict]:
        """
        Build conversation messages for the model.

        Qwen3-VL expects video input as a list of frames.
        """
        system_prompt = self.config.get("prompting", {}).get("system_prompt", "")

        # Build content with images
        content = []

        # Add all frames as images
        for img in images:
            content.append({"type": "image", "image": img})

        # Add question with CoT instruction if enabled
        if include_cot:
            full_question = f"{question}\n\n{self.cot_instruction}"
        else:
            full_question = question

        content.append({"type": "text", "text": full_question})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        return messages

    def _parse_response(self, response: str, include_cot: bool) -> Dict:
        """
        Parse model response to extract reasoning steps and final answer.
        Qwen3-VL outputs: <think>reasoning...</think> followed by final answer.
        """
        result = {
            "full_response": response,
            "reasoning_steps": "",
            "final_answer": ""
        }

        if include_cot:
            # Extract <think>...</think> block
            import re
            think_pattern = r'<think>(.*?)</think>'
            think_match = re.search(think_pattern, response, re.DOTALL | re.IGNORECASE)

            if think_match:
                reasoning_text = think_match.group(1).strip()
                result["reasoning_steps"] = reasoning_text

                # Final answer = everything after </think>
                final_answer = response[think_match.end():].strip()
                result["final_answer"] = final_answer
            else:
                # Fallback: try to split by common markers
                lines = response.split("\n")
                reasoning_lines = []
                final_answer_lines = []
                found_final = False

                for line in lines:
                    line_lower = line.strip().lower()
                    if any(marker in line_lower for marker in ["final answer:", "answer:", "in conclusion", "to summarize", "conclusion:"]):
                        found_final = True
                        final_answer_lines.append(line)
                    elif found_final:
                        final_answer_lines.append(line)
                    else:
                        reasoning_lines.append(line)

                result["reasoning_steps"] = "\n".join(reasoning_lines).strip()
                result["final_answer"] = "\n".join(final_answer_lines).strip() or response

                # If no clear split, treat whole as reasoning
                if not result["final_answer"] and not result["reasoning_steps"]:
                    result["reasoning_steps"] = response
        else:
            result["final_answer"] = response

        return result

    def run_batch_inference(
        self,
        video_paths: List[Union[str, Path]],
        question: str,
        output_file: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """
        Run inference on multiple videos.

        Args:
            video_paths: List of video file paths
            question: Same question for all videos
            output_file: Optional path to save results as JSON

        Returns:
            List of results
        """
        results = []

        for video_path in video_paths:
            try:
                result = self.run_inference(video_path, question)
                results.append(result)
                logger.info(f"Completed: {video_path}")
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    "video_path": str(video_path),
                    "error": str(e),
                    "question": question
                })

        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")

        return results

    def generate_dataset(
        self,
        video_paths: List[Union[str, Path]],
        question_templates: Optional[List[str]] = None,
        num_questions_per_video: int = 3
    ) -> List[Dict]:
        """
        Generate a dataset with multiple Q&A pairs per video.
        Format similar to ShareGPT4Video.

        Args:
            video_paths: List of video file paths
            question_templates: List of question templates to use
            num_questions_per_video: Number of questions to generate per video

        Returns:
            List of dataset entries in ShareGPT4Video format
        """
        if question_templates is None:
            question_templates = [
                self.prompts.get("general", "Describe what is happening in the video."),
                self.prompts.get("action", "What action is being performed?"),
                self.prompts.get("detailed", "Describe the action in detail, step by step.")
            ]

        dataset = []

        for video_path in video_paths:
            video_uid = Path(video_path).stem
            entry = {
                "id": len(dataset),
                "video": str(video_path),
                "video_uid": video_uid,
                "source_split": "custom",
                "source_line": len(dataset),
                "generated_conversations": [],
                "reasoning_per_turn": []
            }

            # Extract frames once
            frames = self.video_processor.extract_frames(video_path)
            pil_images = [Image.fromarray(frame) for frame in frames]

            # Generate multiple Q&A turns
            for turn_idx in range(num_questions_per_video):
                question = question_templates[turn_idx % len(question_templates)]

                # Build messages
                messages = self._build_messages(pil_images, question, include_cot=True)

                # Generate
                response = self.model_loader.generate(
                    messages=messages,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=0.9
                )

                # Parse
                parsed = self._parse_response(response, include_cot=True)

                # Add conversation turn
                entry["generated_conversations"].append({
                    "from": "human",
                    "value": f"<video>\n{question}"
                })
                entry["generated_conversations"].append({
                    "from": "gpt",
                    "value": parsed["final_answer"]
                })

                # Add reasoning for this turn
                entry["reasoning_per_turn"].append({
                    "turn": turn_idx,
                    "reasoning": parsed["reasoning_steps"] or response
                })

            dataset.append(entry)
            logger.info(f"Generated dataset entry for {video_path}")

        return dataset
