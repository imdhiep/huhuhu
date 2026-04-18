# VLM Video Reasoning with Qwen3-VL-2B-Instruct

Project để xây dựng Vision Language Model với **Qwen3-VL-2B-Instruct**, hỗ trợ video MP4 và chain-of-thought reasoning giống format ShareGPT4Video.

## 📋 Features

- ✅ **Qwen3-VL-2B-Instruct** support (2B params, efficient)
- ✅ Video processing từ MP4 (trích xuất frames)
- ✅ Chain-of-Thought reasoning với `<think>` tags
- ✅ Dataset builder format ShareGPT4Video
- ✅ Batch inference
- ✅ Hỗ trợ thinking mode & preserve_thinking
- ✅ Dễ mở rộng cho video dài (32K context)

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
# Tạo virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate  # PowerShell: .venv\Scripts\Activate.ps1

# Hoặc Linux/Mac
source .venv/bin/activate

# Cài đặt packages
pip install -r requirements.txt

# Nếu gặp lỗi với some packages:
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Qwen3-VL-2B-Instruct Model

Model sẽ tự động tải khi chạy lần đầu (khoảng 4GB):

```bash
# Tạo thư mục cache
mkdir models/cache
```

**Lưu ý:** Cần có GPU với ít nhất **8GB VRAM** cho inference (BF16).

### 3. Chuẩn bị Video

Đặt file video `.mp4` vào thư mục `data/`:

```
project/
├── data/
│   ├── meeting.mp4       ← video của bạn
│   └── crowd_scene.mp4
├── outputs/
├── models/cache/
└── main.py
```

### 4. Chạy Inference Cơ Bản

```bash
# Inference trên 1 video
python main.py inference \
    --video data/meeting.mp4 \
    --question "Describe what people are doing in this video step by step."

# Output sẽ được lưu vào outputs/ (nếu dùng --output)
```

**Ví dụ output:**
```json
{
  "video_path": "data/meeting.mp4",
  "question": "Describe what people are doing...",
  "num_frames": 32,
  "reasoning_steps": "Let's analyze the video...\n1. First, I see 4 people in a meeting room...",
  "final_answer": "The video shows a business meeting with 4 people..."
}
```

## 📖 Commands

### Inference trên 1 video

```bash
python main.py inference --video <path> --question "<your question>" [--no-cot] [--output result.json]
```

**Options:**
- `--no-cot`: Tắt chain-of-thought (direct answer)
- `--output`: Lưu kết quả vào file JSON

**Ví dụ:**
```bash
python main.py inference \
    --video data/people_talking.mp4 \
    --question "What action is being performed? Describe in detail." \
    --output outputs/result_meeting.json
```

### Build Dataset Từ Nhiều Video

Tự động tạo dataset format ShareGPT4Video từ tất cả video trong `data/`:

```bash
# Tạo 3 câu hỏi cho mỗi video
python main.py build-dataset --num-questions 3

# Tùy chỉnh câu hỏi
python main.py build-dataset \
    --num-questions 5 \
    --questions "Describe the scene.|What actions occur?|Analyze interactions." \
    --output outputs/my_dataset.json
```

**Output:** `outputs/dataset.json` với format:

```json
[
  {
    "id": 0,
    "video": "data/meeting.mp4",
    "video_uid": "meeting",
    "source_split": "custom",
    "source_line": 0,
    "generated_conversations": [
      {"from": "human", "value": "<video>\nDescribe what is happening..."},
      {"from": "gpt", "value": "The video shows..."}
    ],
    "reasoning_per_turn": [
      {"turn": 0, "reasoning": "Let's think step by step..."}
    ]
  }
]
```

### Batch Inference

Chạy cùng 1 câu hỏi trên nhiều video:

```bash
python main.py batch \
    --video-folder data/ \
    --question "How many people are in the video?" \
    --output outputs/batch_results.json
```

Hoặc dùng file list:

```bash
# videos.txt chứa:
# data/video1.mp4
# data/video2.mp4
# ...
python main.py batch --video-list videos.txt --question "Describe the main action"
```

## ⚙️ Configuration

Chỉnh sửa `config.yaml`:

```yaml
model:
  model_name: "Qwen/Qwen3-VL-2B-Instruct"  # Only this model is supported
  device: "cuda"  # "cpu" nếu không có GPU
  torch_dtype: "bfloat16"  # "float16" hoặc "auto"
  use_flash_attention: true  # Set false if OOM

  # Qwen3-VL settings
  max_new_tokens: 2048  # Max tokens to generate
  temperature: 0.7
  top_p: 0.9
  top_k: 20
  presence_penalty: 1.5
  enable_thinking: true   # Bật thinking mode (có <think> tags)
  preserve_thinking: false  # Giữ reasoning từ history

video:
  frame_stride: 10
  max_frames: 32  # Qwen3-VL xử lý tốt với 32 frames
  target_fps: 3
  resolution: 384  # Resize về 384x384
  sampling_strategy: "uniform"  # "uniform", "fps", "keyframe"

prompting:
  system_prompt: "You are a visual reasoning assistant..."
  # Có thể tuỳ chỉnh templates
```

### Memory Optimization

Nếu GPU < 8GB, dùng các tricks:

```yaml
model:
  device: "cuda"
  torch_dtype: "float16"  # Thay bfloat16 → float16 (tiết kiệm 50%)
  use_flash_attention: false  # Tắt nếu OOM
  max_new_tokens: 1024  # Giảm output length
```

Hoặc dùng ** quantization** (cần thêm code):

```bash
# Cài bitsandbytes
pip install bitsandbytes

# Sau đó load model với:
# load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
```

## 🧠 Qwen3-VL Special Features

### Thinking Mode

Qwen3-VL **mặc định bật thinking mode** - model sẽ suy nghĩ trong `<think>...</think>` trước khi trả lời:

```
<think>
Let me analyze the video step by step:
1. I see 4 people in a conference room...
2. They are standing near a whiteboard...
3. One person is pointing at the board...
</think>

Final answer: The video shows a team meeting...
```

Để tắt thinking mode (direct response):

```yaml
model:
  enable_thinking: false
```

Hoặc qua API:

```python
result = inference_engine.run_inference(
    video_path="...",
    question="...",
    enable_thinking=False  # Thêm parameter
)
```

### Preserve Thinking

Giữ lại reasoning từ các turn trước (hữu ích cho multi-turn chat):

```yaml
model:
  preserve_thinking: true
```

## 🎯 Sample Questions Cho Video Nhiều Người

```bash
# General
--question "Describe what is happening in the video."

# Action-focused
--question "What actions are being performed? List step by step."

# People interactions
--question "Describe all people in the video, their actions, and how they interact."

# Detailed reasoning
--question "Think step by step: First identify the scene, then describe each person's action, then analyze their relationships."

# Counting
--question "How many people are visible? Are they all facing the same direction?"

# Temporal
--question "Describe the sequence of events from start to end with timestamps."
```

## 📁 Project Structure

```
vlm/
├── data/                    # Đặt video .mp4 ở đây
│   └── meeting.mp4
├── src/
│   ├── video_processor.py   # Trích xuất frames từ video
│   ├── model_loader.py      # Load Qwen3-VL model
│   ├── inference.py         # Inference engine + CoT
│   └── dataset_builder.py  # Tạo dataset JSON
├── models/cache/            # Model weights (tự động tải)
├── outputs/                 # Kết quả inference & dataset
│   ├── dataset.json
│   └── batch_results.json
├── config.yaml              # Cấu hình
├── requirements.txt         # Dependencies
└── main.py                  # Entry point
```

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Giảm số frames
# config.yaml:
video:
  max_frames: 16  # Thay vì 32

# Hoặc giảm resolution
video:
  resolution: 256

# Hoặc dùng CPU (chậm)
model:
  device: "cpu"
  torch_dtype: "float16"
```

### Slow Inference

- **Dùng Flash Attention:** `use_flash_attention: true` (nếu GPU hỗ trợ)
- **Giảm max_new_tokens:** `max_new_tokens: 8192`
- **Dùng vLLM/SGLang serving** (xũng below)

### Model Không Tải Được

Đảm bảo `transformers` mới nhất:

```bash
pip install --upgrade transformers accelerate torch
```

Phiên bản tối thiểu:
- `transformers >= 4.48.0` (hỗ trợ Qwen3)
- `torch >= 2.2.0`

### Video Không Đọc Được

Đảm bảo có `opencv-python`:

```bash
pip install opencv-python
```

Nếu vẫn lỗi, convert video trước:

```bash
ffmpeg -i input.avi -c:v libx264 -preset fast output.mp4
```

## 🚀 Advanced: Serve Qwen3-VL với vLLM (Production)

Để throughput cao, dùng vLLM:

```bash
# Cài vLLM
pip install vllm

# Serve model
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --reasoning-parser qwen3

# Sau đó dùng API:
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
resp = client.chat.completions.create(
    model='Qwen/Qwen3-VL-2B-Instruct',
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'video_url', 'video_url': {'url': 'file://data/meeting.mp4'}},
            {'type': 'text', 'text': 'Describe the video'}
        ]
    }],
    max_tokens=2048,
    temperature=0.7,
    extra_body={'top_k': 20, 'mm_processor_kwargs': {'fps': 2}}
)
print(resp.choices[0].message.content)
"
```

## 📊 Benchmark & Performance

**Qwen3-VL-2B-Instruct:**
- Trained on 32K resolution images with video data
- Context: 32K tokens
- Efficient for 8GB GPUs

**Inference speed (approx):**
- 32 frames @ 384px: ~1-3s/turn trên RTX 3090
- Batch size 1 (video): ~5-15s tùy độ dài reasoning

## 🤝 Contributing

Đây là project mẫu. Bạn có thể mở rộng:

1. Thêm **video captioning** + **temporal localization**
2. Tích hợp **wandb** logging
3. Thêm **multi-turn conversation** với memory
4. **Quantization** (GPTQ/AWQ) cho GPU nhỏ
5. **Distributed inference** cho dataset lớn

## 📚 References

- [Qwen3-VL Docs](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)


## 📄 License

Apache 2.0 (same as Qwen3-VL)

---

**Made with ❤️ for VLM research. Happy coding!**
