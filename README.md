# Amour-OpenScience

An open-source project for training socially-aware conversational agents using the **CoST (Chain-of-Social-Thought)** framework. This project implements a comprehensive system for generating, annotating, and training dialogue agents with dynamic personality profiles and social cognition capabilities.

## 🌟 Overview

Amour-OpenScience provides tools for:
- **Data Generation & Annotation**: Generate conversational data with CoST reasoning
- **Profile Management**: Dynamic personality profiles with emotional state tracking
- **Model Training**: Fine-tune LLMs for socially-aware conversations
- **Data Conversion**: Convert data to various training formats (ShareGPT, etc.)

## 📁 Project Structure

```
Amour-OpenScience/
├── annotator/              # Data annotation and generation
│   └── relabel_with_ark.py    # CoST relabeling with DeepSeek-V3
├── profile/                # Profile and social cognition modules
│   ├── profile.py             # Core profile data structures
│   ├── state_tracker.py       # Emotional state tracking
│   ├── social_profiling.py    # Social relationship management
│   ├── memory_enhancer.py     # Memory retrieval with profile context
│   ├── prompt_enhancer.py     # Profile-aware prompt construction
│   └── agent_example.py       # Example agent implementation
├── train.py                # Main training script
├── download_model.py       # Model download utility
├── convert_data_for_llamafactory_sharegpt.py  # Data format conversion
├── main.py                 # Cloud Function entry point
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Amour-OpenScience.git
cd Amour-OpenScience

# Install dependencies
pip install -r requirements.txt

# For training dependencies
pip install -r requirements_train.txt
```

### 2. Download Models

```bash
# Download Llama model from HuggingFace
python download_model.py \
    --model-name meta-llama/Meta-Llama-3.1-8B \
    --local-dir ./models/Meta-Llama-3.1-8B
```

### 3. Generate/Annotate Data

```bash
# Relabel conversations with CoST reasoning using DeepSeek-V3
python annotator/relabel_with_ark.py \
    --input data/raw_conversations.jsonl \
    --output data/cost_annotated.jsonl \
    --api-key YOUR_ARK_API_KEY
```

### 4. Convert Data Format

```bash
# Convert to ShareGPT format for Llama-Factory
python convert_data_for_llamafactory_sharegpt.py \
    --input data/cost_annotated.jsonl \
    --output data/sharegpt_format.json
```

### 5. Train Model

```bash
# Train with default config
python train.py --config train_config_example.json

# Or use Llama-Factory configs
llamafactory-cli train llamafactory_config_h80_lora.yaml
```

## 📚 Core Components

### 1. CoST Relabeling (`annotator/relabel_with_ark.py`)

Generates CoST-format conversational data with:
- **Conversation Length Standardization**: Maintains 3-12 turns (prefers 6-8)
- **First-Person Monologue Generation**: Internal reasoning using DeepSeek-V3
- **Archetype Initialization**: Maps personality to Big-5 traits
- **State Updates**: Tracks emotional and relational states over conversation
- **Multi-worker Processing**: Parallel processing for efficiency

**Key Features:**
```python
# Supports three inference modes:
# 1. Online FaaS (real-time)
# 2. Batch FaaS (cost-efficient)
# 3. Local API (self-hosted)

# State representation includes:
- Emotion vectors (joy, sadness, anger, fear, surprise, disgust)
- Relationship dimensions (intimacy, trust, dominance)
- Dynamic state updates with EMA smoothing
```

**Usage:**
```bash
python annotator/relabel_with_ark.py \
    --input input.jsonl \
    --output output.jsonl \
    --api-key YOUR_KEY \
    --model-name ep-xxxxx \
    --workers 4 \
    --target-turns 8
```

### 2. Profile System (`profile/`)

A comprehensive personality and social cognition framework:

#### Core Components:

- **`profile.py`**: Base profile structure
  - `StaticTraits`: Immutable characteristics (name, occupation, values)
  - `EmotionVector`: Dynamic emotional state
  - `Relation`: Social relationship tracking

- **`state_tracker.py`**: Emotional state updates
  - Processes dialogue context to update emotions
  - Implements EMA (Exponential Moving Average) for smooth transitions
  - Tracks multi-dimensional emotional states

- **`social_profiling.py`**: Social perspective modeling
  - **Inner Profile**: Self-perception (personality, goals)
  - **Relational Profile**: Dynamic relationship matrix per agent
  - Three relationship dimensions: Intimacy, Trust, Dominance

- **`memory_enhancer.py`**: Profile-aware memory retrieval
  - Considers current emotional state
  - Weights memories by relationship context
  - Prioritizes relevant experiences

- **`prompt_enhancer.py`**: Profile-injected prompts
  - Embeds profile info into system prompts
  - Provides relationship context
  - Guides response generation with personality consistency

**Example Usage:**
```python
from profile import Profile, StaticTraits, EmotionVector
from state_tracker import StateTracker
from social_profiling import SocialProfiler

# Create agent profile
profile = Profile(
    static_traits=StaticTraits(name="Alice", occupation="Teacher"),
    emotion_vector=EmotionVector(joy=0.6, trust=0.7)
)

# Track state changes
tracker = StateTracker(profile)
updated_profile = tracker.update_from_dialogue(dialogue_history)

# Manage social relationships
profiler = SocialProfiler(profile)
profiler.update_relationship("Bob", intimacy_delta=0.1, trust_delta=0.05)
```

### 3. Training Pipeline (`train.py`)

Full-featured training script for fine-tuning LLMs:

**Features:**
- Hugging Face Transformers integration
- DeepSpeed support for distributed training
- Automatic evaluation and checkpointing
- WandB logging integration
- Custom CoST data collator
- Mixed precision training (FP16/BF16)

**Configuration:**
```python
# Key training parameters:
- model_name: Base model (e.g., "meta-llama/Meta-Llama-3.1-8B")
- max_seq_length: 8192 tokens (adjustable)
- batch_size: Per-device batch size
- learning_rate: 2e-5 (default)
- num_epochs: 3
- eval_strategy: "steps" or "epoch"
```

**Usage:**
```bash
# Basic training
python train.py \
    --model-name meta-llama/Meta-Llama-3.1-8B \
    --train-data data/train.jsonl \
    --val-data data/val.jsonl \
    --output-dir ./checkpoints

# With custom config
python train.py --config my_config.json
```

### 4. Data Conversion (`convert_data_for_llamafactory_sharegpt.py`)

Converts Amour CoST format to Llama-Factory compatible ShareGPT format:

**Input Format (CoST):**
```json
{
  "scenario": {...},
  "agent_a": {...},
  "agent_b": {...},
  "conversation": [
    {
      "speaker": "A",
      "message": "...",
      "CoST_thought": "...",
      "emotion_after": {...},
      "relation_after": {...}
    }
  ]
}
```

**Output Format (ShareGPT):**
```json
{
  "conversations": [
    {"from": "system", "value": "scenario + role info"},
    {"from": "human", "value": "user message"},
    {"from": "observation", "value": "contextual info"},
    {"from": "gpt", "value": "[Internal Monologue]\n...\n\n```json\n{state}\n```\n\n[Response]\n..."}
  ]
}
```

**Usage:**
```bash
python convert_data_for_llamafactory_sharegpt.py \
    --input train.jsonl \
    --output sharegpt_train.json
```

### 5. Model Download (`download_model.py`)

Utility for downloading models from HuggingFace Hub:

**Features:**
- Downloads both tokenizer and model weights
- Supports safetensors format (recommended)
- Saves to local directory for offline training
- FP16 weights to save disk space

**Usage:**
```bash
# Download Llama model
python download_model.py \
    --model-name meta-llama/Meta-Llama-3.1-8B \
    --local-dir ./models/llama-3.1-8b

# Then use in training
python train.py --model-name ./models/llama-3.1-8b
```

## 🔧 Configuration Files

### Training Configs

- **`train_config_example.json`**: Example training configuration
- **`llamafactory_config_h80.yaml`**: Full fine-tuning config for H80 GPUs
- **`llamafactory_config_h80_lora.yaml`**: LoRA fine-tuning config

### LLaMA-Factory Integration

This project is compatible with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
# Add to LLaMA-Factory's dataset_info.json:
{
  "amour_cost": {
    "file_name": "path/to/sharegpt_format.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}

# Then train with LLaMA-Factory
llamafactory-cli train llamafactory_config_h80_lora.yaml
```

## 📊 Data Format

### CoST Format

The CoST (Chain-of-Social-Thought) format includes:

1. **Scenario**: Conversation context and topic
2. **Agent Profiles**: Static traits and initial states
3. **Conversation Turns**:
   - Speaker ID (A or B)
   - Message content
   - **CoST Thought**: Internal monologue (reasoning process)
   - **Emotion After**: Post-turn emotional state
   - **Relation After**: Updated relationship state
   - Observation (optional contextual info)

**State Representation:**
```python
{
  "emotion": {
    "joy": 0.65,
    "sadness": 0.1,
    "anger": 0.05,
    "fear": 0.0,
    "surprise": 0.2,
    "disgust": 0.0
  },
  "relation": {
    "intimacy": 0.7,  # Closeness level
    "trust": 0.8,     # Trust level
    "dominance": 0.5  # Relative power in relationship
  }
}
```

## 🛠️ Advanced Usage

### Custom Profile Agents

```python
from profile.agent_example import ProfileAgent

# Create custom agent
agent = ProfileAgent(
    profile=my_profile,
    use_state_tracking=True,
    use_memory_enhancement=True
)

# Generate response
response = agent.generate_response(
    user_input="Hello, how are you?",
    conversation_history=[...]
)
```

### Batch Processing

```python
# Process multiple conversations in parallel
from annotator.relabel_with_ark import CoSTRelabeler

relabeler = CoSTRelabeler(
    api_key="YOUR_KEY",
    num_workers=8
)

results = relabeler.process_batch(
    input_conversations,
    target_turns=8
)
```

### Custom State Updates

```python
from profile.state_tracker import StateTracker

# Define custom update rules
tracker = StateTracker(
    profile,
    ema_alpha=0.3,  # Smoothing factor
    coupling_weight=0.2  # Emotion-relation coupling
)

# Update with custom event
new_profile = tracker.update_from_event(
    event_type="conflict",
    intensity=0.7,
    target_agent="Bob"
)
```

## 📖 Examples

See the `profile/example_usage.py` and `profile/agent_example.py` for complete examples of:
- Building profile-aware agents
- Managing multi-agent conversations
- Tracking state evolution
- Memory-enhanced dialogue

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is open-sourced under the MIT License.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{amour_openscience,
  title={Amour-OpenScience: CoST Framework for Socially-Aware Dialogue Agents},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Amour-OpenScience}
}
```

## 🔗 Related Resources

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Efficient LLM fine-tuning framework
- [DeepSeek](https://www.deepseek.com/): Advanced language models
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/): Pre-trained model library

## 📧 Contact

For questions and discussions, please open an issue on GitHub.

---

**Happy Training! 🚀**
