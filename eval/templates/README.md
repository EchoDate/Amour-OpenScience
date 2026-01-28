# Jinja Template Usage Guide

This directory contains Jinja templates for formatting conversation data.

## Template Files

### 1. `system_prompt.j2`
Used to render system prompts, including scenario, topic, and character information.

### 2. `conversation.j2`
Used to render complete conversations, including system prompts and all conversation turns.

### 3. `single_turn.j2`
Used to render single-turn conversations, including system prompts, conversation history, and current user message.

### 4. `chat_messages.j2`
Used to render messages in JSON format (less commonly used, Python functions are recommended).

## Usage

### Basic Usage

```python
from conversation import (
    render_system_prompt,
    render_conversation,
    render_single_turn,
    render_chat_messages
)
from pathlib import Path
from conversation import parse_conversation

# Load data
dataset = parse_conversation(Path("../data/annotated_real_scenes.json"))
conv = dataset.conversations[0]

# 1. Render system prompt
system_prompt = render_system_prompt(conv.system_setting)
print(system_prompt)

# 2. Render complete conversation
conversation_text = render_conversation(conv, max_history_turns=5)
print(conversation_text)

# 3. Render single turn (for generating new responses)
prompt = render_single_turn(
    conv,
    current_user_message="What do you think?",
    max_history_turns=3
)
print(prompt)

# 4. Render as messages format (for API calls)
messages = render_chat_messages(
    conv,
    max_history_turns=3,
    include_system=True,
    use_message_only=True
)
# messages can be used directly with OpenAI-compatible APIs
```

## Template Variables

### system_prompt.j2
- `scenario`: Scenario description
- `topic`: Topic
- `your_character_info`: Your character information (CharacterInfo object or dict)
- `other_character_info`: Other character information (CharacterInfo object or dict)

### conversation.j2
- `system_setting`: SystemSetting object
- `turns`: List of conversation turns (list of Turn objects)

### single_turn.j2
- `system_setting`: SystemSetting object
- `history_turns`: List of historical conversation turns
- `current_user_message`: Current user message

### chat_messages.j2
- `system_setting`: SystemSetting object
- `turns`: List of conversation turns
- `include_system`: Whether to include system message
- `use_message_only`: Whether to use only the message field (excluding state and internal_monologue)

## Custom Templates

If you need to customize templates:

1. Modify existing `.j2` files
2. Create new template files
3. Use the `get_template_env()` function to load custom templates:

```python
from conversation import get_template_env

env = get_template_env()
template = env.get_template('your_custom_template.j2')
result = template.render(your_variables=...)
```

## Dependencies

Install `jinja2`:

```bash
pip install jinja2
```
