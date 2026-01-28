# Profile Module

Profile system based on parameterization and dynamic evolution for better conversational interaction at the Agent level.

## Overview

This module implements two core functionalities:

1. **Profile Process Based on Parameterization and Dynamic Evolution**
   - Profile = (T_static, V_emotion, Relation)
   - T_static: Static traits (name, occupation, core values, etc., immutable)
   - V_emotion: Dynamic emotion vector (Big Five personality or emotional state, updated with interaction)
   - Relation: Social graph (specific views on other Agents)

2. **Social-Perspective Profiling**
   - Inner Profile (self layer): Who am I (personality, goals)
   - Relational Profile (relationship layer): Dynamic relationship matrix for each Agent

## Core Components

### 1. Profile (`profile.py`)

Core data structure, including:
- `StaticTraits`: Static traits
- `EmotionVector`: Emotion vector
- `Relation`: Relationship data

### 2. State Tracker (`state_tracker.py`)

State tracker that updates emotion vector based on conversational interactions:
- Input: Previous round of dialogue + current environmental event
- Process: Calculate how current event changes V_emotion
- Output: Updated Profile for guiding Response generation

### 3. Social Profiler (`social_profiling.py`)

Social perspective analyzer, managing dual-layer Profile:
- Inner Profile: Self-awareness
- Relational Profile: Relationship with each Agent (Intimacy, Trust, Dominance)

### 4. Memory Enhancer (`memory_enhancer.py`)

Enhances Memory Retrieval, considering when retrieving memory:
- Current state of Profile
- Relationship with current conversation agent
- Impact of emotional state on memory selection

### 5. Prompt Enhancer (`prompt_enhancer.py`)

Enhances Prompt Construction, adding to prompt:
- Profile information (static traits + current emotional state)
- Relationship constraints (if responding to a specific agent)
- Response style guidance

### 6. ProfileAgent (`profile_agent.py`)

Agent class integrated with Profile, based on HiAgent design:
- Uses MemoryEnhancer in Memory Retrieval stage
- Uses PromptEnhancer in Prompt Construction stage
- Automatically updates emotional state and relationships
- Supports relationship-aware dialogue generation

## Quick Start

### Basic Usage

```python
from profile import Profile, StaticTraits, EmotionVector

# Create static traits
static_traits = StaticTraits(
    name="Alice",
    occupation="Software Engineer",
    age=28,
    mbti="INTJ",
    personality="Analytical and independent"
)

# Create emotion vector
emotion_vector = EmotionVector(
    openness=0.6,
    extraversion=0.3,
    neuroticism=0.4
)

# Create Profile
profile = Profile(
    static_traits=static_traits,
    emotion_vector=emotion_vector
)
```

### Using State Tracker to Update Emotions

```python
from profile import StateTracker

tracker = StateTracker(profile)

# Update emotions based on dialogue and events
tracker.update_emotion(
    last_dialogue="You did a terrible job!",
    current_event="Received criticism",
    interaction_type="criticism"
)

# Get response guidance
guidance = tracker.get_emotion_guidance()
```

### Using Social Profiler to Manage Relationships

```python
from profile import SocialProfiler

profiler = SocialProfiler(profile)

# Update relationship with Agent B
profiler.update_relation(
    agent_id="Agent_B",
    interaction="Thank you for your help!",
    interaction_type="friendly"
)

# Get relationship constraint prompt
constraint = profiler.get_relation_constraint_prompt("Agent_B")
```

### Enhancing Memory Retrieval

```python
from profile import MemoryEnhancer

enhancer = MemoryEnhancer(profile)

# Retrieve memory based on Profile
enhanced_memory = enhancer.retrieve_with_profile(
    memory=memory_list,
    current_agent_id="Agent_B",
    max_items=10
)
```

### Enhancing Prompt Construction

```python
from profile import PromptEnhancer

enhancer = PromptEnhancer(profile)

# Enhance prompt
enhanced_prompt = enhancer.enhance_prompt(
    base_prompt="Please respond to the message",
    target_agent_id="Agent_B",
    include_profile=True,
    include_emotion=True,
    include_relation=True
)
```

## Using ProfileAgent

### Basic Usage

```python
from profile import ProfileAgent, Profile, StaticTraits, EmotionVector

# 1. Create Profile
static_traits = StaticTraits(
    name="Alice",
    occupation="Software Engineer",
    mbti="INTJ",
    personality="Analytical and independent"
)
emotion_vector = EmotionVector()
profile = Profile(static_traits=static_traits, emotion_vector=emotion_vector)

# 2. Create LLM model (needs to implement generate method)
class MyLLM:
    def generate(self, system_message, prompt):
        # Call LLM API
        return True, "response"

llm = MyLLM()

# 3. Create ProfileAgent
agent = ProfileAgent(
    llm_model=llm,
    profile=profile,
    memory_size=100,
    instruction="You are a helpful assistant.",
    system_message="You are Alice, a software engineer."
)

# 4. Reset Agent
agent.reset(
    goal="Have a conversation",
    init_obs="Agent B says: Hello, how are you?"
)

# 5. Run Agent to generate response
success, action = agent.run()
print(f"Response: {action}")

# 6. Update state (will automatically update Profile)
agent.update(action, "Agent B says: That's great!")

# 7. Run again (will consider updated Profile)
success, action = agent.run(target_agent_id="Agent_B")
```

### Relationship-Aware Conversation

```python
# ProfileAgent automatically detects interaction types and updates relationships
agent.reset(goal="Build relationship", init_obs="Agent C says: I don't trust you.")

# First interaction
success, action = agent.run(target_agent_id="Agent_C")

# Update (negative interaction detected)
agent.update(action, "Agent C says: You're wrong!")

# View relationship
relation = agent.social_profiler.get_relational_profile("Agent_C")
print(f"Trust: {relation['Trust']:.2f}")  # Trust will decrease

# Second interaction (will consider updated relationship)
success, action = agent.run(target_agent_id="Agent_C")
```

### Emotion State Tracking

```python
# ProfileAgent automatically tracks emotional state
agent.reset(goal="Conversation", init_obs="Agent D says: You did a terrible job!")

# Run (will trigger emotion update)
success, action = agent.run()

# Update state (criticism detected)
agent.update(action, "Agent D says: This is unacceptable!")

# View emotional state
emotion = agent.get_emotion_state()
print(f"Neuroticism: {emotion['neuroticism']:.2f}")  # Neuroticism will increase
print(f"Stress: {emotion['stress']:.2f}")  # Stress will increase

# Get emotion guidance
guidance = agent.state_tracker.get_emotion_guidance()
print(guidance)  # "Current emotion is quite sensitive, responses may be more defensive"
```

### Creating from Configuration

```python
from profile import ProfileAgent

config = {
    "memory_size": 100,
    "instruction": "You are a helpful assistant.",
    "examples": ["Example 1", "Example 2"],
    "system_message": "You are a friendly person.",
    "need_goal": False,
    "use_parser": True
}

agent = ProfileAgent.from_config(llm, profile, config)
```

## Integration with Annotator

Create Profile object from Annotator-generated Profile data:

```python
from profile import create_profile_from_annotator_data

# Profile data generated by Annotator
annotator_data = {
    "gender": "Female",
    "age": 25,
    "interests": "reading, music, travel",
    "job": "Graphic Designer",
    "mbti": "ENFP",
    "personality": "Creative and enthusiastic",
    "style": "Expressive and friendly"
}

# Create Profile
profile = create_profile_from_annotator_data(annotator_data)
```

## Data Structure

### Profile Structure

```python
Profile {
    static_traits: StaticTraits {
        name: str
        occupation: str
        age: int
        gender: str
        interests: List[str]
        mbti: str
        personality: str
        talking_style: str
        core_values: List[str]
    }
    
    emotion_vector: EmotionVector {
        openness: float (0-1)
        conscientiousness: float (0-1)
        extraversion: float (0-1)
        agreeableness: float (0-1)
        neuroticism: float (0-1)
        trust: float (0-1)
        stress: float (0-1)
        energy: float (0-1)
    }
    
    relations: Dict[str, Relation] {
        agent_id: Relation {
            intimacy: float (-1 to 1)
            trust: float (0 to 1)
            dominance: float (-1 to 1)
        }
    }
}
```

## Serialization

Profile supports JSON serialization:

```python
# Serialize
json_str = profile.to_json()

# Deserialize
profile = Profile.from_json(json_str)
```

## Examples

For complete examples, please refer to `example_usage.py`.

## ACL Highlights

1. **Simulates Not Only "What People Say" but Also "How People's Hearts Change"**
   - Dynamically updates emotion vector through State Tracker
   - Emotional state influences response style

2. **Relationship-Aware Dialogue Generation**
   - Considers relationship with each Agent through Relational Profile
   - Adds relationship constraints to prompts

3. **Multi-level Profile Modeling**
   - Static traits (immutable)
   - Dynamic emotions (updatable)
   - Social relationships (structured)
