# Profile 模块

基于参数化与动态演化的 Profile 系统，用于在 Agent 层面进行更好的对话交流。

## 概述

本模块实现了两个核心功能：

1. **基于参数化与动态演化的 Profile 过程**
   - Profile = (T_static, V_emotion, Relation)
   - T_static: 静态特质（姓名、职业、核心价值观等，不可变）
   - V_emotion: 动态情绪向量（大五人格或情感状态，随交互更新）
   - Relation: 社交图谱（对其他 Agent 的特定看法）

2. **基于"社会视角"的 Profile (Social-Perspective Profiling)**
   - Inner Profile (自我层): 我是谁（性格、目标）
   - Relational Profile (关系层): 对每个 Agent 的动态关系矩阵

## 核心组件

### 1. Profile (`profile.py`)

核心数据结构，包含：
- `StaticTraits`: 静态特质
- `EmotionVector`: 情绪向量
- `Relation`: 关系数据

### 2. State Tracker (`state_tracker.py`)

状态追踪器，根据对话交互更新情绪向量：
- 输入：上一轮对话 + 当前环境事件
- 处理：计算当前事件如何改变 V_emotion
- 输出：更新后的 Profile 用于指导 Response 生成

### 3. Social Profiler (`social_profiling.py`)

社会视角分析器，管理双层 Profile：
- Inner Profile: 自我认知
- Relational Profile: 对每个 Agent 的关系（Intimacy, Trust, Dominance）

### 4. Memory Enhancer (`memory_enhancer.py`)

增强 Memory Retrieval，在检索记忆时考虑：
- Profile 的当前状态
- 与当前对话 agent 的关系
- 情绪状态对记忆选择的影响

### 5. Prompt Enhancer (`prompt_enhancer.py`)

增强 Prompt Construction，在 prompt 中加入：
- Profile 信息（静态特质 + 当前情绪状态）
- 关系约束（如果是对特定 agent 的回复）
- 响应风格指导

### 6. ProfileAgent (`profile_agent.py`)

集成 Profile 的 Agent 类，参考 HiAgent 的设计：
- 在 Memory Retrieval 环节使用 MemoryEnhancer
- 在 Prompt Construction 环节使用 PromptEnhancer
- 自动更新情绪状态和关系
- 支持关系感知的对话生成

## 快速开始

### 基本使用

```python
from profile import Profile, StaticTraits, EmotionVector

# 创建静态特质
static_traits = StaticTraits(
    name="Alice",
    occupation="Software Engineer",
    age=28,
    mbti="INTJ",
    personality="Analytical and independent"
)

# 创建情绪向量
emotion_vector = EmotionVector(
    openness=0.6,
    extraversion=0.3,
    neuroticism=0.4
)

# 创建 Profile
profile = Profile(
    static_traits=static_traits,
    emotion_vector=emotion_vector
)
```

### 使用 State Tracker 更新情绪

```python
from profile import StateTracker

tracker = StateTracker(profile)

# 根据对话和事件更新情绪
tracker.update_emotion(
    last_dialogue="You did a terrible job!",
    current_event="Received criticism",
    interaction_type="criticism"
)

# 获取响应指导
guidance = tracker.get_emotion_guidance()
```

### 使用 Social Profiler 管理关系

```python
from profile import SocialProfiler

profiler = SocialProfiler(profile)

# 更新与 Agent B 的关系
profiler.update_relation(
    agent_id="Agent_B",
    interaction="Thank you for your help!",
    interaction_type="friendly"
)

# 获取关系约束 prompt
constraint = profiler.get_relation_constraint_prompt("Agent_B")
```

### 增强 Memory Retrieval

```python
from profile import MemoryEnhancer

enhancer = MemoryEnhancer(profile)

# 基于 Profile 检索记忆
enhanced_memory = enhancer.retrieve_with_profile(
    memory=memory_list,
    current_agent_id="Agent_B",
    max_items=10
)
```

### 增强 Prompt Construction

```python
from profile import PromptEnhancer

enhancer = PromptEnhancer(profile)

# 增强 prompt
enhanced_prompt = enhancer.enhance_prompt(
    base_prompt="Please respond to the message",
    target_agent_id="Agent_B",
    include_profile=True,
    include_emotion=True,
    include_relation=True
)
```

## 使用 ProfileAgent

### 基本使用

```python
from profile import ProfileAgent, Profile, StaticTraits, EmotionVector

# 1. 创建 Profile
static_traits = StaticTraits(
    name="Alice",
    occupation="Software Engineer",
    mbti="INTJ",
    personality="Analytical and independent"
)
emotion_vector = EmotionVector()
profile = Profile(static_traits=static_traits, emotion_vector=emotion_vector)

# 2. 创建 LLM 模型（需要实现 generate 方法）
class MyLLM:
    def generate(self, system_message, prompt):
        # 调用 LLM API
        return True, "response"

llm = MyLLM()

# 3. 创建 ProfileAgent
agent = ProfileAgent(
    llm_model=llm,
    profile=profile,
    memory_size=100,
    instruction="You are a helpful assistant.",
    system_message="You are Alice, a software engineer."
)

# 4. 重置 Agent
agent.reset(
    goal="Have a conversation",
    init_obs="Agent B says: Hello, how are you?"
)

# 5. 运行 Agent 生成响应
success, action = agent.run()
print(f"Response: {action}")

# 6. 更新状态（会自动更新 Profile）
agent.update(action, "Agent B says: That's great!")

# 7. 再次运行（会考虑更新后的 Profile）
success, action = agent.run(target_agent_id="Agent_B")
```

### 关系感知的对话

```python
# ProfileAgent 会自动检测交互类型并更新关系
agent.reset(goal="Build relationship", init_obs="Agent C says: I don't trust you.")

# 第一次交互
success, action = agent.run(target_agent_id="Agent_C")

# 更新（检测到负面交互）
agent.update(action, "Agent C says: You're wrong!")

# 查看关系
relation = agent.social_profiler.get_relational_profile("Agent_C")
print(f"Trust: {relation['Trust']:.2f}")  # 信任度会下降

# 第二次交互（会考虑更新后的关系）
success, action = agent.run(target_agent_id="Agent_C")
```

### 情绪状态追踪

```python
# ProfileAgent 会自动追踪情绪状态
agent.reset(goal="Conversation", init_obs="Agent D says: You did a terrible job!")

# 运行（会触发情绪更新）
success, action = agent.run()

# 更新状态（检测到批评）
agent.update(action, "Agent D says: This is unacceptable!")

# 查看情绪状态
emotion = agent.get_emotion_state()
print(f"Neuroticism: {emotion['neuroticism']:.2f}")  # 神经质会上升
print(f"Stress: {emotion['stress']:.2f}")  # 压力会上升

# 获取情绪指导
guidance = agent.state_tracker.get_emotion_guidance()
print(guidance)  # "当前情绪较为敏感，回复时可能更具防御性"
```

### 从配置创建

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

## 与 Annotator 集成

从 Annotator 生成的 Profile 数据创建 Profile 对象：

```python
from profile import create_profile_from_annotator_data

# Annotator 生成的 profile 数据
annotator_data = {
    "gender": "Female",
    "age": 25,
    "interests": "reading, music, travel",
    "job": "Graphic Designer",
    "mbti": "ENFP",
    "personality": "Creative and enthusiastic",
    "style": "Expressive and friendly"
}

# 创建 Profile
profile = create_profile_from_annotator_data(annotator_data)
```

## 数据结构

### Profile 结构

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

## 序列化

Profile 支持 JSON 序列化：

```python
# 序列化
json_str = profile.to_json()

# 反序列化
profile = Profile.from_json(json_str)
```

## 示例

完整示例请参考 `example_usage.py`。

## ACL 卖点

1. **不仅模拟"人说什么"，还模拟"人心里怎么变"**
   - 通过 State Tracker 动态更新情绪向量
   - 情绪状态影响响应风格

2. **关系感知的对话生成**
   - 通过 Relational Profile 考虑与每个 Agent 的关系
   - 在 Prompt 中加入关系约束

3. **多层次 Profile 建模**
   - 静态特质（不可变）
   - 动态情绪（可更新）
   - 社交关系（结构化）

