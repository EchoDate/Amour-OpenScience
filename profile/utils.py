"""
工具函数
"""

from typing import Dict, Any, Optional
from .profile import Profile, StaticTraits, EmotionVector, Relation


def create_profile_from_annotator_data(profile_data: Dict[str, Any]) -> Profile:
    """
    从 annotator 生成的 Profile 数据创建 Profile 对象
    
    Args:
        profile_data: annotator 生成的 profile 数据，格式如：
            {
                "gender": "...",
                "age": ...,
                "interests": "...",
                "job": "...",
                "mbti": "...",
                "personality": "...",
                "style": "..."
            }
    
    Returns:
        Profile 对象
    """
    # 解析年龄（处理"38岁"这样的格式）
    age_value = profile_data.get("age", 0)
    age = 0
    if isinstance(age_value, int):
        age = age_value
    elif isinstance(age_value, str):
        # 提取数字部分（处理"38岁"、"25"等格式）
        import re
        age_match = re.search(r'\d+', age_value)
        if age_match:
            try:
                age = int(age_match.group(0))
            except ValueError:
                age = 0
        else:
            age = 0
    else:
        age = 0
    
    # 创建静态特质
    static_traits = StaticTraits(
        name=profile_data.get("name", ""),
        gender=profile_data.get("gender", ""),
        age=age,
        occupation=profile_data.get("job", ""),
        interests=profile_data.get("interests", "").split(",") if isinstance(profile_data.get("interests"), str) else profile_data.get("interests", []),
        mbti=profile_data.get("mbti", ""),
        personality=profile_data.get("personality", ""),
        talking_style=profile_data.get("style", "")
    )
    
    # 创建默认情绪向量（可以根据 MBTI 初始化）
    emotion_vector = EmotionVector()
    emotion_vector = _initialize_emotion_from_mbti(emotion_vector, static_traits.mbti)
    
    # 创建 Profile
    profile = Profile(
        static_traits=static_traits,
        emotion_vector=emotion_vector
    )
    
    return profile


def _initialize_emotion_from_mbti(emotion: EmotionVector, mbti: str) -> EmotionVector:
    """
    根据 MBTI 类型初始化情绪向量
    
    Args:
        emotion: 情绪向量对象
        mbti: MBTI 类型（如 "INTJ", "ENFP" 等）
    
    Returns:
        更新后的情绪向量
    """
    if not mbti or len(mbti) < 4:
        return emotion
    
    mbti_upper = mbti.upper()
    
    # E vs I (Extraversion)
    if mbti_upper[0] == 'E':
        emotion.extraversion = 0.7
    else:
        emotion.extraversion = 0.3
    
    # S vs N (Intuition/Openness)
    if mbti_upper[1] == 'N':
        emotion.openness = 0.7
    else:
        emotion.openness = 0.5
    
    # T vs F (Thinking/Agreeableness)
    if mbti_upper[2] == 'F':
        emotion.agreeableness = 0.7
    else:
        emotion.agreeableness = 0.5
    
    # J vs P (Judging/Conscientiousness)
    if mbti_upper[3] == 'J':
        emotion.conscientiousness = 0.7
    else:
        emotion.conscientiousness = 0.5
    
    return emotion


def extract_agent_id_from_text(text: str) -> Optional[str]:
    """
    从文本中提取 Agent ID
    
    Args:
        text: 包含 Agent ID 的文本
    
    Returns:
        Agent ID 或 None
    """
    import re
    
    # 尝试匹配常见的 Agent ID 格式
    patterns = [
        r'Agent\s+([A-Z])',  # "Agent A"
        r'Character\s+([A-Z])',  # "Character A"
        r'([A-Z])\s+said',  # "A said"
        r'from\s+([A-Z])',  # "from A"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def detect_interaction_type(text: str, llm_model=None) -> Optional[str]:
    """
    检测交互类型
    
    如果提供了 LLM 模型，使用 LLM 来判断交互类型；否则使用关键词匹配。
    
    Args:
        text: 交互文本
        llm_model: LLM 模型（可选），如果提供则使用 LLM 判断
    
    Returns:
        交互类型（如 "criticism", "praise", "conflict", "friendly", "help", "betrayal" 等）或 None
    """
    # 如果提供了 LLM 模型，使用 LLM 判断
    if llm_model is not None:
        return _detect_interaction_type_with_llm(text, llm_model)
    
    # 否则使用关键词匹配（向后兼容）
    return _detect_interaction_type_with_keywords(text)


def _detect_interaction_type_with_keywords(text: str) -> Optional[str]:
    """
    使用关键词匹配检测交互类型（向后兼容方法）
    
    Args:
        text: 交互文本
    
    Returns:
        交互类型或 None
    """
    text_lower = text.lower()
    
    # 批评相关
    criticism_keywords = ["stupid", "idiot", "wrong", "bad", "hate", "笨蛋", "白痴", "错误", "糟糕"]
    if any(kw in text_lower for kw in criticism_keywords):
        return "criticism"
    
    # 表扬相关
    praise_keywords = ["good", "great", "excellent", "love", "wonderful", "棒", "优秀", "美好"]
    if any(kw in text_lower for kw in praise_keywords):
        return "praise"
    
    # 冲突相关
    conflict_keywords = ["fight", "argue", "disagree", "conflict", "争吵", "争论", "冲突"]
    if any(kw in text_lower for kw in conflict_keywords):
        return "conflict"
    
    # 友好相关
    friendly_keywords = ["thank", "appreciate", "help", "谢谢", "感谢", "帮助"]
    if any(kw in text_lower for kw in friendly_keywords):
        return "friendly"
    
    # 帮助相关
    help_keywords = ["help", "assist", "support", "帮助", "协助", "支持"]
    if any(kw in text_lower for kw in help_keywords):
        return "help"
    
    return None


def _detect_interaction_type_with_llm(text: str, llm_model) -> Optional[str]:
    """
    使用 LLM 检测交互类型
    
    Args:
        text: 交互文本
        llm_model: LLM 模型
    
    Returns:
        交互类型或 None
    """
    # 构建 prompt
    system_message = "You are an expert at analyzing social interactions. Analyze the given text and determine the interaction type."
    
    prompt = f"""Analyze the following interaction text and determine its type.

Text: {text}

Possible interaction types:
- "criticism": Negative feedback, blame, or criticism
- "praise": Positive feedback, compliments, or appreciation
- "conflict": Disagreement, argument, or confrontation
- "friendly": Friendly conversation, greetings, or casual chat
- "help": Offering or requesting help, assistance, or support
- "betrayal": Betrayal, deception, or breaking trust
- None: If the interaction doesn't clearly fit any of the above categories

Respond with ONLY the interaction type (e.g., "criticism", "praise", "conflict", "friendly", "help", "betrayal", or "None").
Do not include any other text or explanation."""

    try:
        # 调用 LLM
        if hasattr(llm_model, 'generate'):
            success, response = llm_model.generate(system_message, prompt)
            if not success:
                # 如果 LLM 调用失败，回退到关键词匹配
                return _detect_interaction_type_with_keywords(text)
        else:
            # 如果 LLM 模型没有 generate 方法，尝试直接调用
            try:
                response = llm_model(prompt)
                success = True
            except Exception:
                # 如果调用失败，回退到关键词匹配
                return _detect_interaction_type_with_keywords(text)
        
        if not success:
            return _detect_interaction_type_with_keywords(text)
        
        # 解析 LLM 响应
        response = response.strip().lower()
        
        # 提取交互类型
        valid_types = ["criticism", "praise", "conflict", "friendly", "help", "betrayal"]
        for interaction_type in valid_types:
            if interaction_type in response:
                return interaction_type
        
        # 如果响应是 "none" 或类似，返回 None
        if "none" in response or "null" in response:
            return None
        
        # 如果无法解析，回退到关键词匹配
        return _detect_interaction_type_with_keywords(text)
        
    except Exception as e:
        # 如果出现任何错误，回退到关键词匹配
        import warnings
        warnings.warn(f"LLM-based interaction type detection failed: {e}. Falling back to keyword matching.")
        return _detect_interaction_type_with_keywords(text)


def merge_profiles(profile1: Profile, profile2: Profile, weight: float = 0.5) -> Profile:
    """
    合并两个 Profile（用于某些场景）
    
    Args:
        profile1: 第一个 Profile
        profile2: 第二个 Profile
        weight: 合并权重（0-1，0.5 表示平均）
    
    Returns:
        合并后的 Profile
    """
    # 静态特质不能合并，使用 profile1 的
    merged = Profile(
        static_traits=profile1.static_traits,
        emotion_vector=EmotionVector(
            openness=profile1.emotion_vector.openness * (1-weight) + profile2.emotion_vector.openness * weight,
            conscientiousness=profile1.emotion_vector.conscientiousness * (1-weight) + profile2.emotion_vector.conscientiousness * weight,
            extraversion=profile1.emotion_vector.extraversion * (1-weight) + profile2.emotion_vector.extraversion * weight,
            agreeableness=profile1.emotion_vector.agreeableness * (1-weight) + profile2.emotion_vector.agreeableness * weight,
            neuroticism=profile1.emotion_vector.neuroticism * (1-weight) + profile2.emotion_vector.neuroticism * weight,
            trust=profile1.emotion_vector.trust * (1-weight) + profile2.emotion_vector.trust * weight,
            stress=profile1.emotion_vector.stress * (1-weight) + profile2.emotion_vector.stress * weight,
            energy=profile1.emotion_vector.energy * (1-weight) + profile2.emotion_vector.energy * weight,
        )
    )
    
    # 合并关系（取并集，值取平均）
    all_agent_ids = set(profile1.relations.keys()) | set(profile2.relations.keys())
    for agent_id in all_agent_ids:
        rel1 = profile1.get_relation(agent_id)
        rel2 = profile2.get_relation(agent_id)
        
        # 合并关系类型：优先使用非 None 的值，如果两个都有则使用第一个
        merged_relation_type = rel1.relation_type if rel1.relation_type is not None else rel2.relation_type
        
        merged_rel = Relation(
            intimacy=rel1.intimacy * (1-weight) + rel2.intimacy * weight,
            trust=rel1.trust * (1-weight) + rel2.trust * weight,
            dominance=rel1.dominance * (1-weight) + rel2.dominance * weight,
            relation_type=merged_relation_type
        )
        merged.update_relation(agent_id, merged_rel)
    
    return merged

