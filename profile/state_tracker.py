"""
State Tracker: 状态追踪器
根据对话交互更新 V_emotion（情绪向量）
"""

from typing import Dict, Optional, Tuple
from .profile import Profile, EmotionVector
import re


class StateTracker:
    """
    状态追踪器：根据对话交互更新情绪向量
    
    Mechanism:
    Input: 上一轮对话 + 当前环境事件
    Process: 计算当前事件如何改变 V_emotion
    Output: 更新后的 Profile 用于指导 Response 生成
    """
    
    def __init__(self, profile: Profile, use_llm: bool = False, llm_model=None):
        """
        初始化状态追踪器
        
        Args:
            profile: 要追踪的 Profile
            use_llm: 是否使用 LLM 进行情绪分析（否则使用规则）
            llm_model: LLM 模型（如果 use_llm=True）
        """
        self.profile = profile
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.last_thought = ""  # 存储最近的思考过程
    
    def update_emotion(self, 
                      last_dialogue: str, 
                      current_event: str,
                      interaction_type: Optional[str] = None) -> Tuple[EmotionVector, str]:
        """
        根据对话和事件更新情绪向量
        
        Args:
            last_dialogue: 上一轮对话内容
            current_event: 当前环境事件
            interaction_type: 交互类型（如 "criticism", "praise", "conflict" 等）
        
        Returns:
            (更新后的情绪向量, 思考过程) 元组
        """
        emotion = self.profile.emotion_vector
        
        if self.use_llm and self.llm_model:
            # 使用 LLM 分析情绪变化
            updated_emotion, thought = self._update_with_llm(last_dialogue, current_event)
        else:
            # 使用规则/关键词分析情绪变化
            updated_emotion, thought = self._update_with_rules(last_dialogue, current_event, interaction_type)
        
        # 应用更新
        self.profile.emotion_vector = updated_emotion
        self.last_thought = thought  # 保存思考过程

        updated_emotion.clamp()  # 确保值在合理范围内
        
        return updated_emotion, thought
    
    def _update_with_rules(self, 
                          last_dialogue: str, 
                          current_event: str,
                          interaction_type: Optional[str] = None) -> Tuple[EmotionVector, str]:
        """
        使用规则更新情绪向量
        
        规则示例：
        - 被批评/辱骂 -> Neuroticism 上升, Trust 下降
        - 被表扬 -> Agreeableness 上升, Trust 上升
        - 冲突 -> Stress 上升, Neuroticism 上升
        - 友好互动 -> Trust 上升, Energy 上升
        """
        emotion = EmotionVector(
            openness=self.profile.emotion_vector.openness,
            conscientiousness=self.profile.emotion_vector.conscientiousness,
            extraversion=self.profile.emotion_vector.extraversion,
            agreeableness=self.profile.emotion_vector.agreeableness,
            neuroticism=self.profile.emotion_vector.neuroticism,
            trust=self.profile.emotion_vector.trust,
            stress=self.profile.emotion_vector.stress,
            energy=self.profile.emotion_vector.energy
        )
        
        # 合并对话和事件文本进行分析
        text = (last_dialogue + " " + current_event).lower()
        
        # 检测负面情绪关键词
        negative_keywords = [
            "stupid", "idiot", "wrong", "bad", "hate", "angry", "mad",
            "笨蛋", "白痴", "错误", "糟糕", "讨厌", "生气", "愤怒"
        ]
        positive_keywords = [
            "good", "great", "excellent", "love", "happy", "nice", "wonderful",
            "好", "棒", "优秀", "爱", "开心", "不错", "美好"
        ]
        conflict_keywords = [
            "fight", "argue", "disagree", "conflict", "against",
            "争吵", "争论", "不同意", "冲突", "反对"
        ]
        
        # 计算关键词匹配
        negative_count = sum(1 for kw in negative_keywords if kw in text)
        positive_count = sum(1 for kw in positive_keywords if kw in text)
        conflict_count = sum(1 for kw in conflict_keywords if kw in text)
        
        # 根据匹配结果更新情绪
        if negative_count > 0:
            # 被批评/辱骂
            emotion.neuroticism += 0.1 * negative_count
            emotion.trust -= 0.1 * negative_count
            emotion.stress += 0.1 * negative_count
            emotion.agreeableness -= 0.05 * negative_count
        
        if positive_count > 0:
            # 被表扬/友好互动
            emotion.agreeableness += 0.1 * positive_count
            emotion.trust += 0.1 * positive_count
            emotion.energy += 0.1 * positive_count
            emotion.stress -= 0.05 * positive_count
        
        if conflict_count > 0:
            # 冲突
            emotion.stress += 0.15 * conflict_count
            emotion.neuroticism += 0.1 * conflict_count
            emotion.trust -= 0.1 * conflict_count
        
        # 如果提供了交互类型，直接应用
        if interaction_type:
            if interaction_type == "criticism":
                emotion.neuroticism += 0.2
                emotion.trust -= 0.15
                emotion.stress += 0.2
            elif interaction_type == "praise":
                emotion.agreeableness += 0.2
                emotion.trust += 0.15
                emotion.energy += 0.15
            elif interaction_type == "conflict":
                emotion.stress += 0.25
                emotion.neuroticism += 0.15
                emotion.trust -= 0.15
            elif interaction_type == "friendly":
                emotion.trust += 0.15
                emotion.energy += 0.15
                emotion.agreeableness += 0.1
        
        # 生成规则基础的思考过程
        thought_parts = []
        if negative_count > 0:
            thought_parts.append(f"检测到 {negative_count} 个负面情绪关键词，这可能会增加神经质和压力，降低信任度。")
        if positive_count > 0:
            thought_parts.append(f"检测到 {positive_count} 个正面情绪关键词，这可能会增加亲和力和信任度，提升能量。")
        if conflict_count > 0:
            thought_parts.append(f"检测到 {conflict_count} 个冲突关键词，这可能会增加压力和神经质，降低信任度。")
        if interaction_type:
            thought_parts.append(f"交互类型为 {interaction_type}，应用相应的情绪变化规则。")
        
        thought = " ".join(thought_parts) if thought_parts else "使用规则方法分析情绪变化，未检测到明显的情绪触发词。"
        
        return emotion, thought
    
    def _update_with_llm(self, last_dialogue: str, current_event: str) -> Tuple[EmotionVector, str]:
        """
        使用 LLM 分析情绪变化
        
        这个方法使用 LLM 来分析对话和事件对情绪的影响
        先让 LLM 进行自然语言推理（thought），然后进行状态改变
        """
        # 获取当前情绪状态
        current_emotion = self.profile.emotion_vector
        
        # 构建 prompt - 要求 LLM 先给出思考过程，再给出状态变化
        system_message = "You are an expert at analyzing emotional states. Analyze how interactions affect emotional dimensions. First, provide your reasoning in natural language, then provide the emotional state changes."
        
        prompt = f"""Analyze how the following dialogue and event affect the emotional state.

Current Emotional State:
- Openness: {current_emotion.openness:.2f}
- Conscientiousness: {current_emotion.conscientiousness:.2f}
- Extraversion: {current_emotion.extraversion:.2f}
- Agreeableness: {current_emotion.agreeableness:.2f}
- Neuroticism: {current_emotion.neuroticism:.2f}
- Trust: {current_emotion.trust:.2f}
- Stress: {current_emotion.stress:.2f}
- Energy: {current_emotion.energy:.2f}

Dialogue: {last_dialogue}
Event: {current_event}

Please analyze this step by step:

1. First, provide your reasoning (thought) in natural language about how this dialogue and event might affect the emotional state. Consider the context, tone, and implications.

2. Then, provide a JSON object containing the changes (deltas) for each dimension, e.g.:
{{
    "openness": 0.05,
    "conscientiousness": -0.02,
    "extraversion": 0.1,
    "agreeableness": -0.1,
    "neuroticism": 0.15,
    "trust": -0.1,
    "stress": 0.2,
    "energy": -0.05
}}

Please respond in the following format:

THOUGHT: [Your natural language reasoning here. Structure your thought as follows:]
- Observation Analysis: What did you observe from the dialogue and event? What is the tone, context, and key information?
- Self-Reflection: How does this make you feel? What are your internal reactions and emotional responses?
- State Update Plan: Based on your analysis, which emotional dimensions should change and why?

Example THOUGHT:
"Observation Analysis: The neighbor is dismissing the emotional value of the trees and focusing purely on utility and safety. The tone is dismissive and practical, showing little regard for emotional attachment.
Self-Reflection: This feels cold to me. I feel my values are being trivialized. My stress is rising because I fear he doesn't care about the heritage and emotional significance I attach to these trees.
State Update Plan: Increase stress (due to feeling dismissed), increase neuroticism (due to emotional sensitivity), and decrease trust (due to feeling misunderstood)."

CHANGES: {{
    "openness": ...,
    "conscientiousness": ...,
    "extraversion": ...,
    "agreeableness": ...,
    "neuroticism": ...,
    "trust": ...,
    "stress": ...,
    "energy": ...
}}

Values should be between -1.0 and 1.0. Provide the THOUGHT first, then the CHANGES JSON object."""

        try:
            # 调用 LLM
            if hasattr(self.llm_model, 'generate'):
                success, response = self.llm_model.generate(system_message, prompt)
                if not success:
                    # 如果 LLM 调用失败，回退到规则方法
                    return self._update_with_rules(last_dialogue, current_event, None)
            else:
                # 如果 LLM 模型没有 generate 方法，尝试直接调用
                try:
                    response = self.llm_model(prompt)
                    success = True
                except Exception:
                    # 如果调用失败，回退到规则方法
                    return self._update_with_rules(last_dialogue, current_event, None)
            
            if not success:
                return self._update_with_rules(last_dialogue, current_event, None)
            
            # 解析 LLM 响应（提取 thought 和 JSON）
            import json
            import re
            
            # 提取 THOUGHT 部分
            thought = ""
            thought_match = re.search(r'THOUGHT:\s*(.+?)(?=CHANGES:|$)', response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1).strip()
            else:
                # 如果没有找到 THOUGHT 标记，尝试提取 CHANGES 之前的所有文本作为 thought
                changes_match = re.search(r'CHANGES:', response, re.IGNORECASE)
                if changes_match:
                    thought = response[:changes_match.start()].strip()
                    # 移除可能的标记
                    thought = re.sub(r'^(THOUGHT|thought|思考|推理):\s*', '', thought, flags=re.IGNORECASE)
                else:
                    thought = "LLM 分析了对话和事件对情绪的影响。"
            
            # 尝试提取 JSON 对象
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    deltas = json.loads(json_str)
                except json.JSONDecodeError:
                    # 如果 JSON 解析失败，回退到规则方法
                    return self._update_with_rules(last_dialogue, current_event, None)
            else:
                # 如果没有找到 JSON，回退到规则方法
                return self._update_with_rules(last_dialogue, current_event, None)
            
            # 应用变化
            updated_emotion = EmotionVector(
                openness=max(0.0, min(1.0, current_emotion.openness + deltas.get("openness", 0.0))),
                conscientiousness=max(0.0, min(1.0, current_emotion.conscientiousness + deltas.get("conscientiousness", 0.0))),
                extraversion=max(0.0, min(1.0, current_emotion.extraversion + deltas.get("extraversion", 0.0))),
                agreeableness=max(0.0, min(1.0, current_emotion.agreeableness + deltas.get("agreeableness", 0.0))),
                neuroticism=max(0.0, min(1.0, current_emotion.neuroticism + deltas.get("neuroticism", 0.0))),
                trust=max(0.0, min(1.0, current_emotion.trust + deltas.get("trust", 0.0))),
                stress=max(0.0, min(1.0, current_emotion.stress + deltas.get("stress", 0.0))),
                energy=max(0.0, min(1.0, current_emotion.energy + deltas.get("energy", 0.0)))
            )
            
            return updated_emotion, thought
            
        except Exception as e:
            # 如果出现任何错误，回退到规则方法
            import warnings
            warnings.warn(f"LLM-based emotion update failed: {e}. Falling back to rule-based method.")
            return self._update_with_rules(last_dialogue, current_event, None)
    
    def get_emotion_guidance(self) -> str:
        """
        根据当前情绪状态生成响应风格指导
        
        例如：如果 Neuroticism 高，回复应该更具防御性
        """
        emotion = self.profile.emotion_vector
        guidance = []
        
        if emotion.neuroticism > 0.7:
            guidance.append("Current emotion is quite sensitive, responses may be more defensive")
        elif emotion.neuroticism < 0.3:
            guidance.append("Current emotion is relatively stable, responses are more composed")
        
        if emotion.stress > 0.7:
            guidance.append("Current stress is high, responses may be shorter or more impatient")
        elif emotion.stress < 0.3:
            guidance.append("Current stress is low, responses are more relaxed and natural")
        
        if emotion.trust > 0.7:
            guidance.append("High trust in the current interlocutor, responses are more open and sincere")
        elif emotion.trust < 0.3:
            guidance.append("Low trust in the current interlocutor, responses are more cautious and reserved")
        
        if emotion.energy > 0.7:
            guidance.append("Current energy is high, responses are more positive and active")
        elif emotion.energy < 0.3:
            guidance.append("Current energy is low, responses may be shorter or tired")
        
        return "; ".join(guidance) if guidance else "Emotional state is normal, respond in your usual style"

