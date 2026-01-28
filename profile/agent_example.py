"""
ProfileAgent Usage Examples
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile import ProfileAgent, Profile, StaticTraits, EmotionVector, Relation, create_profile_from_annotator_data

# Import Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("Warning: vertexai is not installed, Gemini functionality will be unavailable. Please run: pip install google-cloud-aiplatform")


class MockLLM:
    """Mock LLM model (for testing)"""
    
    def __init__(self):
        self.context_length = 4096
        self.max_tokens = 512
        self.engine = "mock"
    
    def generate(self, system_message: str, prompt: str) -> tuple:
        """Simulate generating a response"""
        # Simple simulation: return a sample action
        response = "I understand. Let me help you with that."
        return True, response
    
    def num_tokens_from_messages(self, messages: list) -> int:
        """Estimate token count"""
        total_text = " ".join([msg.get("content", "") for msg in messages])
        return len(total_text.split())  # Simple estimation


class GeminiLLM:
    """Gemini LLM model wrapper (using Vertex AI)"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", config_path: str = "keys.json"):
        """
        Initialize Gemini LLM
        
        Args:
            model_name: Model name, such as "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"
            config_path: Configuration file path, containing project_id and location
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("vertexai is not installed, please run: pip install google-cloud-aiplatform")
        
        self.model_name = model_name
        self.engine = model_name
        
        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        if model_name not in configs:
            raise ValueError(f"Model '{model_name}' not found in configuration file")
        
        config = configs[model_name]
        project_id = config.get('project_id')
        location = config.get('location', 'us-central1')
        
        if not project_id or project_id == "your-gcp-project-id":
            raise ValueError(f"Please set a valid project_id in the configuration file")
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(model_name)
        
        # Set model parameters
        self.context_length = 8192  # Gemini 2.5 Flash context length
        self.max_tokens = 2048
    
    def generate(self, system_message: str, prompt: str) -> tuple:
        """
        Generate a response
        
        Args:
            system_message: System message
            prompt: User prompt
            
        Returns:
            (success, response): Whether successful, generated response
        """
        try:
            # Combine system message and prompt
            full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            
            # Call Gemini
            response = self._model.generate_content(full_prompt)
            
            if response and response.text:
                return True, response.text
            else:
                return False, "Generated response is empty"
        except Exception as e:
            return False, f"Generation error: {str(e)}"
    
    def num_tokens_from_messages(self, messages: list) -> int:
        """
        Estimate token count (simple estimation)
        
        Args:
            messages: Message list, format: [{"role": "system/user", "content": "..."}]
            
        Returns:
            Estimated token count
        """
        total_text = " ".join([msg.get("content", "") for msg in messages])
        # Simple estimation: English ~4 chars = 1 token, Chinese ~1.5 chars = 1 token
        # Using a conservative estimate here: average 3 chars = 1 token
        return len(total_text) // 3


def example_basic_usage():
    """Example 1: Basic Usage"""
    print("=" * 60)
    print("Example 1: ProfileAgent Basic Usage")
    print("=" * 60)
    
    # 1. Create Profile
    static_traits = StaticTraits(
        name="Alice",
        occupation="Software Engineer",
        age=28,
        mbti="INTJ",
        personality="Analytical and independent",
        talking_style="Direct and concise"
    )
    emotion_vector = EmotionVector(openness=0.6, extraversion=0.3)
    profile = Profile(static_traits=static_traits, emotion_vector=emotion_vector)
    
    # 2. Create LLM model (mock)
    llm = MockLLM()
    
    # 3. Create ProfileAgent
    agent = ProfileAgent(
        llm_model=llm,
        profile=profile,
        memory_size=50,
        instruction="You are a helpful assistant. Respond naturally based on your profile.",
        system_message="You are Alice, a software engineer."
    )
    
    # 4. Reset Agent
    agent.reset(
        goal="Have a conversation",
        init_obs="Agent B says: Hello, how are you?"
    )
    
    # 5. Run Agent to generate response
    success, action = agent.run()
    print(f"Success: {success}")
    print(f"Action: {action}")
    
    # 6. Update state
    agent.update(action, "Agent B says: That's great! I'm doing well too.")
    
    # 7. Run again
    success, action = agent.run()
    print(f"\nSecond response:")
    print(f"Action: {action}")
    
    # 8. View Profile state
    print(f"\nCurrent emotion state:")
    emotion = agent.get_emotion_state()
    for key, value in emotion.items():
        print(f"  {key}: {value:.2f}")


def example_with_relation():
    """Example 2: Using Relationship Awareness"""
    print("\n" + "=" * 60)
    print("Example 2: Relationship-Aware Conversation")
    print("=" * 60)
    
    # Create Profile
    profile = Profile(static_traits=StaticTraits(name="Bob"))
    llm = MockLLM()
    
    # Create Agent
    agent = ProfileAgent(
        llm_model=llm,
        profile=profile,
        instruction="Respond based on your relationship with the other agent."
    )
    
    # Reset
    agent.reset(
        goal="Build relationship",
        init_obs="Agent C says: I don't trust you."
    )
    
    # First interaction (negative)
    success, action = agent.run(target_agent_id="Agent_C")
    print(f"First response: {action}")
    
    # Update (negative interaction detected)
    agent.update(action, "Agent C says: You're wrong about that!")
    
    # View relationship
    relation = agent.social_profiler.get_relational_profile("Agent_C")
    print(f"\nRelationship with Agent_C:")
    print(f"  Trust: {relation['Trust']:.2f}")
    print(f"  Intimacy: {relation['Intimacy']:.2f}")
    
    # Second interaction (relationship updated)
    success, action = agent.run(target_agent_id="Agent_C")
    print(f"\nSecond response (with updated relation): {action}")


def example_emotion_tracking():
    """Example 3: Emotion Tracking"""
    print("\n" + "=" * 60)
    print("Example 3: Emotion State Tracking")
    print("=" * 60)
    
    # Create Profile
    profile = Profile(static_traits=StaticTraits(name="Charlie"))
    llm = MockLLM()
    
    # Create Agent
    agent = ProfileAgent(
        llm_model=llm,
        profile=profile
    )
    
    agent.reset(goal="Conversation", init_obs="Agent D says: You did a terrible job!")
    
    # Initial emotion state
    print("Initial emotion state:")
    emotion = agent.get_emotion_state()
    print(f"  Neuroticism: {emotion['neuroticism']:.2f}")
    print(f"  Stress: {emotion['stress']:.2f}")
    print(f"  Trust: {emotion['trust']:.2f}")
    
    # Run (will trigger emotion update)
    success, action = agent.run()
    
    # Update state (criticism detected)
    agent.update(action, "Agent D says: This is unacceptable!")
    
    # View updated emotion state
    print("\nAfter criticism:")
    emotion = agent.get_emotion_state()
    print(f"  Neuroticism: {emotion['neuroticism']:.2f}")
    print(f"  Stress: {emotion['stress']:.2f}")
    print(f"  Trust: {emotion['trust']:.2f}")
    
    # Get emotion guidance
    guidance = agent.state_tracker.get_emotion_guidance()
    print(f"\nResponse guidance: {guidance}")


def example_from_annotator():
    """Example 4: Creating Agent from Annotator Data"""
    print("\n" + "=" * 60)
    print("Example 4: Creating Agent from Annotator Data")
    print("=" * 60)
    
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
    
    # Create Profile from Annotator data
    profile = create_profile_from_annotator_data(annotator_data)
    
    # Create Agent
    llm = MockLLM()
    agent = ProfileAgent(
        llm_model=llm,
        profile=profile,
        instruction="Respond naturally based on your profile and personality."
    )
    
    print("Profile created from Annotator data:")
    print(agent.get_profile_summary())
    
    # Use Agent
    agent.reset(goal="Conversation", init_obs="Agent E says: Hi there!")
    success, action = agent.run()
    print(f"\nResponse: {action}")


def example_from_config():
    """Example 5: Creating Agent from Configuration"""
    print("\n" + "=" * 60)
    print("Example 5: Creating Agent from Configuration")
    print("=" * 60)
    
    # Create Profile
    profile = Profile(static_traits=StaticTraits(name="David"))
    llm = MockLLM()
    
    # Configuration
    config = {
        "memory_size": 100,
        "instruction": "You are a helpful assistant.",
        "examples": ["Example 1", "Example 2"],
        "system_message": "You are David, a friendly person.",
        "need_goal": False,
        "use_parser": True
    }
    
    # Create from configuration
    agent = ProfileAgent.from_config(llm, profile, config)
    
    print("Agent created from config")
    print(f"Memory size: {agent.memory_size}")
    print(f"Instruction: {agent.instruction}")


def example_with_gemini():
    """示例6: 使用 Gemini LLM 的 Agent"""
    print("\n" + "=" * 60)
    print("示例6: 使用 Gemini LLM 的 Agent")
    print("=" * 60)
    
    if not VERTEX_AI_AVAILABLE:
        print("错误: vertexai 未安装，无法使用 Gemini")
        print("请运行: pip install google-cloud-aiplatform")
        return
    
    try:
        # 1. 创建 Gemini LLM
        print("正在初始化 Gemini LLM...")
        llm = GeminiLLM(
            model_name="gemini-2.5-flash",
            config_path="keys.json"
        )
        print(f"✓ Gemini LLM 初始化成功 (模型: {llm.model_name})")
        
        # 2. 创建 Profile
        static_traits = StaticTraits(
            name="Emma",
            occupation="Data Scientist",
            age=30,
            mbti="INTP",
            personality="Analytical, curious, and logical",
            talking_style="Precise and data-driven"
        )
        emotion_vector = EmotionVector(openness=0.8, conscientiousness=0.7, extraversion=0.4)
        profile = Profile(static_traits=static_traits, emotion_vector=emotion_vector, relations={"Agent_F": Relation(intimacy=0.5, trust=0.5, dominance=0.0,relation_type="teacher_student")})
        
        # 3. Create ProfileAgent
        agent = ProfileAgent(
            llm_model=llm,
            profile=profile,
            memory_size=50,
            instruction="You are Emma, a data scientist. Respond naturally based on your profile and personality.",
            system_message="You are Emma, a data scientist who loves analyzing data and solving problems."
        )
        
        print("\nProfile Information:")
        print(agent.get_profile_summary())
        
        # 4. Reset Agent
        agent.reset(
            goal="Have a technical conversation",
            init_obs="Agent F says: Hi Emma! Can you help me understand machine learning?"
        )
        
        # 5. Run Agent to generate response
        print("\nGenerating response...")
        success, action = agent.run()
        
        if success:
            print(f"\n✓ Response generation successful:")
            print(f"Action: {action}")
        else:
            print(f"\n✗ Response generation failed: {action}")
        
        # 6. Update state and continue conversation
        agent.update(action, "Agent F says: Fuck you! I hate you. Can you explain neural networks?")
        
        print("\nContinuing conversation...")
        success, action = agent.run()
        
        if success:
            print(f"\n✓ Second response:")
            print(f"Action: {action}")
        else:
            print(f"\n✗ Second response failed: {action}")
        
        # 7. View emotion state
        print(f"\nCurrent emotion state:")
        emotion = agent.get_emotion_state()
        for key, value in emotion.items():
            print(f"  {key}: {value:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure keys.json file exists and contains correct configuration")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_basic_usage()
    example_with_relation()
    example_emotion_tracking()
    example_from_annotator()
    example_from_config()
    # Uncomment the line below to run Gemini example (requires keys.json configuration)
    example_with_gemini()

