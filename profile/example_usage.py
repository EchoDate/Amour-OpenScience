"""
Profile Module Usage Examples
"""

from profile import (
    Profile,
    StaticTraits,
    EmotionVector,
    StateTracker,
    SocialProfiler,
    MemoryEnhancer,
    PromptEnhancer,
    create_profile_from_annotator_data
)


def example_basic_profile():
    """Example 1: Creating Basic Profile"""
    print("=" * 60)
    print("Example 1: Creating Basic Profile")
    print("=" * 60)
    
    # Create static traits
    static_traits = StaticTraits(
        name="Alice",
        occupation="Software Engineer",
        age=28,
        gender="Female",
        interests=["coding", "reading", "hiking"],
        mbti="INTJ",
        personality="Analytical and independent",
        talking_style="Direct and concise"
    )
    
    # Create emotion vector
    emotion_vector = EmotionVector(
        openness=0.6,
        conscientiousness=0.8,
        extraversion=0.3,
        agreeableness=0.5,
        neuroticism=0.4
    )
    
    # Create Profile
    profile = Profile(
        static_traits=static_traits,
        emotion_vector=emotion_vector
    )
    
    print(profile.get_summary())
    print("\nProfile JSON:")
    print(profile.to_json())


def example_state_tracker():
    """Example 2: Using State Tracker to Update Emotions"""
    print("\n" + "=" * 60)
    print("Example 2: Using State Tracker to Update Emotions")
    print("=" * 60)
    
    # Create Profile
    static_traits = StaticTraits(name="Bob", occupation="Teacher")
    emotion_vector = EmotionVector()
    profile = Profile(static_traits=static_traits, emotion_vector=emotion_vector)
    
    # Create State Tracker
    tracker = StateTracker(profile)
    
    print("Initial emotion state:")
    print(f"Neuroticism: {profile.emotion_vector.neuroticism:.2f}")
    print(f"Trust: {profile.emotion_vector.trust:.2f}")
    print(f"Stress: {profile.emotion_vector.stress:.2f}")
    
    # Simulate criticism scenario
    print("\nEvent: Received criticism")
    tracker.update_emotion(
        last_dialogue="You did a terrible job!",
        current_event="Received criticism",
        interaction_type="criticism"
    )
    
    print("\nUpdated emotion state:")
    print(f"Neuroticism: {profile.emotion_vector.neuroticism:.2f}")
    print(f"Trust: {profile.emotion_vector.trust:.2f}")
    print(f"Stress: {profile.emotion_vector.stress:.2f}")
    
    # Get response guidance
    print("\nResponse guidance:")
    print(tracker.get_emotion_guidance())


def example_social_profiling():
    """Example 3: Using Social Profiling to Manage Relationships"""
    print("\n" + "=" * 60)
    print("Example 3: Using Social Profiling to Manage Relationships")
    print("=" * 60)
    
    # Create Profile
    profile = Profile(static_traits=StaticTraits(name="Charlie"))
    
    # Create Social Profiler
    profiler = SocialProfiler(profile)
    
    # Get Inner Profile
    print("Inner Profile:")
    inner = profiler.get_inner_profile()
    print(f"Name: {inner['name']}")
    print(f"Personality: {inner.get('personality', 'N/A')}")
    
    # Update relationship with Agent B
    print("\nInteraction with Agent B: Friendly interaction")
    profiler.update_relation("Agent_B", "Thank you for your help!", interaction_type="friendly")
    
    # Get relationship information
    relation = profiler.get_relational_profile("Agent_B")
    print(f"\nRelationship with Agent B:")
    print(f"  Intimacy: {relation['Intimacy']:.2f}")
    print(f"  Trust: {relation['Trust']:.2f}")
    print(f"  Dominance: {relation['Dominance']:.2f}")
    
    # Get relationship constraint prompt
    print("\nRelationship constraint prompt:")
    print(profiler.get_relation_constraint_prompt("Agent_B"))


def example_memory_enhancer():
    """Example 4: Using Memory Enhancer to Enhance Memory Retrieval"""
    print("\n" + "=" * 60)
    print("Example 4: Using Memory Enhancer to Enhance Memory Retrieval")
    print("=" * 60)
    
    # Create Profile
    profile = Profile(static_traits=StaticTraits(name="David"))
    
    # Create Memory Enhancer
    enhancer = MemoryEnhancer(profile)
    
    # Simulate memory
    memory = [
        ("Action", "Hello Agent B"),
        ("Observation", "Agent B greeted back"),
        ("Action", "How are you?"),
        ("Observation", "Agent B said: I'm fine, thanks!"),
        ("Action", "That's great!"),
        ("Observation", "Agent B smiled")
    ]
    
    # Retrieve memory based on Profile
    enhanced_memory = enhancer.retrieve_with_profile(
        memory,
        current_agent_id="Agent_B",
        max_items=4
    )
    
    print("Enhanced memory:")
    for item in enhanced_memory:
        print(f"  {item[0]}: {item[1]}")


def example_prompt_enhancer():
    """Example 5: Using Prompt Enhancer to Enhance Prompts"""
    print("\n" + "=" * 60)
    print("Example 5: Using Prompt Enhancer to Enhance Prompts")
    print("=" * 60)
    
    # Create Profile
    static_traits = StaticTraits(
        name="Eve",
        occupation="Psychologist",
        personality="Empathetic and understanding",
        talking_style="Warm and supportive"
    )
    profile = Profile(static_traits=static_traits)
    
    # Create Prompt Enhancer
    enhancer = PromptEnhancer(profile)
    
    # Base prompt
    base_prompt = "Please respond to the following message: 'How are you doing?'"
    
    # Enhance prompt
    enhanced_prompt = enhancer.enhance_prompt(
        base_prompt,
        target_agent_id="Agent_F",
        include_profile=True,
        include_emotion=True,
        include_relation=True
    )
    
    print("Enhanced prompt:")
    print(enhanced_prompt)
    
    # Get response style instruction
    print("\nResponse style instruction:")
    print(enhancer.get_response_style_instruction("Agent_F"))


def example_from_annotator_data():
    """Example 6: Creating Profile from Annotator Data"""
    print("\n" + "=" * 60)
    print("Example 6: Creating Profile from Annotator Data")
    print("=" * 60)
    
    # Simulate profile data generated by annotator
    annotator_data = {
        "gender": "Female",
        "age": 25,
        "interests": "reading, music, travel",
        "job": "Graphic Designer",
        "mbti": "ENFP",
        "personality": "Creative and enthusiastic",
        "style": "Expressive and friendly"
    }
    
    # Create Profile from annotator data
    profile = create_profile_from_annotator_data(annotator_data)
    
    print("Profile created from Annotator data:")
    print(profile.get_summary())


if __name__ == "__main__":
    # Run all examples
    example_basic_profile()
    example_state_tracker()
    example_social_profiling()
    example_memory_enhancer()
    example_prompt_enhancer()
    example_from_annotator_data()

