"""
Cloud Function entry point for Amour directory
This is the main entry point file required by Cloud Functions
"""
import sys
from pathlib import Path

# Add annotator directory to path
annotator_dir = Path(__file__).parent / "annotator"
if str(annotator_dir) not in sys.path:
    sys.path.insert(0, str(annotator_dir))

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import the actual function from annotator.cloud_functions.main
# This function already has @functions_framework.http decorator
from annotator.cloud_functions.main_rough import generate_conversation_rough
from annotator.cloud_functions.main_origin import generate_conversation
from annotator.cloud_functions.main_qwen import generate_conversation_qwen
from annotator.cloud_functions.main_rough_CoST import generate_conversation_rough_thought
from annotator.cloud_functions.main_origin_CoST import generate_conversation_CoST
from annotator.cloud_functions.main_deepseek_CoST import generate_conversation_deepseek_CoST
from annotator.cloud_functions.main_deepseek_CoST_batch import generate_conversation_deepseek_CoST_batch
from annotator.cloud_functions.main_deepseek_rough_CoST import generate_conversation_deepseek_rough_CoST

# Re-export for Cloud Functions (the function is already decorated)
__all__ = ['generate_conversation','generate_conversation_rough', 'generate_conversation_qwen', 'generate_conversation_rough_thought', 'generate_conversation_CoST', 
           'generate_conversation_deepseek_CoST', 'generate_conversation_deepseek_CoST_batch','generate_conversation_deepseek_rough_CoST']
