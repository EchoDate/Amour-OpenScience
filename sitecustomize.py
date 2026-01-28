"""
sitecustomize.py -  Python  profile 
 Python ，
"""
import sys
import os

# Project root directory
_project_root = os.path.dirname(os.path.abspath(__file__))
_profile_dir = os.path.join(_project_root, 'profile')

#  profile ， sys.path
if os.path.exists(_profile_dir):
    # (remove existing paths)
    paths_to_remove = ['', _project_root]
    for path in paths_to_remove:
        if path in sys.path:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
    
    # ，
    if _project_root not in sys.path:
        sys.path.append(_project_root)
