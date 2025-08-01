import importlib
import os

# for module api
curr_dir = os.path.abspath(os.path.dirname(__file__))
entries = os.listdir(curr_dir)
for entry in entries:
    path = os.path.join(curr_dir, entry)
    if os.path.isdir(path):
        if entry == "__pycache__":
            continue
        importlib.import_module(f"dax.parallel.patches.{entry}")
