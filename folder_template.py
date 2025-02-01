import os
from pathlib import Path

project_name = "smoking_history_prediction"

files_to_add = [
    f"{project_name}/__init__.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    ".gitignore"
]

for path in files_to_add:
    file_path = Path(path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass