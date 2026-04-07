"""Utility script: cài đặt PyTorch + dependencies cho JeDepth training trên Kaggle (Blackwell GPU).

Script này chạy trên kernel có internet, output sẽ được mount vào training kernel
(không có internet) tại /kaggle/input/jedepth-utility-script/.

Packages cài vào /kaggle/working/ → trở thành output của kernel.
"""
import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = f"/kaggle/working:{env.get('PYTHONPATH', '')}"

commands = [
    # PyTorch nightly với CUDA 12.8 (Blackwell GPU support)
    "uv pip uninstall torch torchvision torchaudio",
    "uv pip install --target=/kaggle/working --system --pre torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/nightly/cu128",

    # Depth training dependencies
    "uv pip install --target=/kaggle/working --system "
    "easydict opencv-python-headless matplotlib pandas tensorboard tqdm "
    "kornia timm antialiased-cnns pyyaml",
]

for cmd in commands:
    print(f"Running: {cmd[:80]}...")
    subprocess.run(cmd, shell=True, check=True, env=env)

print("\nAll JeDepth dependencies installed successfully!")
print("Output will be available at /kaggle/input/jedepth-utility-script/ in training kernel.")
