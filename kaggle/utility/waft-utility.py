"""Utility script: cài đặt PyTorch + dependencies cho WAFT-Stereo training trên Kaggle.

Script này chạy trên kernel có internet, output sẽ được mount vào training kernel
(không có internet) tại /kaggle/input/waft-utility/.

Packages được cài vào /kaggle/working/ → trở thành output của kernel.
"""
import os
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = f"/kaggle/working:{env.get('PYTHONPATH', '')}"

commands = [
    # PyTorch nightly với CUDA 12.8 (Blackwell GPU support)
    "uv pip uninstall torch torchvision torchaudio",
    "uv pip install --target=/kaggle/working --system --pre torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/nightly/cu128",

    # WAFT-Stereo dependencies
    "uv pip install --target=/kaggle/working --system "
    "timm einops kornia antialiased-cnns peft transformers accelerate "
    "tensorboard tqdm pyyaml yacs termcolor tabulate "
    "opencv-python-headless pandas matplotlib "
    "h5py imageio scipy Pillow",
]

for cmd in commands:
    print(f"Running: {cmd[:80]}...")
    subprocess.run(cmd, shell=True, check=True, env=env)

print("\nAll WAFT-Stereo dependencies installed successfully!")
print("Output will be available at /kaggle/input/waft-utility/ in training kernel.")
