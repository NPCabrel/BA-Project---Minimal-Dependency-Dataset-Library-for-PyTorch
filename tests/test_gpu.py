import torch
import os

print(f"Hostname: {os.uname().nodename}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
  print(f"Number of GPUs: {torch.cuda.device_count()}")
  for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f" Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
  print("No GPU detected - are you running on a computer node?")
