import torch
import sys
import torch

print("Python ejecutado en:", sys.executable)
print("torch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))