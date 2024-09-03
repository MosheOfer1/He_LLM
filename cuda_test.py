import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# Check CUDA version being used
cuda_version = torch.version.cuda
print(f"CUDA Version: {cuda_version}")

# Check GPU details
if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
