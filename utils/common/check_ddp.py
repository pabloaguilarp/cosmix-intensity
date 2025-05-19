import torch

def is_cuda_available():
    print("Checking CUDA availability...")
    # Check if GPUs are available
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    else:
        print("CUDA available")
        return True