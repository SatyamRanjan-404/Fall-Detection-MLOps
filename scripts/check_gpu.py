"""Check if GPU is visible to PyTorch and print device info."""
import sys

def main():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install -e .")
        print("For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version (PyTorch):", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 2**20:.0f} MiB)")
        print("Default device: cuda:0")
    else:
        print("GPU not available to PyTorch. Install CUDA build:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")

    

if __name__ == "__main__":
    main()