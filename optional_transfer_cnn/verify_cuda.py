from __future__ import annotations

import importlib

import torch


def package_version(package: str, module: str | None = None) -> str:
    try:
        mod = importlib.import_module(module or package)
    except ModuleNotFoundError:
        return "not installed"
    return getattr(mod, "__version__", "unknown")


print(f"torch: {torch.__version__}")
print(f"torch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available from this Python environment.")

print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
x = torch.randn(2048, 2048, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print(f"test tensor device: {y.device}")
print(f"test checksum: {float(y.mean().item()):.6f}")

for package, module in [
    ("numpy", None),
    ("pandas", None),
    ("scikit-learn", "sklearn"),
    ("scipy", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("opencv-python", "cv2"),
    ("scikit-image", "skimage"),
    ("pillow", "PIL"),
    ("xgboost", None),
]:
    print(f"{package}: {package_version(package, module)}")
