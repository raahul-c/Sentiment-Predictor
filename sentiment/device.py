"""Pick the compute device: CUDA (NVIDIA, or AMD ROCm on Linux), then DirectML (any DirectX 12
GPU on Windows, e.g. AMD Radeon), then CPU."""
import torch


def _directml():
    try:
        import torch_directml
    except ModuleNotFoundError:
        return None
    except (ImportError, OSError) as e:  # installed but broken, or no DirectX 12 runtime
        print(f"  torch-directml could not be loaded ({e}); not using DirectML")
        return None
    if not torch_directml.is_available():
        print("  torch-directml is installed but found no DirectX 12 GPU; not using DirectML")
        return None
    return torch_directml


def pick_device(name="auto"):
    """Returns (device, description). name: auto, cuda, dml, cpu (or any torch device string)."""
    name = name.lower()
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"
    if name in ("auto", "dml", "directml"):
        dml = _directml()
        if dml is not None:
            return dml.device(), f"DirectML ({dml.device_name(dml.default_device())})"
        if name != "auto":
            raise SystemExit("DirectML requested but torch-directml is not installed or found no GPU.")
    if name in ("auto", "cpu"):
        return torch.device("cpu"), "cpu"
    if name == "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(name), name


def is_directml(device):
    return device.type == "privateuseone"
