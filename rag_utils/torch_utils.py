import torch


def torch_device(device_id: int = 0) -> torch.device:
    device, use_cuda = "cpu", False
    if torch.cuda.is_available():
        device, use_cuda = "cuda", True
        if isinstance(device_id, int) and device_id < torch.cuda.device_count():
            device = f"cuda:{device_id}"
    return torch.device(device)
