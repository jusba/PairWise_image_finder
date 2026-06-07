import numpy as np
import torch


def load_image(gray_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a grayscale uint8 array (H×W) to a normalised float tensor (1×1×H×W)."""
    gray_tensor = torch.from_numpy(gray_np.astype(np.float32) / 255.0)
    return gray_tensor.unsqueeze(0).unsqueeze(0).to(device)
