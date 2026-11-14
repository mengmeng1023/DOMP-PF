"""Fixed-scale, aligned intensity inference; accepts images, never labels."""
import numpy as np
import torch
from torch.nn import functional as F


def gray_uint8(frame):
    """The frozen MATLAB-compatible grayscale conversion, without filtering."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError('Expected original uint8 image encoding')
    if not frame.size or not (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 3)):
        raise ValueError('Expected a nonempty grayscale or RGB image')
    if frame.ndim == 2:
        return frame
    rgb = frame.astype(np.float64)
    luminance = (0.298936021293775 * rgb[..., 0]
                 + 0.587043074451121 * rgb[..., 1]
                 + 0.114020904255103 * rgb[..., 2])
    return np.clip(np.floor(luminance + .5), 0, 255).astype(np.uint8)


def predict_intensity(model, frame, device='cpu'):
    """Return float64 native-size intensity in [0,255] after one sigmoid.

    Caller loads weights, moves the model to device, and measures load/warmup
    separately. CUDA synchronization here makes surrounding elapsed timing valid.
    """
    gray = gray_uint8(frame)
    height, width = gray.shape
    x = torch.from_numpy(np.array(gray, copy=True)).to(device=device, dtype=torch.float32)
    x = x[None, None] / 255.0
    x = F.pad(x, (0, (-width) % 16, 0, (-height) % 16), mode='replicate')
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        if logits.shape != x.shape or not torch.isfinite(logits).all():
            raise ValueError('Frontend must return finite raw B,1,H,W logits at input size')
        prediction = logits.sigmoid()[0, 0, :height, :width]
        result = prediction.to(device='cpu', dtype=torch.float64).numpy() * 255.0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result
