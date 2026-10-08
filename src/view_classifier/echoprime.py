from typing import Union, Optional

import numpy as np 

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import v2 as transforms
from torchvision.models import convnext_base

ECHOPRIME_MEAN = [29.110628, 28.076836, 29.096405]
ECHOPRIME_STD = [47.989223, 46.456997, 47.20083]
ECHOPRIME_RESOLUTION = (224, 224)
ECHOPRIME_VIEWS = [
    'A2C',
    'A3C',
    'A4C',
    'A5C',
    'Apical_Doppler',
    'Doppler_Parasternal_Long',
    'Doppler_Parasternal_Short',
    'Parasternal_Long',
    'Parasternal_Short',
    'SSN',
    'Subcostal'
]


def setup_echoprime_view_classifier(weights, device: str, compile: bool=False):
    """Initialize the EchoPrime view classifier on `device`."""
    model = EchoPrimeViewClassifier(weights=weights).to(device).eval()
    if compile:
        model = torch.compile(model)
    echo_transforms = transforms.Compose([
        transforms.Resize(ECHOPRIME_RESOLUTION),
        transforms.ToDtype(torch.float32),
        transforms.Grayscale(num_output_channels=3), 
        transforms.Normalize(mean=ECHOPRIME_MEAN, std=ECHOPRIME_STD),
    ])
    return model, echo_transforms  


def classify_view_torch(echoprime_model, echoprime_transforms, video: np.ndarray, batch_size: int = 32, device: Union[str, torch.device] = 'cuda') -> np.ndarray:
    """Run EchoPrime on `video` using this worker's PyTorch model and device."""
    B, H, W, C = video.shape
    assert (H, W) == ECHOPRIME_RESOLUTION, f"Expected image resolution {ECHOPRIME_RESOLUTION}, got {(H, W)}"
    assert C == 3, f"Expected num_channels=3, got {C}"
    extra_batch = 0 if B % batch_size == 0 else 1
    video_pt = echoprime_transforms(torch.from_numpy(video).permute(0, 3, 1, 2))
    probits = []
    for k in range(B // batch_size + extra_batch):
        image_batch = video_pt[k * batch_size:min((k + 1) * batch_size, B)].to(device)
        with torch.no_grad():
            batch_probits = echoprime_model(image_batch).softmax(dim=-1)
        probits.append(batch_probits.cpu().numpy())
    return np.concatenate(probits, axis=0)


class EchoPrimeViewClassifier(nn.Module):
    def __init__(self, weights: Optional[str] = None):
        super(EchoPrimeViewClassifier, self).__init__()
        self.model = convnext_base()
        self.model.classifier[-1] = torch.nn.Linear(
            self.model.classifier[-1].in_features, 11
        )
        if weights is not None:
            vc_state_dict = torch.load(weights, weights_only=False, map_location='cpu')
            self.model.load_state_dict(vc_state_dict)

        self.view_names = ECHOPRIME_VIEWS

    def forward(self, x):
        assert x.shape[2:] == ECHOPRIME_RESOLUTION, f"Expected input resolution to be {ECHOPRIME_RESOLUTION}, got {x.shape[2:]}" 
        return self.model(x)
