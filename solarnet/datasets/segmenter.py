import numpy as np
import torch
from pathlib import Path

from .utils import normalize


class SegmenterDataset:
    def __init__(self, processed_folder=Path('data/processed'), normalize=True,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 mask=None):

        self.device = device
        self.normalize = normalize

        # We will only segment the images which we know have solar panels in them; the
        # other images should be filtered out by the classifier
        solar_folder = processed_folder / 'solar'

        self.org_solar_files = list((solar_folder/ 'org').glob("*.npy"))
        self.mask_solar_files = []
        # we want to make sure the masks and org files align; this is to make sure they do
        for file in self.org_solar_files:
            self.mask_solar_files.append(solar_folder / 'mask' / file.name)

        if mask is not None:
            self.add_mask(mask)

    def add_mask(self, mask):
        """Add a mask to the data
        """
        assert len(mask) == len(self.org_solar_files), \
            f"Mask is the wrong size! Expected {len(self.org_solar_files)}, got {len(mask)}"
        self.org_solar_files = [x for include, x in zip(mask, self.org_solar_files) if include]
        self.mask_solar_files = [x for include, x in zip(mask, self.mask_solar_files) if include]

    def __len__(self):
        return len(self.org_solar_files)

    def __getitem__(self, index):

        x = np.load(self.org_solar_files[index])
        if self.normalize: x = normalize(x)
        y = np.load(self.mask_solar_files[index])
        return torch.as_tensor(x, device=self.device).float(), \
            torch.as_tensor(y, device=self.device).float()
