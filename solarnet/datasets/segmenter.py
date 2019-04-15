import numpy as np
import torch
from pathlib import Path
import random

from .utils import normalize
from .transforms import no_change, horizontal_flip, vertical_flip, colour_jitter


class SegmenterDataset:
    def __init__(self, processed_folder=Path('data/processed'), normalize=True,
                 transform_images=True, mask=None,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        self.device = device
        self.normalize = normalize
        self.transform_images = transform_images

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

    def _transform_images(self, image, mask):
        transforms = [
            no_change,
            horizontal_flip,
            vertical_flip,
            colour_jitter,
        ]
        chosen_function = random.choice(transforms)
        return chosen_function(image, mask)

    def __getitem__(self, index):

        x = np.load(self.org_solar_files[index])
        y = np.load(self.mask_solar_files[index])
        if self.transform_images: x, y = self._transform_images(x, y)
        if self.normalize: x = normalize(x)
        return torch.as_tensor(x.copy(), device=self.device).float(), \
            torch.as_tensor(y.copy(), device=self.device).float()
