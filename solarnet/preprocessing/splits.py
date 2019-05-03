import numpy as np
from numpy.random import randint
import pandas as pd
from collections import defaultdict
from pathlib import Path
import rasterio
from tqdm import tqdm

from typing import Tuple

from .masks import IMAGE_SIZES


class ImageSplitter:
    """Solar panels cover a relatively small landmass compared to
    the total image sizes, so to make sure there are contiguous solar
    panels in all segments, we will center each image on the solar panel.
    For every image, we will also randomly sample (at a buffer away from the
    solar panels) a number of images which don't contain any solar panels
    to pretrain the network

    Attributes:
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in data/README.md
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder

        # setup; make the necessary folders
        self.processed_folder = data_folder / 'processed'
        if not self.processed_folder.exists(): self.processed_folder.mkdir()

        self.solar_panels = self._setup_folder('solar')
        self.empty = self._setup_folder('empty')

    def _setup_folder(self, folder_name: str) -> Path:

        full_folder_path = self.processed_folder / folder_name
        if not full_folder_path.exists(): full_folder_path.mkdir()

        for subfolder in ['org', 'mask']:
            subfolder_name = full_folder_path / subfolder
            if not subfolder_name.exists(): subfolder_name.mkdir()

        return full_folder_path

    def read_centroids(self) -> defaultdict:

        metadata = pd.read_csv(self.data_folder / 'metadata/polygonDataExceptVertices.csv',
                               usecols=['city', 'image_name', 'centroid_latitude_pixels',
                                        'centroid_longitude_pixels'])
        org_len = len(metadata)
        metadata = metadata.dropna()
        print(f'Dropped {org_len - len(metadata)} rows due to NaN values')

        # for each image, we want to know where the solar panel centroids are
        output_dict: defaultdict = defaultdict(lambda: defaultdict(set))

        for idx, row in metadata.iterrows():
            output_dict[row.city][row.image_name].add((
                row.centroid_latitude_pixels, row.centroid_longitude_pixels
            ))
        return output_dict

    @staticmethod
    def adjust_coords(coords: Tuple[float, float], image_radius: int,
                      org_imsize: Tuple[int, int]) -> Tuple[float, float]:
        x_imsize, y_imsize = org_imsize
        x, y = coords
        # we make sure that the centroid isn't at the edge of the image
        if x < image_radius: x = image_radius
        elif x > (x_imsize - image_radius): x = x_imsize - image_radius

        if y < image_radius: y = image_radius
        elif y > (y_imsize - image_radius): y = y_imsize - image_radius
        return x, y

    @staticmethod
    def size_okay(image: np.array, imsize: int) -> bool:
        if image.shape == (3, imsize, imsize):
            return True
        return False

    def process(self, imsize: int = 224, empty_ratio: int = 2) -> None:
        """Creates the solar and empty images, and their corresponding masks

        Parameters
        ----------
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.

        Images and masks are saved in {solar, empty}/{org_mask}, with their original
        city in their filename (i.e. {city}_{idx}.npy) where idx is incremented with every
        image-mask combination saved to ensure uniqueness
        """

        image_radius = imsize // 2
        centroids_dict = self.read_centroids()

        im_idx = 0
        for city, images in centroids_dict.items():
            print(f"Processing {city}")
            for image_name, centroids in tqdm(images.items()):

                org_file = rasterio.open(self.data_folder / f"{city}/{image_name}.tif").read()

                org_x_imsize, org_y_imsize = IMAGE_SIZES[city]
                if org_file.shape != (3, org_x_imsize, org_y_imsize):
                    print(f'{city}/{image_name}.tif is malformed with shape {org_file.shape}. '
                          f'Skipping!')
                    continue
                mask_file = np.load(self.data_folder / f"{city}_masks/{image_name}.npy")

                # first, lets collect the positive examples
                for centroid in centroids:
                    x, y = self.adjust_coords(centroid, image_radius, (org_x_imsize, org_y_imsize))
                    max_width, max_height = int(x + image_radius), int(y + image_radius)
                    min_width, min_height = max_width - imsize, max_height - imsize

                    clipped_orgfile = org_file[:, min_width: max_width, min_height: max_height]
                    mask = mask_file[min_width: max_width, min_height: max_height]

                    if self.size_okay(clipped_orgfile, imsize):
                        np.save(self.solar_panels / f"org/{city}_{im_idx}.npy", clipped_orgfile)
                        np.save(self.solar_panels / f"mask/{city}_{im_idx}.npy", mask)

                        im_idx += 1

                # next, the negative examples. We randomly search for negative examples
                # until we have a) found the number we want, or b) hit positive examples
                # too often (and maxed out our patience), in which case we give up
                # this is a crude way of doing it, and could probably be improved
                patience, max_patience = 0, 10
                num_empty, max_num_empty = 0, len(centroids) * empty_ratio
                while (patience < max_patience) and (num_empty < max_num_empty):
                    rand_x = randint(0, org_x_imsize - imsize)
                    rand_y = randint(0, org_y_imsize - imsize)
                    rand_x_max, rand_y_max = rand_x + imsize, rand_y + imsize
                    # this makes sure no solar panel is present
                    mask_candidate = mask_file[rand_x: rand_x_max, rand_y: rand_y_max]

                    if mask_candidate.sum() == 0:
                        clipped_orgfile = org_file[:, rand_x: rand_x_max, rand_y: rand_y_max]
                        if self.size_okay(clipped_orgfile, imsize):
                            np.save(self.empty / f"org/{city}_{im_idx}.npy",
                                    org_file[:, rand_x: rand_x_max, rand_y: rand_y_max])
                            np.save(self.empty / f"mask/{city}_{im_idx}.npy",
                                    mask_candidate)
                            im_idx += 1
                            num_empty += 1
                    else:
                        patience += 1
        print(f"Generated {im_idx} samples")
