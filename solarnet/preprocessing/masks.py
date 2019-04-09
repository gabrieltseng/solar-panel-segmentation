import pandas as pd
import numpy as np
from matplotlib.path import Path as PolygonPath
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

IMAGE_SIZES = {
    'Modesto': (5000, 5000),
    'Fresno': (5000, 5000),
    'Oxnard': (4000, 6000),
    'Stockton': (5000, 5000)
}


class MaskMaker:
    """This class looks for all files defined in the metadata, and
    produces masks for all of the .tif files saved there.
    These files will be saved in <org_folder>_mask/<org_filename>.npy

    Attributes:
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
    """

    def __init__(self, data_folder=Path('data')):
        self.data_folder = data_folder

    def _read_data(self):
        metadata_folder = self.data_folder / 'metadata'

        polygon_pixels = self._csv_to_dict_polygon_pixels(
            pd.read_csv(metadata_folder / 'polygonVertices_PixelCoordinates.csv')
        )
        # TODO: potentially filter on jaccard index
        polygon_images = self._csv_to_dict_image_names(
            pd.read_csv(metadata_folder / 'polygonDataExceptVertices.csv',
                        usecols=['polygon_id', 'city', 'image_name', 'jaccard_index']
                        )
        )
        return polygon_images, polygon_pixels

    def process(self):

        polygon_images, polygon_pixels = self._read_data()

        for city, files in polygon_images.items():
            print(f'Processing {city}')
            if city != 'Oxnard': continue
            # first, we make sure the mask file exists; if not,
            # we make it
            masked_city = self.data_folder / f"{city}_masks"
            x_size, y_size = IMAGE_SIZES[city]
            if not masked_city.exists(): masked_city.mkdir()

            for image, polygons in tqdm(files.items()):
                mask = np.zeros((x_size, y_size))
                for polygon in polygons:
                    mask += self.make_mask(polygon_pixels[polygon], (x_size, y_size))

                np.save(masked_city / f"{image}.npy", mask)

    @staticmethod
    def _csv_to_dict_polygon_pixels(polygon_pixels):
        output_dict = {}

        for idx, row in polygon_pixels.iterrows():
            vertices = []
            for i in range(1, int(row.number_vertices) + 1):
                vertices.append((row[f"lat{i}"], row[f"lon{i}"]))
            output_dict[int(row.polygon_id)] = vertices
        return output_dict

    @staticmethod
    def _csv_to_dict_image_names(polygon_images):
        output_dict = defaultdict(lambda: defaultdict(list))

        for idx, row in polygon_images.iterrows():
                output_dict[row.city][row.image_name].append(int(row.polygon_id))
        return output_dict

    @staticmethod
    def make_mask(coords, imsizes):
        """https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        """
        poly_path = PolygonPath(coords)

        x_size, y_size = imsizes
        x, y = np.mgrid[:x_size, :y_size]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = poly_path.contains_points(coors)

        return mask.reshape(x_size, y_size).astype(float)
