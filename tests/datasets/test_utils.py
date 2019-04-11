import numpy as np

from solarnet.datasets.utils import normalize, denormalize, make_masks


class TestUtils:

    def test_normalize_denormalize(self):
        input_image = np.random.randint(low=0, high=255, size=(3, 244, 244))

        reconstructed_image = denormalize(normalize(input_image))

        # An absolute tolerance of 1 is set to handle rounding errors
        assert np.allclose(input_image, reconstructed_image, atol=1), \
            f'Image not properly reconstructed'

    def test_make_mask(self):

        mask_length = 10000
        train_mask, val_mask, test_mask = make_masks(mask_length, 0.1, 0.1)

        # first, make sure they are all the right length
        assert np.isclose(val_mask.sum(), mask_length * 0.1, rtol=0.1), \
            f'Got {val_mask.sum()} val samples, expected {mask_length * 0.1}'

        assert np.isclose(test_mask.sum(), mask_length * 0.1, rtol=0.1), \
            f'Got {test_mask.sum()} test samples, expected {mask_length * 0.1}'

        assert np.isclose(train_mask.sum(), mask_length * 0.8, rtol=0.1), \
            f'Got {train_mask.sum()} val samples, expected {mask_length * 0.8}'

        # Next, make sure they are not overlapping, and that no instances were left out
        mask_sum = train_mask + val_mask + test_mask
        assert (mask_sum.max() == 1) and (mask_sum.min() == 1), "Got overlapping / missing values!"
