import numpy as np

from solarnet.datasets.transforms import no_change, horizontal_flip, vertical_flip


class TestTransforms:

    def test_mask_consistency(self):
        """Tests that the masks and images are transformed
        in the same way
        """
        transforms = [
            no_change, horizontal_flip, vertical_flip
        ]

        test_im = np.zeros((3, 224, 224))
        test_mask = np.zeros((224, 224))

        seg_x, seg_y = (10, 20), (100, 130)
        test_mask[seg_x[0]: seg_x[1], seg_y[0]: seg_y[1]] = 1
        for channel in range(test_im.shape[0]):
            test_im[channel] = test_mask

        for transform in transforms:
            transformed_im, transformed_m = transform(test_im, test_mask)

            for t_channel in transformed_im:
                assert (t_channel == transformed_m).all(), \
                    'Mask not consistent with image after transform!'
