from solarnet.preprocessing.splits import ImageSplitter


class TestImageSplitter:

    # rasterio requires a driver to write .tif files,
    # preventing a set up to test the entire ImageSplitter
    # class.

    def test_adjust_coords(self):
        # tests that the centroid coordinates are
        # properly adjusted if they will lead to an
        # out of bounds image

        coords = (99, 1)  # the location of the centroid
        imrad = 10  # The size of the split image
        org_imsize = (100, 100)  # The size of the original image

        new_coords = ImageSplitter.adjust_coords(coords, imrad, org_imsize)

        assert new_coords[0] == 90, "x coordinate improperly adjusted"
        assert new_coords[1] == 10, "y coordinate improperly adjusted"
