__author__ = 'Pankaj Daga'

from scipy import ndimage

from image import *
from helper import generate_identity_deformation, \
                    generate_displacement_from_deformation, \
                    generate_position_from_displacement


class ImageResampler(object):
    """
    Class to resample image according to a deformation field
    """
    def __init__(self, order=1, prefilter=False):
        """
        Linear interpolation by default.
        """
        self.order = order
        self.prefilter = prefilter

    def resample(self, source, deformation, warped):
        """
        Resample the source image in the space of target.

        Parameters:
        :param source: The source image.
        :param deformation: The deformation field
        :param warped: Warped image. Should be allocated in the space of target
        and must have the same number of time points as the source image.
        """

        assert source.time_points == warped.time_points
        # Matrix to go from world to voxel space
        ijk_2_voxel = source.mm_2_voxel
        vol_ext = warped.vol_ext

        def_data = [deformation.data[..., i].reshape(vol_ext, order='F')
                    for i in range(deformation.data.shape[-1])]

        # Create the sampling grid in which the source image will be evaluated.
        def_data = [ijk_2_voxel[i][3] + sum(ijk_2_voxel[i][k] * def_data[k]
                    for k in range(len(def_data)))
                    for i in range(len(vol_ext))]

        if source.time_points == 1:
            ndimage.map_coordinates(source.data, np.asarray(def_data),
                                    warped.data, order=self.order,
                                    prefilter=self.prefilter)
        else:
            for i in range(source.time_points):
                ndimage.map_coordinates(source.data[i], def_data,
                                        warped.data[i], order=self.order,
                                        prefilter=self.prefilter)


class NearestNeighbourResampler(ImageResampler):
    """
    Set to nearest neighbourhood resampling
    """
    def __init__(self):
        super(NearestNeighbourResampler, self).__init__(order=0)


class CubicSplineResampler(ImageResampler):
    """
    Set to cubic spline resampling
    """
    def __init__(self):
        super(CubicSplineResampler, self).__init__(order=2, prefilter=True)


class PositionFieldComposer(object):
    """
    Compose position fields using linear interpolation.
    """
    def __init__(self):
        self.order = 1

    def set_order(self, order):
        if order > 0:
            self.order = order
        else:
            self.order = order

    def compose(self, left, right):
        """
        Compose position fields.
        Parameters:
        -----------
        :param left: Outer field.
        :param right: Inner field
        Order of composition: left(right(x))
        :return The composed position field
        """

        d = np.zeros(left.data.shape)
        result = Image.from_data(d, header=left.get_header())

        vol_ext = left.vol_ext[:left.data.shape[-1]]
        left_data = [left.data[..., i].reshape(vol_ext, order='F')
                     for i in range(left.data.shape[-1])]

        data = np.squeeze(result.data)

        for i in range(data.shape[-1]):
            # todo: Incorrect behaviour at boundary
            # This is an issue we need to revisit.
            # Nearest mode is not good when we are dealing with fields.
            # Unfortunately map_coordinates does not allow for leaving the value
            # as is and we will need to implement this at some point.
            ndimage.map_coordinates(np.squeeze(right.data[..., i]),
                                    left_data,
                                    data[..., i],
                                    mode='nearest',
                                    order=self.order, prefilter=True)

        return result


class DisplacementFieldComposer(object):
    """
    Compose displacement fields using linear interpolation.
    """
    def __init__(self):
        self.order = 1

    @staticmethod
    def compose(left, right):
        """
        Compose displacement fields.
        Parameters:
        -----------
        :param left: Outer displacement field.
        :param right: Inner displacement field.
        :param result: Resulting position field after composition.
        If it is none, it will be allocated
        Order of composition: left(right(x))
        """
        left_def_image = generate_position_from_displacement(left)
        right_def_image = generate_position_from_displacement(right)

        composer = PositionFieldComposer()
        result = composer.compose(left_def_image, right_def_image)
        result = generate_displacement_from_deformation(result)
        return result





