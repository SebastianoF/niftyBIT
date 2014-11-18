import nibabel as nib
"""
from utils.image import Image
import transformations.svf as SVF
from utils.helper import *
from utils.helper import generate_identity_deformation
"""
import numpy as np
import scipy.ndimage.filters as fil
from src.utils.helper import generate_random_smooth_deformation, compute_spatial_gradient

'''
Variable names and concepts:

Velocity field [vf]: vf(d,x,y,z), d = 0,1,2 index with coordinates x,y,z = 0,...dim_x,y,z.
   vf(x,y,z) = (vf_1(x,y,z), vf_2(x,y,z), vf_3(x,y,z)) = vf_d(x,y,z)    d = 0,1,2
Smooth velocity field [smooth_vf]: vector field after gaussian smoothing
Nifti smooth velocity field [img]: nifti image out of the smooth vector field.
'''

# generate random smooth deformation:


def random_smooth_vector_field(dim_x=128, dim_y=128, dim_z=128):
    """
    Generate a smooth random 3d vector field on a 3d grid.

    :param dim_x,dim_y,dim_z: dimension of the vector field
    :return: random smooth vector field ready to be initialized as a SVF object
    :rtype : np.array
    """
    vf = np.ones((3, dim_x, dim_y, dim_z), dtype=np.float64)
    for i in range(3):
        vf[i] = np.random.normal(0, 1, (dim_x, dim_y, dim_z))
    smooth_vf = fil.gaussian_filter(vf, 2)
    return smooth_vf


def smooth_vf2nibabel(smooth_vf, a=np.eye(4)):
    """
    From a smooth vector field to a nibabel image
    :param smooth_vf: smooth vector field
    :param a: homogenous affine giving relationship between voxel coordinates and world coordinates.
    :return: the nifty type image
    :rtype: nifty image
    """
    img = nib.Nifti1Image(smooth_vf, a)
    return img


def gradient_smooth_vector_field(smooth_vf):
    """
    Gradient of a smooth vector field

    :param smooth_vf: Smooth vector field
    :return: the gradient of the field
    """
    grad = np.zeros(3, dtype=np.float64)
    for i in range(3):
        grad[i] = np.array(np.gradient(smooth_vf[i]))[i]
    return grad


rsd = generate_random_smooth_deformation((128, 128, 128))
grad_rsd = compute_spatial_gradient(rsd)


# from nibabel nifti type to vector field to SVF



# BCH order 0


# BCH order 1