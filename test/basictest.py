import nibabel as nib
import numpy as np
from utils import image
import transformations.svf as SVF
from utils.helper import *

data = np.zeros([128, 128, 128])
im = image.Image(nib.Nifti1Image(data, np.eye(4)))
# print im.data.size

mysvf=SVF.SVF()
mysvf.init_field(im)
# print mysvf.field.data

def_im = initialise_field(im)
generate_identity_deformation(def_im)
#print def_im.data

mysvf.exponentiate(def_im)