import nibabel as nib
from numpy import zeros, eye
from utils.image import Image
import transformations.svf as SVF
from utils.helper import *
from utils.helper import generate_identity_deformation

data = zeros([128, 128, 128])
im = Image(nib.Nifti1Image(data, eye(4)))
print data.size

mysvf = SVF.SVF()
mysvf.init_field(im)
print mysvf.field.data

def_im = initialise_field(im)
generate_identity_deformation(def_im)
print def_im.data

mysvf.exponentiate(def_im)