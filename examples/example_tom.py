# Import modules
from src.utils.helper import *
from src.utils.resampler import *
import nibabel as nib
import src.transformations.svf as SVF
import scipy.ndimage.filters
import matplotlib.pyplot as plt
import copy
import numpy as np
import image
reload(image)

# Generate an image mostly to get a basic structure
#data = np.zeros((128, 128, 128))
#im = Image(nib.Nifti1Image(data,np.eye(4)))
data = np.zeros((128, 128))
im = image.Image(nib.Nifti1Image(data,np.eye(4)))
print 'im shape ' + str(im.data.shape)

# Generate a first smooth velocity field (should switch to Pankaj's code at some point)
vel_im0 = SVF.SVF()
vel_im0.init_field(im)
print 'vel im shape ' + str(vel_im0.field.data.shape)
vel_im0.field.data = 10*np.random.randn(vel_im0.field.data.shape[0], vel_im0.field.data.shape[1], vel_im0.field.data.shape[2], vel_im0.field.data.shape[3], vel_im0.field.data.shape[4])
#vel_im0.field.data = 10*np.ones(vel_im0.field.data.shape)
for i in range(0, vel_im0.field.data.shape[4]):
    vel_im0.field.data[:,:,:,0,i] = scipy.ndimage.filters.gaussian_filter(vel_im0.field.data[:,:,:,0,i],2)

# Generate a second smooth velocity field (should switch to Pankaj's code at some point)
vel_im1 = SVF.SVF()
vel_im1.init_field(im)
vel_im1.field.data = 3*np.random.randn(vel_im1.field.data.shape[0], vel_im1.field.data.shape[1], vel_im1.field.data.shape[2], vel_im1.field.data.shape[3], vel_im1.field.data.shape[4])
for i in range(0, vel_im1.field.data.shape[4]):
    vel_im1.field.data[:,:,:,0,i] = scipy.ndimage.filters.gaussian_filter(vel_im1.field.data[:,:,:,0,i],2)

# Display the first velocity field for sanity check
plt.figure()
plt.title('vel0')
plt.imshow(np.squeeze(vel_im0.field.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
print 'min vel0 ' + str(np.min(vel_im0.field.data))
print 'max vel0 ' + str(np.max(vel_im0.field.data))
print 'median vel0 ' + str(np.median(vel_im0.field.data))
plt.colorbar()
plt.show(block=False)

plt.figure()
plt.title('vel0 Y')
plt.imshow(np.squeeze(vel_im0.field.data[:,:,0,0,1]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

# Compute the first displacement field
def_im0 = vel_im0.exponentiate()
print 'def im shape ' + str(def_im0.data.shape)

#exit()

# Compute the second displacement field
def_im1 = vel_im1.exponentiate()

# Display the first displacement field for sanity check
plt.figure()
plt.title('def0')
plt.imshow(np.squeeze(def_im0.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
print 'min def0 ' + str(np.min(def_im0.data))
print 'max def0 ' + str(np.max(def_im0.data))
print 'median def0 ' + str(np.median(def_im0.data))
plt.colorbar()
plt.show(block=False)

# Display the second displacement field for sanity check
plt.figure()
plt.title('def1')
plt.imshow(np.squeeze(def_im1.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
print 'min def1 ' + str(np.min(def_im1.data))
print 'max def1 ' + str(np.max(def_im1.data))
print 'median def1 ' + str(np.median(def_im1.data))
plt.colorbar()
plt.show(block=False)

# Compute the composition
dfc = DisplacementFieldComposer()
def_gt = dfc.compose(def_im0, def_im1)

# Display the composed displacement field
plt.figure()
plt.title('defgt')
plt.imshow(np.squeeze(def_gt.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

# Compute error with no update at all
def_err_noup = copy.deepcopy(def_im0)
def_err_noup.data -= def_gt.data
# Compute average error on a central region to avoid taking into account border effects problem
err2_noup = np.linalg.norm(def_err_noup.data[32:96,32:96,:,:,:])
print 'No update err: ' + str(np.sqrt(err2_noup))

# Compute BCH0 update (u+v)
vel_bch0 = copy.deepcopy(vel_im0)
vel_bch0.field.data += vel_im1.field.data
def_bch0 = vel_bch0.exponentiate()

# Display diplacement field from BCH0 result
plt.figure()
plt.title('defbch0')
plt.imshow(np.squeeze(def_bch0.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

# Compute error with BCH0
def_err_bch0 = copy.deepcopy(def_bch0)
def_err_bch0.data -= def_gt.data
# Compute average error on a central region to avoid taking into account border effects problem
err2_bch0 = np.linalg.norm(def_err_bch0.data[32:96,32:96,:,:,:])
print 'BCH0 err: ' + str(np.sqrt(err2_bch0))

# Compute parrallel transport update
tmp_vel = copy.deepcopy(vel_im0)
tmp_vel.field.data /= 2
tmp_defa = tmp_vel.exponentiate()
tmp_vel.field.data = -tmp_vel.field.data
tmp_defb = tmp_vel.exponentiate()
tmp_defc = dfc.compose( dfc.compose(tmp_defa, def_im1), tmp_defb)

vel_pt = copy.deepcopy(vel_im0)
vel_pt.field.data += tmp_defc.data
def_pt = vel_pt.exponentiate()

# Display some intermediate results for the parallel transport
## plt.figure()
## plt.title('defa')
## plt.imshow(np.squeeze(tmp_defa.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
## plt.colorbar()
## plt.show(block=False)

## plt.figure()
## plt.title('defb')
## plt.imshow(np.squeeze(tmp_defb.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
## plt.colorbar()
## plt.show(block=False)

## plt.figure()
## plt.title('defc')
## plt.imshow(np.squeeze(tmp_defc.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
## plt.colorbar()
## plt.show(block=False)

# Display displacement field computed with parallel transport
plt.figure()
plt.title('defpt')
plt.imshow(np.squeeze(def_pt.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

# Compute error with parallel transport
def_err_pt = copy.deepcopy(def_pt)
def_err_pt.data -= def_gt.data
# Compute average error on a central region to avoid taking into account border effects problem
err2_pt = np.linalg.norm(def_err_pt.data[32:96,32:96,:,:,:])
print 'PT err: ' + str(np.sqrt(err2_pt))

# Display displacement error field with BCH
plt.figure()
plt.title('err bch0')
plt.imshow(np.squeeze(def_err_bch0.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

plt.figure()
plt.title('err bch0 Y')
plt.imshow(np.squeeze(def_err_bch0.data[:,:,0,0,1]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

# Display displacement error field with parallel transport
plt.figure()
plt.title('err pt')
plt.imshow(np.squeeze(def_err_pt.data[:,:,0,0,0]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=False)

plt.figure()
plt.title('err pt Y')
plt.imshow(np.squeeze(def_err_pt.data[:,:,0,0,1]), cmap = plt.get_cmap('gray'), interpolation='nearest')
plt.colorbar()
plt.show(block=True)
