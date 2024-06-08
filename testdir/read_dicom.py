#!/usr/bin/env python3

import pydicom
import numpy as np

# Read the DICOM file
dcm_file = '../d004/7T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0004.0001.2023.12.08.07.38.12.980291.186281054.IMA' 
dicom = pydicom.dcmread(dcm_file)
tmp = dicom.pixel_array

# Print a portion of the pixel values (e.g., top-left 10x10 region)
print('Python Pixel Values:')
print(tmp[:10, :10])
# Save the 100x100 region to a text file
np.savetxt('python_pixels.txt', tmp[:100, :100], fmt='%d')
