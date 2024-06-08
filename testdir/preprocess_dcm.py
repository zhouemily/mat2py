
import pydicom
import numpy as np
import cv2
import pywt
from PIL import Image
import matplotlib.pyplot as plt

# Read the DICOM file
dcm_file = 'd004/7T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0004.0256.2023.12.08.07.38.12.980291.186296403.IMA'  # Replace with your DICOM file path
dicom = pydicom.dcmread(dcm_file)

# Extract the image data from the DICOM file
image_data = dicom.pixel_array

# Normalize the image data to the range 0-255
image_data = (np.maximum(image_data, 0) / image_data.max()) * 255.0
image_data = np.uint8(image_data)

# Apply median filter to remove noise and spikes
median_filtered = cv2.medianBlur(image_data, 5)

# Optionally apply Gaussian filter to smooth the image further
gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)

# Apply wavelet denoising
coeffs = pywt.wavedec2(gaussian_filtered, 'db1', level=2)

# Threshold detail coefficients
thresholded_coeffs = list(coeffs)
thresholded_coeffs[1:] = [(pywt.threshold(c[0], value=10, mode='soft'),
                           pywt.threshold(c[1], value=10, mode='soft'),
                           pywt.threshold(c[2], value=10, mode='soft')) for c in coeffs[1:]]

# Reconstruct the image from the thresholded coefficients
wavelet_denoised = pywt.waverec2(thresholded_coeffs, 'db1')
wavelet_denoised = np.uint8(wavelet_denoised)

# Display the original and processed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_data, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Processed Image")
plt.imshow(wavelet_denoised, cmap='gray')
plt.show()

# Save the processed image as a PNG file
output_image = Image.fromarray(wavelet_denoised)
output_image.save('processed_image.png')

