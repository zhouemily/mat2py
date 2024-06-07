# mat2py project: convert matlab script to python script for dicom analysis

read_dicom.m and read_dicom.py:
By reading the same DICOM file in both MATLAB and Python and printing a portion of the pixel values, 
you can verify that the data is being read consistently. The printed pixel values from MATLAB and Python should match, 
confirming that both environments correctly interpret the DICOM file. If there are discrepancies, 
it could indicate differences in how the libraries handle the file, which might require further investigation.
