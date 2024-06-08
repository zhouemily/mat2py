# mat2py project: convert matlab script to python script for dicom analysis

read_dicom.m and read_dicom.py:
By reading the same DICOM file in both MATLAB and Python and printing a portion of the pixel values, 
you can verify that the data is being read consistently. The printed pixel values from MATLAB and Python should match, 
confirming that both environments correctly interpret the DICOM file. If there are discrepancies, 
it could indicate differences in how the libraries handle the file, which might require further investigation.

Test output notes:
bash-5.1$ python3 diff_matlab_python.py
The files are identical (ignoring spaces and newlines).
(both matlab and python script read dicom file correctly: same results)
