# Mat2Py Project: Convert MATLAB Script to Python Script for DICOM Analysis

## Overview

The Mat2Py project aims to convert MATLAB scripts to Python scripts for DICOM analysis. This project is designed to help users transition their DICOM image processing workflows from MATLAB to Python, taking advantage of Python's extensive libraries and ease of integration with other data processing tools.

## Features

- Convert MATLAB DICOM analysis scripts to Python
- Perform signal processing and statistical analysis on DICOM images
- Apply noise reduction techniques such as median and Gaussian filtering
- Analyze and compare results to ensure accuracy

## Requirements

- Python 3.x
- NumPy
- SciPy
- PyDicom
- Matplotlib (optional, for visualizing results)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/mat2py.git
   cd mat2py

read_dicom.m and read_dicom.py:
By reading the same DICOM file in both MATLAB and Python and printing a portion of the pixel values, 
you can verify that the data is being read consistently. The printed pixel values from MATLAB and Python should match, 
confirming that both environments correctly interpret the DICOM file. If there are discrepancies, 
it could indicate differences in how the libraries handle the file, which might require further investigation.

Test output notes:
bash-5.1$ python3 diff_matlab_python.py
The files are identical (ignoring spaces and newlines).
(both matlab and python script read dicom file correctly: same results)
