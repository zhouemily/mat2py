% Read the DICOM file
fname = 'd004/7T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0004.0256.2023.12.08.07.38.12.980291.186296403.IMA';  % Replace with DICOM file path
tmp = dicomread(fname);

% Print a portion of the pixel values (e.g., top-left 10x10 region)
disp('MATLAB Pixel Values:');
disp(tmp(1:10, 1:10));

% Save the 100x100 region to a text file
fid = fopen('matlab_pixels.txt', 'w');
fprintf(fid, '%d ', tmp(1:100, 1:100)');
fclose(fid);
