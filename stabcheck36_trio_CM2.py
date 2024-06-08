#!/usr/bin/env python3

import os
import numpy as np
import pydicom
from scipy import ndimage
from scipy.stats import norm
from scipy.stats import linregress
import glob
import sys
from datetime import datetime
import warnings
from numpy import RankWarning
import argparse
import string
import datetime

DEBUG=0
dirpath=''
def debug_print(msg):
    # Get the current line number
    line_number = sys._getframe().f_lineno
    print(f"Debug [{line_number}]: {msg}")

parser = argparse.ArgumentParser(description="This script processes DICOM images in a  directory")
parser.add_argument('-d', '--dir', type=str, required=False,
                     help="Specify the path to the directory to process.")
parser.add_argument('-v', '--verbose', action='store_true',
                     help="Enable verbose mode for detailed output of the program's execution.")
parser.add_argument('-D', '--debug', type=int, choices=[0, 1, 2, 3], default=0,
                    help="Set the debug level: 0=none, 1=errors, 2=warnings, 3=info.")

args = parser.parse_args()
if args.debug == 0:
    print("Starting script...")
elif args.debug == 1:
    print("Showing only error messages.")
elif args.debug == 2:
    print("Showing warnings and error messages.")
elif args.debug == 3:
    print("Showing informational, warning, and error messages.")
    DEBUG=1

if args.dir:
    dirpath=args.dir+'/'
    if not os.path.isdir(dirpath):
        print(f"Error: The directory does not exist: "+args.dir)
        sys.exit(1)
else:
    #dirpath="/data/"  #this is for negative dirpath testing only, change it if needed
    dirpath="./data/"
    if not os.path.isdir(dirpath):
        print(f"Error: The default directory does not exist, please check the dirpath given")
        sys.exit(1)

#######################################################################################################

def main():
    data_path = dirpath 
    log_path = './logdir'
    fnames = [f for f in os.listdir(data_path) if f.endswith('.IMA')]
    nn = len(fnames)

    if nn == 0:
        print('In stabcheck36.py')
        print('No dicom images found or already done')
        output = 0
        return

    nsl = 36
    sqrtnsl = np.sqrt(nsl)
    mtxsz = 64
    if args.debug > 0:
        print("nsl="+str(nsl)+ " sqrtnsl="+str(sqrtnsl)+" mtxsz="+str(mtxsz))
    
    dinfo = pydicom.dcmread(os.path.join(data_path, fnames[0]))
    ncol = dinfo.Columns / sqrtnsl
    nrow = dinfo.Rows / sqrtnsl
    mm = min(ncol, nrow)

    tmp_mat = np.zeros((mtxsz, mtxsz, nsl, nn))
    if args.debug >= 1:
        print("ncol, nrow, mm ::"+str(ncol)+","+str(nrow)+","+str(mm))
        print("tmp_mat(mtxsz, mtxsz, nsl, nn) :: "+str(mtxsz)+" "+str(mtxsz)+" "+str(nsl)+" "+str(nn))

    for ii in range(nn):
        # Read the DICOM file and extract the pixel data
        try:
            dcm_path = os.path.join(data_path, fnames[ii])
            dicom = pydicom.dcmread(dcm_path)
            tmp = dicom.pixel_array
            if ii == 0 and args.debug > 0:
                print('Python Pixel Values for first dicom file fname[0] saved as python_pixels0.txt:')
                np.savetxt('python_pixels0.txt', tmp[:100, :100], fmt='%d')
        except Exception as e:
            print(f"Error reading DICOM file {fnames[ii]}: {e}")
            continue

        tmp2 = []
        cind = 0

        # Split image data into chunks of size `mtxsz` rows
        while cind < tmp.shape[0]:
            chunk = tmp[cind:cind + mtxsz, :]
            tmp2.append(chunk)
            cind += mtxsz

        # Concatenate chunks vertically to form a single array
        try:
            tmp2 = np.concatenate(tmp2, axis=0)
        except ValueError as e:
            print(f"Error concatenating chunks for file {fnames[ii]}: {e}")
            continue

        # Check if the reshaped dimensions match the expected shape
        if tmp2.size != mtxsz * mtxsz * nsl:
            print(f"Unexpected shape after reshaping for file {fnames[ii]}.")
            continue

        # Reshape and store the data in the 4D matrix
        try:
            tmp_mat[:, :, :, ii] = tmp2.reshape((mtxsz, mtxsz, nsl))
        except ValueError as e:
            print(f"Error reshaping or storing data for file {fnames[ii]}: {e}")
            continue

        # Debugging output
        if DEBUG and ii == nn - 1:
           print(tmp_mat[0, :, :, ii])

    cmxs = np.zeros(nsl)
    cmys = np.zeros(nsl)
    for gnsl in range(nsl):
        cmx = 0
        cmy = 0
        mtot = 0
        for gi in range(mtxsz):
            for gj in range(mtxsz):
                cmx += tmp_mat[gi, gj, gnsl, 0] * gi
                cmy += tmp_mat[gi, gj, gnsl, 0] * gj
                mtot += tmp_mat[gi, gj, gnsl, 0]
        cmxs[gnsl] = cmx / mtot
        cmys[gnsl] = cmy / mtot
        if DEBUG:
            print("CM:")
            print(cmxs[gnsl])
            print(cmys[gnsl])
            print(mtot)

        vx2 = np.arange(ncol)
        vy2 = np.arange(nrow)
        xx2, yy2 = np.meshgrid(vy2, vx2)
        msk_cm = np.sqrt((xx2 - cmys[gnsl])**2 + (yy2 - cmxs[gnsl])**2) < (mm / 4.2)
        if gnsl == 0:
            msk_cm2 = np.repeat(msk_cm[:, :, np.newaxis], nsl, axis=2)
        else:
            msk_cm2[:, :, gnsl] = msk_cm

    tmp_mat = tmp_mat.astype(np.float64)

    ROIvar = np.zeros((nsl, nn - 1))
    time_N = np.zeros((nsl, nn))
    tmp_detrend = np.zeros_like(tmp_mat)

    from numpy.polynomial.polynomial import Polynomial
    # DE-TRENDING
    for ns in range(nsl):
        tmp = msk_cm2[:, :, ns]
        A = tmp.flatten().astype(bool)
        for ii in range(nn):
            tmp = tmp_mat[:, :, ns, ii]
            B = tmp.flatten()
            C = B[A]
            time_N[ns, ii] = np.mean(C)

        x = np.arange(1, nn + 1)
        y = time_N[ns, :]
        p = np.polyfit(x, y, 2)
        #ya = np.polyval(p, x)  ##note:: this np.polyval function does not work well here
        ya = p[0] * (x**2) + p[1] * x   ##note:: direct tranlate from matlab works well here

        for ii in range(nn):
            tmp_detrend[:, :, ns, ii] = tmp_mat[:, :, ns, ii] - ya[ii]

        # Debugging polynomial fitting
        if DEBUG:
            print(f'Polynomial coefficients for slice {ns}: {p}')
            print(f'Detrended values for slice {ns}: {tmp_detrend[:, :, ns, :]}')

        for ii in range(nn - 1):
            tmp = (tmp_detrend[:, :, ns, ii] - tmp_detrend[:, :, ns, ii + 1]) / np.sqrt(2)
            B = tmp.flatten()
            C = B[A]
            ROIstd = np.std(C)
            tmp = (tmp_detrend[:, :, ns, ii] + tmp_detrend[:, :, ns, ii + 1]) / 2
            B = tmp.flatten()
            C = B[A]
            ROImean = np.mean(C)
            ROIvar[ns, ii] = 100 * (ROIstd / ROImean)

    ss = 18  # all other metrics just over the center slice
    time_N = time_N[ss, :]
    tmp_detrend = tmp_detrend[:, :, ss, :]

    xt = np.arange(1, nn + 1) ##or range from 0 to nn (number of dicome files)
    yt = time_N
    p = np.polyfit(xt, yt, 2)
    yp = p[0] * (xt**2) + p[1] * xt     ##note:: better correlation achived
    #yp = p[0] * (xt**2) + p[1] * xt + p[2]  ## direct translation from Matlab
    ya = p[0] * (xt**2) + p[1] * xt ##note:: ya is not used later

    #spk_thrsh = 0.5 #matlab value 
    #spk_thrsh = 0.8 
    spk_thrsh = 1.0  ##note:: adjust the spk_thrsh to set proper sensitivity 
    A = ROIvar[ROIvar > spk_thrsh]
    SPK = len(A.flatten())

    Smean = np.mean(time_N)
    Smax = np.max(time_N)
    Smin = np.min(time_N)
    Spk = 100 * (Smax - Smin) / Smean
    Sres = time_N - Smean
    Srms = 100 * np.sqrt(np.sum(Sres**2)) / Smean

    Simg = np.mean(tmp_detrend, axis=2)
    Nimg = np.std(tmp_detrend, axis=2)
    SFNRimg = Simg / Nimg

    tmp = msk_cm2[:, :, ss]
    A = tmp.flatten().astype(bool)

    tmp = SFNRimg
    B = tmp.flatten()
    C = B[A]
    SFNRi = np.mean(C)
    tmp = Simg
    B = tmp.flatten()
    C = B[A]
    Si = np.mean(C)
    tmp = Nimg
    B = tmp.flatten()
    C = B[A]
    Ni = np.mean(C)

    x = yt - yp
    x = x.flatten()
    y_sorted = np.sort(x)
    n = len(y_sorted)
    pp = (np.arange(1, n + 1) - 0.5) / n
    xx = norm.ppf(pp)
    p, s = linregress(xx, y_sorted)[:2]
    Rqq = np.sqrt(1 - np.sum((y_sorted - p * xx - s)**2) / np.sum(y_sorted**2))

    W = np.zeros((1, 21))
    for sqsz in range(1, 22):
        sqhf = (sqsz - 1) / 2
        x1 = int(cmxs[ss] - sqhf)
        x2 = int(cmxs[ss] + sqhf)
        y1 = int(cmys[ss] - sqhf)
        y2 = int(cmys[ss] + sqhf)

        sqmsk = np.zeros_like(tmp_detrend[:, :, 0])
        sqmsk[x1:x2+1, y1:y2+1] = 1
        A = sqmsk.flatten().astype(bool)

        imn = np.zeros((nn, 1))
        for tt in range(nn):
            img = tmp_detrend[:, :, tt]
            B = img.flatten()
            C = B[A]
            imn[tt] = np.mean(C)

        Wmn = np.mean(imn)
        Wsd = np.std(imn)
        W[0, sqsz - 1] = 100 * (Wsd / Wmn)

    Wrdc = W[0, 0] / W[0, 20]

    fname_res = f'./logdir/results_{dinfo.InstanceCreationDate[:10]}'.replace(':', '_')
    ppath = os.getcwd()
    fname_res_cur = f'{fname_res}{ppath[-1]}.txt'

    with open(fname_res_cur, 'w') as fid:
        fid.write(f'{dinfo.InstanceCreationDate}\n')
        fid.write('Stability Run, 36 slice \n\n')
        fid.write(f'Slice {ss} \t Mean = {Smean:.1f} \t Peak-to-peak % = {Spk:.2f} \t RMS % = {Srms:.3f} \t SNR = {SFNRi:.3f} \t Rqq = {Rqq:.3f} \t RDC = {Wrdc:.3f} \t Spike Count = {SPK}\n')
        fid.write('Test line\n')
        print(f'Slice {ss} \t Mean = {Smean:.1f} \t Peak-to-peak % = {Spk:.2f} \t RMS % = {Srms:.3f} \t SNR = {SFNRi:.3f} \t Rqq = {Rqq:.3f} \t RDC = {Wrdc:.3f} \t Spike Count = {SPK}\n')

    with open('./logdir/stability_history_2013.txt', 'a') as fid:
        yeartxt = dinfo.InstanceCreationDate[8:12]
        monthtxt = dinfo.InstanceCreationDate[4:7]
        daytxt = dinfo.InstanceCreationDate[1:3]
        timetxt = dinfo.InstanceCreationTime[:8]
        runtxt = ppath[-1]
        fid.write(f'{yeartxt}\t{monthtxt}\t{daytxt}\t{timetxt}\t{runtxt}\t{nn}\t{Smean:.1f}\t{Spk:.2f}\t{Srms:.2f}\t{SFNRi:.1f}\t{Rqq:.3f}\t{Wrdc:.3f}\t{SPK}\n')
        print("DICOM InstanceCreationDate and InstanceCreationTime :: number of files :: Smean :: Spk :: Srms :: SFNRi :: Rqq :: WRDC :: SPK")
        print(f'{yeartxt}\t{monthtxt}\t{daytxt}\t{timetxt}\t{runtxt}\t{nn}\t{Smean:.1f}\t{Spk:.2f}\t{Srms:.2f}\t{SFNRi:.1f}\t{Rqq:.3f}\t{Wrdc:.3f}\t{SPK}\n')

    print('done')

if __name__ == '__main__':
    main()
