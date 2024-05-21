#!/usr/bin/env python3

import os
import numpy as np
import pydicom
from scipy import ndimage
from scipy.stats import norm
from scipy.stats import linregress

DEBUG=0

def main():
    data_path = './data/'
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
    print(mtxsz)
    
    dinfo = pydicom.dcmread(os.path.join(data_path, fnames[0]))
    ncol = dinfo.Columns / sqrtnsl
    nrow = dinfo.Rows / sqrtnsl
    mm = min(ncol, nrow)

    tmp_mat = np.zeros((mtxsz, mtxsz, nsl, nn))
    if DEBUG:
        print("mtxsz, mtxsz, nsl, nn="+str(mtxsz)+" "+str(mtxsz)+" "+str(nsl)+" "+str(nn))

    for ii in range(nn):
        tmp = pydicom.dcmread(os.path.join(data_path, fnames[ii])).pixel_array
        tmp2 = []
        cind = 0
        while cind < tmp.shape[0]:
            tmp2.append(tmp[cind:cind+mtxsz, :])
            cind += mtxsz
        tmp2 = np.concatenate(tmp2, axis=0)
        tmp_mat[:, :, :, ii] = tmp2.reshape((mtxsz, mtxsz, nsl))
        if DEBUG and ii==nn-1:
            print(tmp_mat[0,:,:,ii])

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
        #ya = np.polyval(p, x)
        ya = p[0] * (x**2) + p[1] * x

        for ii in range(nn):
            tmp_detrend[:, :, ns, ii] = tmp_mat[:, :, ns, ii] - ya[ii]

        # Debugging polynomial fitting
        if DEBUG:
            print(f'Polynomial coefficients for slice {ns}: {p}')
            print(f'Detrended values for slice {ns}: {tmp_detrend[:, :, ns, :]}')
        """
        tmp = msk_cm2[:, :, ns]
        A = tmp.flatten().astype(bool)
        for ii in range(nn):
            B = tmp_mat[:, :, ns, ii].flatten()
            C = B[A]
            time_N[ns, ii] = np.mean(C)

        x = np.arange(1, nn + 1)
        y = time_N[ns, :]
        p = np.polyfit(x, y, 2)
        ya = p[0] * (x**2) + p[1] * x

        for ii in range(nn):
            tmp_detrend[:, :, ns, ii] = tmp_mat[:, :, ns, ii] - ya[ii]
        """
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

    xt = np.arange(1, nn + 1)
    yt = time_N
    p = np.polyfit(xt, yt, 2)
    yp = p[0] * (xt**2) + p[1] * xt + p[2]
    ya = p[0] * (xt**2) + p[1] * xt

    spk_thrsh = 0.5
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
        print(f'{yeartxt}\t{monthtxt}\t{daytxt}\t{timetxt}\t{runtxt}\t{nn}\t{Smean:.1f}\t{Spk:.2f}\t{Srms:.2f}\t{SFNRi:.1f}\t{Rqq:.3f}\t{Wrdc:.3f}\t{SPK}\n')

    print('done')

if __name__ == '__main__':
    main()

