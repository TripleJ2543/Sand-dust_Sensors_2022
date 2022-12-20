#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:37:04 2022

@author: ispl-public
"""

from skimage import morphology
import numpy as np
import cv2

import time
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_dir', dest='InputPath', default='/home/ispl-public/Desktop/Dataset/sand-dust_imageset', help='directory for testing inputs')
parser.add_argument('--output_dir', dest='OutputPath', default='./', help='directory for testing outputs')
parser.add_argument('--gamma_max', dest='gamma_max', type=float, default=2.0, help='gamma_max value')
args = parser.parse_args()


## pixel value: i2f: 0-255 to 0-1, f2i: 0-1 to 0-255
def i2f(i_image):
    f_image = np.float32(i_image)/255.0
    return f_image

def f2i(f_image):
    i_image = np.uint8(f_image*255.0)
    return i_image

def GetIntensity(fi):
    return cv2.divide(fi[:,:,0] + fi[:,:,1] + fi[:,:,2], 3)

def GetSaturation(fi):
    intensity = cv2.max(cv2.max( fi[:,:,0], fi[:,:,1]),fi[:,:,2])
    min_rgb = cv2.min(cv2.min( fi[:,:,0], fi[:,:,1]),fi[:,:,2])
    me = np.finfo(np.float32).eps
    S = 1.0 - min_rgb/(intensity+me)
    return S 

def GetSaturationI(fi):
    intensity = cv2.divide(fi[:,:,0] + fi[:,:,1] + fi[:,:,2], 3)
    min_rgb = cv2.min(cv2.min( fi[:,:,0], fi[:,:,1]),fi[:,:,2])
    me = np.finfo(np.float32).eps
    S = 1.0 - min_rgb/(intensity+me)
    return S 

def GetMax(fi):
    return cv2.max(cv2.max(fi[:,:,0],fi[:,:,1]), fi[:,:,2])

def GetMin(fi):
    return cv2.min(cv2.min(fi[:,:,0],fi[:,:,1]), fi[:,:,2])



## Compute 'A' as described by Tang et al. (CVPR 2014)
def A_Tang(im):
    erosion_window = 15
    n_bins = 200

    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]
    
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))

    [h, edges] = np.histogram(dark, n_bins)
    numpixel = im.shape[0]*im.shape[1]
    thr_frac = numpixel*0.99
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr

    rs = R[mask]
    gs = G[mask]
    bs = B[mask]
    
    A = np.zeros((1,3))

    A[0, 2] = np.median(rs)
    A[0, 1] = np.median(gs)
    A[0, 0] = np.median(bs)

    return A


def Norm(im):
    aim = np.empty(im.shape,im.dtype) 

    for ind in range(0,3):
        im_h = np.max(im[:,:,ind])
        im_l = np.min(im[:,:,ind])
        aim[:, :, ind] = (im[:, :, ind]-im_l)/(im_h-im_l)
        aim[:,:,ind] = np.clip(aim[:,:,ind], 0.0, 1.0)
        
    return np.clip(aim, 0.0, 1.0)


## ColorCorrection
def ColorCorrectionFirst(im):
    res1 = SDCorrection(im)
    res2 = GreenMeanPreservingNormalization(res1)
    res3 = Stretch(res2)   
    return np.clip(res3, 0.0, 1.0)


def ColorCorrectionLast(im):
    res1 = GreenMeanPreservingNormalization(im)
    res2 = MeanCorrection(res1)   
    return np.clip(res2, 0.0, 1.0)


def SDCorrection(im):
    res = np.empty(im.shape,im.dtype) 
    sd = np.zeros((1,3))
    alpha = np.zeros((1,3))
    
    i_sum = GetIntensity(im)
    
    for ind in range(0,3):
        sd[0,ind] = np.std(im[:,:,ind])
    
    for ind in range(0,3):  
        alpha[0,ind] = np.minimum(sd[0,ind],sd[0,1])/(sd[0,ind]+sd[0,1])
        res[:,:,ind] = alpha[0,ind]*im[:,:,ind]+(1.0-alpha[0,ind])*im[:,:,1]
    
    c_sum = GetIntensity(res) + np.finfo(np.float32).eps
    
    bi = np.minimum(i_sum/c_sum, 2.0)
    for ind in range(0,3):  
        res[:,:,ind] = res[:,:,ind]*bi
    
    return res


def GreenMeanPreservingNormalization(im):
    res = np.empty(im.shape,im.dtype) 
    ave = np.zeros((1,3))
    im_h = np.zeros((1,3))
    im_l = np.zeros((1,3)) 

    for ind in range(0,3):
        ave[0,ind] = np.mean(im[:,:,ind])
    
    for ind in range(0,3):
        im_h[0,ind] = np.percentile(im[:,:,ind],99)
        im_l[0,ind] = np.percentile(im[:,:,ind],1)
        res[:,:,ind] = (im[:,:,ind]-ave[0,ind])/(im_h[0,ind]-im_l[0,ind])+ave[0,1]
    return res


def Stretch(im):
    res = np.empty(im.shape,im.dtype) 
    im_h = np.zeros((1,3))
    im_l = np.zeros((1,3)) 
    
    im_h = np.percentile(im, 99.5)
    im_l = np.percentile(im,0.5)

    imax = im_h
    imin = im_l
   
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-imin)/(imax-imin) 
        
    return res


def MeanCorrection(im):
    res = np.empty(im.shape,im.dtype) 
    imn = np.empty(im.shape,im.dtype)  
    kapa = np.zeros((1,3))   
        
    im_sum = 3.0*GetIntensity(im)+np.finfo(np.float32).eps
  
    for ind in range(0,3):
        imn[:,:,ind]=im[:,:,ind]/im_sum
     
    histb = cv2.calcHist([f2i(imn[:,:,0])],[0],None,[256],[0,255])
    histg = cv2.calcHist([f2i(imn[:,:,1])],[0],None,[256],[0,255])
    histr = cv2.calcHist([f2i(imn[:,:,2])],[0],None,[256],[0,255])

    hstb = np.ravel(histb)
    hstg = np.ravel(histg)
    hstr = np.ravel(histr)
    
    sh=255
    co_gb= np.correlate(hstb, hstg,'full' )
    kapa[0,0] = np.argmax(co_gb)-sh
      
    co_gr = np.correlate(hstr, hstg, 'full')
    kapa[0,2] = np.argmax(co_gr)-sh
 
    kapa[0,1] = 0
    
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind] + kapa[0,ind]/sh)
          
    return res


## Transmission
def EstimateTransmission(I, gamma_max):
    T_min = 0.1
    me = np.finfo(np.float32).eps
    hi = GetIntensity(I)
    hmin = GetMin(I)

    V = PixelAdaptiveGamma(I, gamma_max)
 
    ji = GetIntensity(V)
    jmin = GetMin(V)
    
    tn = np.maximum(ji*hmin - hi*jmin, me)   
    td = np.maximum(ji*hi - jmin*hi, me)
    Tmap = 1.0 - hi*(tn/td)
         
    return np.clip(Tmap, T_min, 1.0)


def PixelAdaptiveGamma(I, gamma_max):
    V = np.empty(I.shape,I.dtype)
    imax = np.mean(GetIntensity(I))
    
    a = (gamma_max-1)/(np.exp(1)-1)
    b = 1-a
    gamma = a*np.exp(imax) + b
    
    for ind in range(0,3):
       V[:,:,ind] = (I[:,:,ind]**gamma)
    
    return V


## Recover
def Recover(I, T, A):
    res = np.empty(I.shape,I.dtype)
    
    for ind in range(0,3):
        res[:,:,ind] = (I[:,:,ind]-A[0,ind])/T+A[0,ind]
    
    return np.clip(res, 0.0, 1.0)

'''
 Main
'''
def main(InputImg, gamma_max=2.0):
    start_time = time.time()
######################################################
    Img = i2f(InputImg)

    CCFImg = ColorCorrectionFirst(Img)
    A = A_Tang(CCFImg)

    CCFNormImg = np.empty(Img.shape,Img.dtype)
    for ind in range(0,3):
        CCFNormImg[:,:,ind] = CCFImg[:,:,ind]/A[0,ind]
    CCFNormImg = Norm(CCFNormImg)
    Tmap = EstimateTransmission(CCFNormImg, gamma_max)

    RecoverImg = Recover(CCFImg, Tmap, A)
    OutputImg = ColorCorrectionLast(RecoverImg)
######################################################
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    
    return OutputImg


if __name__ == "__main__":
    FolderTemp = args.InputPath.split('/')
    OutputPath = args.OutputPath + '/Output/' + FolderTemp[-1]
    os.makedirs(OutputPath, exist_ok=True, mode=0o777)
    
    FileList = [file for file in os.listdir(args.InputPath) if
                    (file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg"))
                    or file.endswith(".webp") or file.endswith(".tiff") or file.endswith(".tif")
                    or file.endswith(".bmp") or file.endswith(".png")]

    for FileNum in range(0, len(FileList)):
        FilePathName = args.InputPath + '/' + FileList[FileNum]
        InputImg = cv2.imread(FilePathName, cv2.IMREAD_COLOR)
    
        OutputImg = main(InputImg, args.gamma_max)

        Name = os.path.splitext(FileList[FileNum])
        cv2.imwrite(OutputPath + '/' + Name[0] + '.png', f2i(OutputImg))
