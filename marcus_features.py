'''
marcus' part of the feature extraction code
'''
from common import *
import numpy as np
import ellipticity as ell

def bulge_features(img,thresh_img):
    # first get the statistics
    disc_pix = img[thresh_img==1]
    disc_med = np.median(disc_pix)
    disc_std = np.std(disc_pix)
    # then calculate the bulge prominence and bulge presence scores
    bulge_threshold = disc_med + 2*disc_std
    bulge_pix = disc_pix[disc_pix > bulge_threshold]
    if len(bulge_pix) > 0:
        bulge_prominence = 1.0 * len(bulge_pix) / len(disc_pix)
        bulge_presence = np.median(bulge_pix) / np.median(disc_pix[disc_pix <= bulge_threshold])
    else:
        bulge_prominence = 0
        bulge_presence = 0
    # then compute the standard deviation of disc - bulge pixels, to use in
    # determining spirality
    disc_dev = np.std(disc_pix[disc_pix <= bulge_threshold])
    
    return bulge_prominence, bulge_presence, disc_dev
    
def inv_moments(thresh_img):
    # Get the normalized central geometric moments
    n20 = ell.mt_ncgm(2, 0, thresh_img)
    n02 = ell.mt_ncgm(0, 2, thresh_img)
    n11 = ell.mt_ncgm(1, 1, thresh_img)
    # Get Hu's moments
    moments = [0,0]
    moments[0] = ell.mt_I1_p(n20, n02)
    moments[1] = ell.mt_I2_p(n20, n02, n11)
    return moments
