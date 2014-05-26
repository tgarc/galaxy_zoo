'''
tomas' part of the feature extraction code
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import morphology as mphlgy
from common import *


Cx, Cy = (212, 212)


class Blob:
    def __init__(self, mask):
        self.mask = mask
        self.area = np.sum(self.mask)
        self.x, self.y = self.findCoM()

    def findCoM(self):
        colnums = range(np.shape(self.mask)[1])
        rownums = range(np.shape(self.mask)[0])

        x = np.sum(np.apply_along_axis(lambda x: x*colnums, 0, self.mask)) \
            // self.area
        y = np.sum(np.apply_along_axis(lambda x: x*rownums, 1, self.mask)) \
            // self.area

        return x, y

    def __repr__(self):
        return str(vars(self).values())

    def __str__(self):
        # return ", ".join(["%s: %s" % (k.capitalize(),v) for k, v in vars(self).items()
        #                   if k != 'mask' and k != 'contour' and k != 'ellipse'])
        return ("Centroid: (%g, %g), Area: %d px"
                % (self.x, self.y, self.area))


def FindGalaxyBlob(img):
    kern = mphlgy.square(5)    

    # binary threshold the image with otsu algorithm
    t, timg = cv2.threshold(img, 0, 1
                            ,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # timg = np.array(timg > t,dtype=np.uint8)
    # mphlgy.opening(timg, kern, timg)

    # label blobs and sort id's by blob size
    labeledimg, numblobs = mphlgy.label(timg, neighbors=8
                                        ,return_num=True, background=0)
    blobAreas = [np.sum(labeledimg == bid) for bid in range(numblobs)]
    sortedLabels = np.argsort(blobAreas)

    # prefer most centered blob
    if np.sum(np.array(blobAreas) > 1000) > 1:
        bigblobs = [Blob(labeledimg == b) for b in sortedLabels
                    if blobAreas[b] > 1000]
        offsets = tuple(abs(b.x-Cx)+abs(b.y-Cy) for b in bigblobs)
        bestBlob = bigblobs[np.argmin(offsets)]
    else:
        bestBlob = Blob(labeledimg == sortedLabels[-1])

    # extract the region of interest from the image
    diff = bestBlob.mask.any(axis=0)
    ones = np.flatnonzero(diff)
    xmin, xmax = ones[0], ones[-1]
    diff = bestBlob.mask.any(axis=1)
    ones = np.flatnonzero(diff)
    ymin, ymax = ones[0], ones[-1]

    roi = img[ymin:ymax+1, xmin:xmax+1].astype(np.uint8)
    mask = bestBlob.mask[ymin:ymax+1, xmin:xmax+1].astype(np.uint8)

    return (mask, roi)


def EllipseFeatures(bmask):
    # find contours on external boundary only
    contours = cv2.findContours(bmask,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[0]
    try:
        if len(contours) > 1:
            bestContourIdx = np.argmax([cv2.contourArea(c) for c in contours])
        else:
            bestContourIdx = 0
        ellipse = cv2.fitEllipse(contours[bestContourIdx])
    except:
        print "\n Error: Ellipse fit failed."
        ellipticity = None
        area = None
        total_lum = None
        concentration = None
    else:
        area = np.sum(bmask)
        centroid, axes, angle = ellipse
        minoraxis_length, majoraxis_length = axes

        if minoraxis_length == 0:
            ellipticity = np.inf
            print "\n Warning: Ellipse has zero minor axis."
        else:
            ellipticity = majoraxis_length / minoraxis_length

        # create an elliptical mask with half the axis
        # lengths of the disc
        # bulge = np.zeros((majoraxis_length,majoraxis_length),dtype=np.uint8)
        # bellipse = (centroid
        #             ,(0.1*minoraxis_length,0.1*majoraxis_length)
        #             ,angle)
        # cv2.ellipse(bulge,ellipse,1,thickness=-1)            

        # total_lum = np.sum(img[bestBlob.mask])
        # bulge_lum = np.sum(img[bulge])
        # concentration = bulge_lum / total_lum

        # plt.close()
        # cv2.ellipse(bulge,ellipse,1)
        # cv2.ellipse(img,ellipse,255,thickness=1)
        # cv2.ellipse(img,bellipse,255,thickness=-1)
        # f = plt.figure(); ax = f.add_subplot(111); ax.imshow(img,plt.cm.gray); plt.show()

    return ellipticity, area

# featureDict['total_luminance_%s'%Channels[c]] = total_lum
# featureDict['concentration_%s'%Channels[c]] = concentration

