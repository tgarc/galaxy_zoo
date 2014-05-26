#!/usr/bin/python

import os, sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tomas_features as tg
import marcus_features as mt
from common import *

if __name__ == "__main__":
    import optparse

    parser = optparse.OptionParser(usage="image_features.py [options]")

    parser.add_option("-i", "--imagepath", dest="impath"
                      ,default=None
                      ,help="Directory of input galaxies (required)")
    parser.add_option("-n", "--number", dest="number"
                      ,type="int", default=None
                      ,help="Number of images to process (required)")
    parser.add_option("--gid", dest="gid", default=None
                      ,help="Specific galaxy ID(s) by comma separated list.")
    parser.add_option("-o", "--offset", dest="offset"
                      ,type="int", default="0"
                      ,help="Start processing after [offset] images")
    parser.add_option("-s", "--savepath", dest="savepath"
                      ,default=None
                      ,help="Destination for output CSV file")    
    parser.add_option("-p", "--plot", dest="plot"
                      ,action="store_true", default=False
                      ,help="Save png plots of results for each galaxy (not implemented)")
    parser.add_option("-v", "--verbose", dest="verbose"
                      ,action="store_true", default=False
                      ,help="Print verbose output to stdout.")

    (opts, args) = parser.parse_args()

    if not opts.impath or (not opts.number and not opts.gid):
        sys.exit("One or more required parameter is missing.")
    # if not opts.savepath and opts.plot:
    #     sys.exit("Savepath required.")
    opts.impath = os.path.abspath(opts.impath)

    # print feature order to first line
    logfile = open(opts.savepath,'w') if opts.savepath else sys.stdout
    print >> logfile, ','.join(Featureslist)

    # load image paths
    if opts.gid:
        fnames = [gid + '.jpg' for gid in opts.gid.split(",")]
    else:
        fnames = os.listdir(os.path.abspath(opts.impath))[opts.offset : opts.offset+opts.number]
    galaxyIDs = [int(fn.strip('.jpg')) for fn in fnames]

    # preallocate space for separate channel images
    G, R, I = [np.empty((424,424),dtype=np.uint8)] * 3

    # set up plot objects
    if opts.plot:
        if not opts.savepath:
            plt.ion()
        fig = plt.figure()
        axes = [fig.add_subplot(220+i) for i in range(1,5)]

    # intitialize a dictionary of features
    featureDict = dict(zip(Featureslist, [None]*len(Featureslist)))
    for i, img in enumerate(cv2.imread(os.path.join(opts.impath,fn)) for fn in fnames):
        if opts.verbose:
            sys.stdout.write("\rProcessing Galaxy ID %6d..." % galaxyIDs[i])

        if opts.plot:
            axes[0].imshow(img)
            axes[0].set_title("galaxy id %d" % galaxyIDs[i])
            
        featureDict['gid'] = galaxyIDs[i]

        G, R, I = (img[..., j] for j in range(3))
        for c, chan in enumerate((G, R, I)):
            mask, roi = tg.FindGalaxyBlob(chan)
            ellipticity, area = tg.EllipseFeatures(mask)
            bulge_prominence, bulge_presence, disc_dev = mt.bulge_features(roi, mask)
            moments = mt.inv_moments(mask)
            ### insert additional feature extraction functions here ###

            featureDict['ellipticity_%s' % Channels[c]] = ellipticity
            featureDict['area_%s' % Channels[c]] = area
            featureDict['bulge_prominence_%s' % Channels[c]] = bulge_prominence
            featureDict['bulge_presence_%s' % Channels[c]] = bulge_presence
            featureDict['disc_dev_%s' % Channels[c]] = disc_dev
            featureDict['m1_%s' % Channels[c]] = moments[0]
            featureDict['m2_%s' % Channels[c]] = moments[1]
            ### assign additional features here ###

            if opts.plot:
                axes[c+1].imshow(mask,vmin=0,vmax=1)
                axes[c+1].set_title("channel %s" % Channels[c])

        if opts.plot:
            if opts.savepath:
                fig.savefig("gid_%d.png" % galaxyIDs[i])
            else:
                fig.canvas.draw()

        # output all the features for this image
        featureStr = Output_format % featureDict
        print >> logfile, featureStr

        # flush out the file write buffer to minimize risk of lost results
        if (i % 10) == 0:
            logfile.flush()

        # clear dictionary
        for k in featureDict:
            featureDict[k] = None

    if logfile != sys.stdout:
        logfile.close()

    if opts.plot and not opts.savepath:
        a = raw_input("Enter anything to exit...")


