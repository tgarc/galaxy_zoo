import matplotlib.pyplot as plt

Channels = ['g', 'r', 'i']
Chan_features = ["ellipticity", "area"
                 ,"bulge_prominence", "bulge_presence", "disc_dev"
                 ,"m1","m2"]

Nonchan_features = ["gid"]
Featureslist = Nonchan_features \
               + [f + "_%s" % c for c in Channels for f in Chan_features]
Output_format = ",".join(["%%(%s)s" % f for f in Featureslist])


def plotimg(imgs, titles=None, fig=None, ncols=None, nrows=None, **plotargs):
    fig = plt.figure() if fig is None else fig
    nplots = len(imgs)

    if not nrows and not ncols:
        nrows = nplots
        ncols = 1
    elif not ncols:
        ncols = nplots / nrows + nplots % nrows
    elif not nrows:
        nrows = nplots / ncols + nplots % ncols

    axidx = lambda n: nrows*100 + ncols*10 + n + 1

    for i, im in enumerate(imgs):
        axis = fig.add_subplot(axidx(i))
        axis.imshow(im, plt.cm.gray, **plotargs)
        axis.set_title(titles[i] if titles is not None else i)

    plt.tight_layout()

    return fig

