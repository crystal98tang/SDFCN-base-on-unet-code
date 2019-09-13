import skimage.transform as trans
import numpy as np
import numpy
import scipy.misc as misc
import os

from PIL import Image
import matplotlib.pyplot as plt

def dc(input1, input2):
    """
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC = \frac{2|A\capB|}{|A|+|B|}

    , where A is the first and B the second set of samples (here binary objects).

    Parameters
    ----------
    input1: array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2: array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc: float
        The Dice coefficient between the object(s) in `input1` and the
        object(s) in `input2`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric.
    """
    input1 = numpy.atleast_1d(input1.astype(numpy.bool))
    input2 = numpy.atleast_1d(input2.astype(numpy.bool))

    intersection = numpy.count_nonzero(input1 & input2)

    size_i1 = numpy.count_nonzero(input1)
    size_i2 = numpy.count_nonzero(input2)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def dice_ratio(preds, labels):
    '''
    preds & labels should only contain 0 or 1.
    '''
    if np.sum(preds) + np.sum(labels) == 0:
        return 1
    return np.sum(preds[labels==1])*2.0 / (np.sum(preds) + np.sum(labels))
 

test_path = "temp_data/temp_test"
sum = num_image = 1000
target_size = (256,256)
sum_SDFCN = 0
sum_FCN = 0
sum_differ = 0
for i in range(num_image):
    image = misc.imread(os.path.join(test_path, "%d.tif" % i)).astype(np.float64)
    img = misc.imread(os.path.join(test_path,"%d_2_predict.tif"%i)).astype(np.float64)
    img[img == 0] = 0
    img[img == 255] = 1

    img_FCN = misc.imread(os.path.join(test_path,"%d_2_FCN.tif"%i)).astype(np.float64)
    # img_FCN = np.resize(img_FCN,(256,256))
    img_FCN[img_FCN == 0] = 0
    img_FCN[img_FCN == 255] = 1

    mark = misc.imread(os.path.join(test_path,"%d_gt.tif"%i)).astype(np.float64)
    # mark = np.resize(mark,(256,256))
    mark[mark == 0] = 0
    mark[mark == 255] = 1

    test_SDFCN = dice_ratio(img,mark)
    test_FCN = dice_ratio(img_FCN, mark)
    if test_SDFCN == 0:
        sum -= 1
        continue

    sum_SDFCN += test_SDFCN
    sum_FCN += test_FCN
    print("%d:test_SDFCN:%.3f\ttest_FCN:%.3f"%(i,test_SDFCN,test_FCN))

avg_SDFCN = sum_SDFCN / sum
avg_FCN = sum_FCN / sum
print("平均Dice\nSDFCN:%.5f\nFCN:%.5f"%(avg_SDFCN,avg_FCN))
