from __future__ import print_function

from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from tensorflow.python.keras.utils import np_utils
import scipy.misc as misc

def adjustData(img,mask):
    """
    Pre-
    :param img:
    :param mask:
    :return:
    """
    if np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    mask = to_categorical(mask, num_classes=2)
    return img,mask

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (256,256),seed = 1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,  # None'categorical'
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None, # None'categorical'
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)

def testGenerator(test_path,num_image): # flag_multi_class = False,target_size = (256,256)
    for i in range(num_image):
        img = misc.imread(os.path.join(test_path,"%d.tif"%i),mode='RGB')
        img = img / 255
        # img = trans.resize(img,target_size)
        img = np.reshape(img, (1,) + img.shape)
        # img = np.reshape(img,img.shape+(1,)) if not flag_multi_class else img
        yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img1 = item[:,:,0]
        img2 = item[:,:,0]
        # 256图
        misc.imsave(os.path.join(save_path,"%d_255_predict.tif"%i),img1)
        # 0或1 二值图
        img2[img2 >= 0.5] = 0
        img2[img2 < 0.5] = 255
        misc.imsave(os.path.join(save_path,"%d_2_predict.tif"%i),img2)



def saveBigResult(save_path,npyfile,box,each_image_size,num):
    # （left, upper, right, lower）

    # < class 'tuple'>: (4352, 0, 4864, 512)
    # < class 'tuple'>: (3072, 4096, 3584, 4608)
    empty_plate = np.zeros([5000, 5000, 3], np.uint8)
    for i,item in enumerate(npyfile):
        for count in each_image_size:
            Image.blend(box[count][0])
        if i % each_image_size:
            misc.imsave(os.path.join(save_path, "%d_big_predict.tif" % i), empty_plate)
            empty_plate = np.zeros([5000, 5000, 3], np.uint8)