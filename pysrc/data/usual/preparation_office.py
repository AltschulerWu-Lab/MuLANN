#------------------------------------------------------------------------
#-- Copyright 2018, University of California, San Francisco
#-- Author: Alice Schoenauer Sebag for the Altschuler and Wu Lab
#--
#-- All rights reserved.
#-- This program is free software: you can redistribute it and/or modify
#-- it under the terms of the GNU General Public License as published by
#-- the Free Software Foundation, version 3 of the License.
#--
#-- This program is distributed in the hope that it will be useful,
#-- but WITHOUT ANY WARRANTY; without even the implied warranty of
#-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#-- GNU General Public License for more details:
#-- < http://www.gnu.org/licenses/ >.
#------------------------------------------------------------------------
"""
Only necessary when running Torch models - PyTorch rescales images using the initial 'transform'
when loading dataset.

Warning: conflict between cv2 and Python > 3.5
"""

import cv2, os, shutil
from scipy import ndimage
import numpy as np
from optparse import OptionParser
'''
This will perform the scaling of all images in office dataset to 256x256
'''
goal_size = 256.0
num_channels = 3


class OfficeScaler(object):
    def __init__(self):
        pass

    @staticmethod
    def _rescale(image_filename):
        im = cv2.imread(image_filename)

        curr_size = im.shape[0]
        scale = goal_size/curr_size
        newImage = ndimage.zoom(im, [scale, scale, 1])
        newImage[np.where(newImage > 255)] = 255
        newImage = np.array(newImage, dtype='uint16')
        return newImage

    @staticmethod
    def save_image(im, dataset, class_, im_name):
        cv2.imwrite(os.path.join(data_folder, 'scaled', dataset, class_, im_name), im)

    def rescale(self, dataset):
        classes = os.listdir(os.path.join(data_folder, 'raw', dataset, 'images'))

        for class_ in classes:
            try:
                os.mkdir(os.path.join(data_folder, 'scaled', dataset, class_))
            except FileExistsError:
                pass
            images = os.listdir(os.path.join(data_folder, 'raw', dataset, 'images', class_))

            for im in images:
                newImage = self._rescale(os.path.join(data_folder, 'raw', dataset, 'images', class_, im))
                self.save_image(newImage, dataset, class_, im)

        return

    def __call__(self, *args, **kwargs):
        datasets = os.listdir(os.path.join(data_folder, 'raw'))
        for ds in datasets:
            try:
                os.mkdir(os.path.join(data_folder, 'scaled', ds))
            except FileExistsError:
                pass
            print('Rescaling ', ds)
            self.rescale(ds)


if __name__ == '__main__':
    code_folder = os.path.dirname(os.getcwd())
    # Working on Ubuntu 16.04, Python3.6.5
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder

    for f in ['raw', 'scaled']:
        try:
            os.mkdir(os.path.join(data_folder, f))
        except FileExistsError:
            pass

    for folder in ['amazon', 'dslr', 'webcam']:
        try:
            shutil.move(os.path.join(data_folder, folder), os.path.join(data_folder, 'raw', folder))
        except FileNotFoundError:
            print('Pb, did not find ', folder)

    scaler = OfficeScaler()
    print('Launching image down-scaling')
    scaler()

    # Now downloading VGG_16 if not already there
    cmd = 'wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
    if not os.path.isfile(os.path.join(code_folder, 'metadata', 'vgg_16', 'VGG_ILSVRC_16_layers.caffemodel')):
        os.system(cmd)
        shutil.move('VGG_ILSVRC_16_layers.caffemodel', os.path.join(code_folder, 'metadata', 'vgg_16'))

    # Creating a result folder
    try:
        os.mkdir(os.path.join(code_folder, 'results'))
    except FileExistsError:
        pass

    # Finally, adding the location data_folder to the lua settings_file
    f = open(os.path.join(code_folder, 'luasrc', 'settings.lua'), 'a')
    f.write("\noffice_folder = '{}'\n".format(os.path.join(data_folder, 'scaled')))
    f.close()
