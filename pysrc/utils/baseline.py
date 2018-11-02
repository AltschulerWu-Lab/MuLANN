from utils.settings import Settings
from collections import defaultdict
from math import ceil, sqrt
import numpy as np
import os, sys, pdb, pickle, cv2
from scipy import ndimage

class SingleImagePrinter(object):
    '''
    Baseline code for opening 1-color images from the same field of view, eventually renormalizing them,
    and saving as a RGB image.

    Approved: as of the 23d of Jan 2017, tested that it does keep the intensity ranges correctly and does not
    arbitrarily renormalize images one by one.
    '''

    def __init__(self, settings, channels):
        self.settings = settings

        self.channels = channels
        try:
            self.imagetype = self.settings.imagetype
        except:
            self.imagetype = 'png'

    def normalize(self, image, levelset, imageFolder):
        '''
        Opening an image in UINT16, normalizing it and returning it
        :param image:
        :param levelset:
        :param imageFolder:
        :return:
        '''

        im = cv2.imread(os.path.join(imageFolder, image), -1).T
        #print(np.max(im))
        min_, max_=levelset

        im[np.where(im<min_)]=min_
        im[np.where(im>max_)]=max_

        res= np.array((im-min_)/float(max_-min_)*(256-1), dtype=np.dtype('uint8'))
        #print (np.max(res))
        return res

    def save(self, image, plate, well, outputFolder = '', image_filename = '', verbose=True, single_folder = False, site=0):
        '''
        Saving renormalized image
        :param image:
        :param plate:
        :param well:
        :param outputFolder:
        :return:
        '''

        outputFolder = self.settings.image_folder if outputFolder=='' else outputFolder
        image_filename = self.settings.image_filename if image_filename =='' else image_filename
        out_file = image_filename.format(plate, well, site)

        if not os.path.isdir(outputFolder):
            print("Creating ", outputFolder)
            os.mkdir(outputFolder)

        if plate not in os.listdir(outputFolder) and not single_folder:
            os.mkdir(os.path.join(outputFolder, plate))
            print("Creating ", os.path.join(outputFolder, plate))

        #When you specify the correct writing and opening types then no renormalization occurs
        image = np.array(image, dtype=np.uint8)
        image = np.transpose(image, (1,0,2))
        if not single_folder:
            full_output_file = os.path.join(outputFolder, plate, out_file)
        else:
            full_output_file =  os.path.join(outputFolder, out_file)
        written = cv2.imwrite(full_output_file, image)
        if not written:
            raise OSError('Could not write {}'.format(full_output_file))
        if verbose:
            print ("Saved image {}".format(out_file))

        return full_output_file


    def _get_normalized_image(self, levelsets, imagelist, inputFolder):
        res_image = None
        for i, image in enumerate(imagelist):
            currChannel = self.settings.channel_extractor(image)
            if self.channels != "all" and currChannel not in self.channels:
                continue

            res_image = self.normalize(image, levelsets[i], inputFolder) if res_image is None \
                else np.dstack((res_image, self.normalize(image, levelsets[i], inputFolder)))

        return res_image

    def get_normalized_image(self, plate, well, levelsets, site="all"):

        folder = os.path.join(self.settings.input_folder, plate, self.settings.subFolder, well)
        images = sorted(os.listdir(folder))
        images = list(filter(lambda x: x[:2]!='._' and '.DS_Store' not in x, images))
        images = list(filter(lambda  x: self.imagetype in x, images))

        #Easy case when only one site per well
        yield 0, self._get_normalized_image(levelsets, images, folder)
