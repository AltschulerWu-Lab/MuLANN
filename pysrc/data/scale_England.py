import os, shutil, cv2
import numpy as np
from scipy import ndimage
from optparse import OptionParser

from utils.settings import Settings
from stitch_England import CaieStitcher
from utils import bgs_folder, image_setting_file, output_image_name, england_plates
from utils.baseline import SingleImagePrinter

class CaieScaler(CaieStitcher):
    def __init__(self, metadata_dir, outputfolder, levelsets=[(0, 15000), (0, 27000), (0, 20000)], zoom=0.25, verbose=False):
        super(CaieScaler, self).__init__(metadata_dir)

        self.zoom = zoom
        self.levelsets = levelsets
        self.verbose = verbose
        self.metadata_dir = metadata_dir
        self.image_setting_file = os.path.join(metadata_dir, image_setting_file)
        self.outputfolder = outputfolder
        self.scaledfolder = os.path.join(os.path.dirname(self.outputfolder), 'ScaledCaie')

    def scaledown(self, plateList):

        settings = Settings(self.image_setting_file)
        settings.input_folder = self.outputfolder
        settings.imagetype = settings.imagetype_Caie
        scaledfolder = self.scaledfolder

        if not os.path.isdir(scaledfolder):
            os.mkdir(scaledfolder)
        #Preparing out folders
        self._prepare_folders(self.final_df, scaledfolder)

        printer = SingleImagePrinter(settings, channels="all")

        print("Median filter on MCF-7, size 2 pixels")
        print("Multiplying size by ", self.zoom)
        print("Using levelsets ", self.levelsets)

        for plate in plateList:
            print(plate)
            wells = sorted(os.listdir(os.path.join(self.outputfolder, plate, bgs_folder)))

            for well in wells:
                for site, image in printer.get_normalized_image(plate, well, levelsets=self.levelsets):
                    image[:, :, 1] = ndimage.median_filter(image[:, :, 1], 2)
                    image = image[:,:, [2,1,0]]
                    if self.zoom != -1:
                        image = ndimage.zoom(image, [self.zoom, self.zoom, 1])
                    printer.save(image, plate, well, outputFolder=os.path.join(scaledfolder, plate, bgs_folder, well),
                                 verbose=self.verbose, single_folder=True, site=site)

        return

    def move_images(self, plateList, final_outputfolder):
        for plate in plateList:
            print(plate)
            if not os.path.isdir(os.path.join(final_outputfolder, plate)):
                os.mkdir(os.path.join(final_outputfolder, plate))

            curr_infolder = os.path.join(self.scaledfolder, plate, bgs_folder)
            wells = sorted(os.listdir(curr_infolder))

            for well in wells:
                line = self.final_df[(self.final_df.Plate==plate)&(self.final_df.Well==well)]
                drug_class = line.DrugClass.values[0]

                if not os.path.isdir(os.path.join(final_outputfolder, plate, drug_class)):
                    os.mkdir(os.path.join(final_outputfolder, plate, drug_class))

                cpd = line.Drug.values[0]
                curr_finalfolder = os.path.join(final_outputfolder, plate, drug_class, cpd)
                if not os.path.isdir(curr_finalfolder):
                    os.mkdir(curr_finalfolder)

                image = os.listdir(os.path.join(curr_infolder, well))[0]
                shutil.move(os.path.join(curr_infolder, well, image), curr_finalfolder)

if __name__ == '__main__':
    code_folder = os.path.dirname(os.getcwd())
    #Working on Ubuntu 16.04, Python3.6.5
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder

    metadata_dir = os.path.join(code_folder, 'metadata')
    outputfolder = os.path.join(data_folder, 'StitchedCaie')
    final_outputfolder = os.path.join(data_folder, 'Cells_div4')

    scaler = CaieScaler(metadata_dir=metadata_dir, outputfolder=outputfolder)
    plateList = list(filter(lambda x: os.path.isdir(os.path.join(outputfolder, x)), sorted(os.listdir(outputfolder))))
    scaler.scaledown(plateList)

    #Finally, we need to place these images in the right folder
    scaler.move_images(plateList, final_outputfolder)