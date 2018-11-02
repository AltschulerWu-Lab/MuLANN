import os, cv2, pandas, shutil, math, pdb
import scipy.ndimage as ndimage
import numpy as np
from optparse import OptionParser

from utils import bgs_folder, output_image_name, compute_stitching_macro, copy_stitching_macro

redo = False

class CaieStitcher(object):
    def __init__(self, metadata_dir, intelligent_stitching=False):
        # i. read csv file which tells us all about the images
        self.df = pandas.read_csv(os.path.join(metadata_dir, 'Caie_info', 'BBBC021_v1_image.csv'))

        #ii. Read csv which tells us about the images we will be using (phenoactive images)
        final_df = pandas.read_csv(os.path.join(metadata_dir, 'Caie_info', 'Final_dataset_Caie.csv'))
        self.final_df = pandas.DataFrame(final_df[final_df.DrugClass!='Inactive'])

        # For images with few cells, we cannot do the intelligent stitching because not able to paste images against eachother
        # using intensity correlations
        self.intelligent_stitching = intelligent_stitching

    def _prepare_folders(self, df, datafolder):
        plates = df.Plate.unique()
        for plate in plates:
            try:
                os.mkdir(os.path.join(datafolder, plate))
            except:
                pass
            try:
                os.mkdir(os.path.join(datafolder, plate, bgs_folder))
            except:
                pass

            wells = df[df.Plate == plate].Well.unique()
            for well in wells:
                try:
                    os.mkdir(os.path.join(datafolder, plate, bgs_folder, well))
                except:
                    pass

    def stitch_condition(self, currPlate, currWell):
        '''
        # a. copy the images to the right folder
        # b. for each channel rename to tile_01--04.tif
        # c. apply the stitching
        # read the text file and crop the output image
        # save it using output_image_name
        # delete the intermediary results
        :param condition:
        :return:
        '''
        currDf = self.df[(self.df.Well == currWell)&(self.df.Plate==currPlate)]

        self.folder = os.path.join(outputfolder, currPlate, bgs_folder, currWell)
        if not redo and len(list(filter(lambda x: 'Caie_plate' in x, os.listdir(self.folder)))) == 3:
            return

        # So for the Oracl we had H2B, then XRCC5 then cytoplasm.
        images = {1: currDf.Image_FileName_DAPI,
                  2: currDf.Image_FileName_Tubulin,
                  3: currDf.Image_FileName_Actin  # looks more like a classic cytoplasm than the tubulin channel
                  }
        self.dimensions = np.zeros(shape=(3, 2))
        if self.intelligent_stitching:
            # First of all, compute stitching on the Actin channel
            print('###########TAKING CARE OF CHANNEL 3')
            self.stitch_actin(currPlate, currWell, images[3])
            # Then use this stitching for the other two channels
            print('###########TAKING CARE OF CHANNEL 1')
            self.transfer_actin_stitch(currPlate, currWell, channel=1, images=images[1])
            print('###########TAKING CARE OF CHANNEL 2')
            self.transfer_actin_stitch(currPlate, currWell, channel=2, images=images[2])
            # Finally checking that nothing bad happened during stitching
            print('#######CHECKING GAPS')
            self.check_channel_dimensions(images, currPlate, currWell)

        else:
            print('Going for unintelligent stitching')
            # Just putting the 4 images next to eachother and scaling down the size
            for channel, image_list in images.items():
                self._unintelligent_stitching(image_list, currPlate, currWell, channel)

    def _unintelligent_stitching(self, image_list, plate, well, channel):
        self._copy_images(image_list, plate)
        final_im = np.ndarray(shape=(1024 * 2, 1280 * 2), dtype=np.uint16)

        im = cv2.imread(os.path.join(self.folder, 'tile_01.tif'), -1)
        final_im[:1024, :1280] = im

        im = cv2.imread(os.path.join(self.folder, 'tile_02.tif'), -1)
        final_im[:1024, 1280:] = im

        im = cv2.imread(os.path.join(self.folder, 'tile_03.tif'), -1)
        final_im[1024:, :1280] = im

        im = cv2.imread(os.path.join(self.folder, 'tile_04.tif'), -1)
        final_im[1024:, 1280:] = im

        curr_output_image_name = output_image_name.format(plateID=plate, well=well, channel=channel)
        final_im = ndimage.zoom(final_im, 0.5)

        print('Writing ', curr_output_image_name)
        done = cv2.imwrite(os.path.join(self.folder, curr_output_image_name), final_im)
        self._del_leftover_files(done)

    def check_channel_dimensions(self, images, plate, well):
        print(self.dimensions)
        gaps = np.abs(self.dimensions - self.dimensions[2])
        # So the correct channel is the third
        print(gaps)
        self._check_channel_dimension(gaps[0], images[1], plate, well, 1)
        self._check_channel_dimension(gaps[1], images[2], plate, well, 2)

    def _check_channel_dimension(self, gaps, images, plate, well, channel):
        if gaps[0] > 10 or gaps[1] > 10:
            print('##########GAP CHANNEL {}'.format(channel))

            # Then I'm going to use the file with registration to just fuse the images roughly
            final_im = np.zeros(shape=(3000, 3000), dtype=np.uint16)
            x1, y1, x2, y2, x3, y3, x4, y4 = self._read_registration_for_restitching()

            self._copy_images(images, plate)
            im1 = cv2.imread(os.path.join(self.folder, 'tile_01.tif'), -1)
            final_im[y1:y1 + im1.shape[0], x1:x1 + im1.shape[1]] = im1
            # cv2.imwrite(os.path.join(self.folder, 'img_mystitch.tif'), final_im)

            im2 = cv2.imread(os.path.join(self.folder, 'tile_02.tif'), -1)
            final_im[y2:y2 + im2.shape[0], x2:x2 + im2.shape[1]] = im2[:min(im2.shape[0], final_im.shape[0] - y2)]
            # cv2.imwrite(os.path.join(self.folder, 'img_mystitch.tif'), final_im)
            X = max(y1 + im1.shape[0], y2 + im2.shape[0])
            Y = max(x1 + im1.shape[1], x2 + im2.shape[1])

            im3 = cv2.imread(os.path.join(self.folder, 'tile_03.tif'), -1)
            final_im[y3:y3 + im3.shape[0], x3:x3 + im3.shape[1]] = im3
            #            cv2.imwrite(os.path.join(self.folder, 'img_mystitch.tif'), final_im)
            X = max(X, y3 + im3.shape[0])
            Y = max(Y, x3 + im3.shape[1])

            im4 = cv2.imread(os.path.join(self.folder, 'tile_04.tif'), -1)
            final_im[y4:y4 + im4.shape[0], x4:x4 + im4.shape[1]] = im4[:min(im4.shape[0], final_im.shape[0] - y4)]
            X = max(X, y4 + im4.shape[0])
            Y = max(Y, x4 + im4.shape[1])

            cv2.imwrite(os.path.join(self.folder, 'img_mystitch.tif'), final_im[:X, :Y])
            done = self.crop_stitched_result(plate, well, channel)
            self._del_leftover_files(done)

    def _del_leftover_files(self, done):
        if done:
            files = os.listdir(self.folder)
            for file in files:
                if 'img_' in file or 'tile_0' in file:
                    os.remove(os.path.join(self.folder, file))

    def _copy_images(self, images, plate):
        for image in images:
            stitch_id = int(image.split('_s')[1][0])
            shutil.copy(os.path.join(inputfolder, plate, image),
                        os.path.join(self.folder, 'tile_0{}.tif'.format(stitch_id)))

    def transfer_actin_stitch(self, plate, well, channel, images):
        self._copy_images(images, plate)
        macro = copy_stitching_macro.format(outputdir=self.folder)
        f = open('macro.ijm', 'w')
        f.write(macro)
        f.close()
        os.system('{} --headless -macro {} > out.txt'.format(fiji_location, os.path.join(os.getcwd(), 'macro.ijm')))

        done = self.crop_stitched_result(plate, well, channel=channel)
        self._del_leftover_files(done)

    def stitch_actin(self, plate, well, actin_images):
        self._copy_images(actin_images, plate)
        macro = compute_stitching_macro.format(outputdir=self.folder)
        f = open('macro.ijm', 'w')
        f.write(macro)
        f.close()
        os.system('{} --headless -macro {} > out.txt'.format(fiji_location, os.path.join(os.getcwd(), 'macro.ijm')))

        done = self.crop_stitched_result(plate, well, channel=3)
        # Copying right registration file
        print('Copying file to ', outputfolder)
        shutil.copy(os.path.join(self.folder, 'CorrectTileConfiguration.registered.txt'), outputfolder)
        self._del_leftover_files(done)

    def crop_stitched_result(self, plate, well, channel):
        '''
        Stitching leaves black patches on the corners of the image, so this step is cutting
        the final image so that there are no black patches.

        :param plate:
        :param well:
        :param channel:
        :return:
        '''
        curr_output_image_name = output_image_name.format(plateID=plate, well=well, channel=channel)
        files = os.listdir(self.folder)
        # The stitched image
        img_result = list(filter(lambda x: 'img_' in x, files))[0]
        # The cropping info
        top, left, right, bottom = self._read_registration_for_cropping(files)

        # Bien checker que tout est bon niveau format de l'image
        im = cv2.imread(os.path.join(self.folder, img_result), -1)
        if channel == 3:
            self.channel3_dimension = im.shape

        im = ndimage.zoom(im[top:-bottom, left:-right], 0.5)

        self.dimensions[channel - 1] = im.shape
        print('Writing ', curr_output_image_name)
        return cv2.imwrite(os.path.join(self.folder, curr_output_image_name), im)

    def _read_registration_for_restitching(self):
        shutil.copy(os.path.join(outputfolder, 'CorrectTileConfiguration.registered.txt'), self.folder)
        f = open(os.path.join(self.folder, 'CorrectTileConfiguration.registered.txt'), 'r')
        lines = f.readlines();
        f.close()
        x2, b, c, y3, x4, y4 = self.__read_registration_file(lines)
        print(x2, b, c, y3, x4, y4)
        if c > 0:
            x1 = 0
            x3 = c
        else:
            x1 = abs(c)
            x3 = 0

        if b > 0:
            y1 = 0
            y2 = b
        else:
            y1 = abs(b)
            y2 = 0
        print(x1, y1, x2, y2, x3, y3, x4, y4)
        return math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2), math.ceil(x3), math.ceil(y3), math.ceil(
            x4), math.ceil(y4)

    def __read_registration_file(self, lines):
        for line in lines:
            if 'tile' in line:
                ll = line.split('(')[1].split(',')
                if 'tile_02' in line:
                    a = float(ll[0])
                    top = float(ll[-1][:-2])

                if 'tile_03' in line:
                    left = float(ll[0])
                    c = float(ll[-1][:-2])

                if 'tile_04' in line:
                    b = float(ll[0])
                    d = float(ll[-1][:-2])

        return a, top, left, c, b, d

    def _read_registration_for_cropping(self, files):
        registration_results = list(filter(lambda x: 'registered' in x, files))[0]
        f = open(os.path.join(self.folder, registration_results), 'r')
        lines = f.readlines();
        f.close()

        a, top, left, c, b, d = self.__read_registration_file(lines)
        top = math.ceil(abs(top))
        left = math.ceil(abs(left))
        right = math.ceil(abs(b - a))
        bottom = math.ceil(abs(c - d))

        return top, left, right, bottom


    def harmonize_pixel_crops(self, plateList):
        for plate in plateList:
            folder = os.path.join(outputfolder, plate, bgs_folder)
            if not os.path.isdir(folder):
                continue
            wells = os.listdir(folder)

            for well in wells:
                images = list(
                    filter(lambda x: x[-4:] == '.tif', os.listdir(os.path.join(outputfolder, plate, bgs_folder, well))))

                if len(images) == 3:
                    try:
                        self._crop_refactor(folder, plate, well, images)
                    except AttributeError as e:
                        print ('ERRRRR (del)', plate, well, e)
                        for el in os.listdir(os.path.join(folder, well)):
                            os.remove(os.path.join(folder, well, el))

    def _crop_refactor(self, folder, plate, well, images):

        im_list = {}
        for image in sorted(images):
            im_list[image] = cv2.imread(os.path.join(folder, well, image), -1)
        xs = np.array([el.shape[0] for el in im_list.values()])
        ys = np.array([el.shape[1] for el in im_list.values()])

        if np.any(xs < 950) or np.any(ys < 1200):
            print('ATTENTION IMAGE ANORMALEMENT PETITE (del)', plate, well, xs, ys)
            for el in os.listdir(os.path.join(folder, well)):
                os.remove(os.path.join(folder, well, el))
            return

        x = np.min(xs)
        y = np.min(ys)

        diff_x = np.array([el - x for el in xs])
        diff_y = np.array([el - y for el in ys])

        if np.max(diff_x) > 20 or np.max(diff_y) > 20:
            print('ATTENTION GROS DECALAGE (del)', plate, well, diff_x, diff_y)
            for el in os.listdir(os.path.join(folder, well)):
                os.remove(os.path.join(folder, well, el))
            return

        for image in images:
            cv2.imwrite(os.path.join(folder, well, image), im_list[image][:x, :y])

    def __call__(self, data_folder):
        # i. Prepare output folders
        self._prepare_folders(self.final_df, data_folder)

        # ii. For each plate for each well,
        for i,el in self.final_df.iterrows():
#For test            if el.Plate=='Week10_40111':
            self.stitch_condition(currPlate=el.Plate, currWell=el.Well)

if __name__ == '__main__':
    code_folder = os.path.dirname(os.getcwd())
    #Working on Ubuntu 16.04, Python3.6.5
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    parser.add_option('--fiji', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder
    fiji_location = options.fiji

    metadata_dir = os.path.join(code_folder, 'metadata')
    inputfolder = os.path.join(data_folder, 'RawCaie')
    outputfolder = os.path.join(data_folder, 'StitchedCaie')

    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)

    #Doing intellignt stitching - ie using FiJi
    preparer = CaieStitcher(metadata_dir, intelligent_stitching=True)
    preparer(outputfolder)

    #Not perfect so smt it screws up image sizes in the different channels
    print('Now harmonizing the image sizes')
    plateList = filter(lambda x: os.path.isdir(os.path.join(outputfolder, x)), os.listdir(outputfolder))
    preparer.harmonize_pixel_crops(plateList)

    #Now for the failed images, just putting them together
    print('Now finishing stitching')
    preparer = CaieStitcher(metadata_dir, intelligent_stitching=False)
    preparer(outputfolder)