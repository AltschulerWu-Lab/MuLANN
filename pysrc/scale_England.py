class CaieQC(CaieStitcher):
    def __init__(self, levelsets=[(0, 15000), (0, 27000), (0, 20000)], zoom=0.25, verbose=False):
        super(CaieQC, self).__init__()
        self.zoom = zoom
        self.levelsets = levelsets
        self.verbose = verbose


    def step3_scaledown(self, plate):
        plateList = [plate] if plate is not None else sorted(os.listdir(outputfolder))

        settings = Settings(image_setting_file)
        settings.input_folder = settings.input_folder['Caie']
        settings.imagetype = settings.imagetype_Caie
        printer = SingleImagePrinter(settings, channels="all")

        print("Median filter on MCF-7, size 2 pixels")
        print("Multiplying size by ", self.zoom)
        print("Using levelsets ", self.levelsets)
        cell_line = 'MCF7'

        for plate in plateList:
            print(plate)
            wells = sorted(os.listdir(os.path.join(outputfolder, plate, bgs_folder)))

            for well in wells:
                for site, image in printer.get_normalized_image(plate, well, levelsets=self.levelsets):
                    image[:, :, 1] = ndimage.median_filter(image[:, :, 1], 2)
                    if self.zoom != -1:
                        image = ndimage.zoom(image, [self.zoom, self.zoom, 1])
                    printer.save(image, plate, well, outputFolder=os.path.join(scaledfolder, plate, bgs_folder, well),
                                 verbose=self.verbose, single_folder=True, site=site)

        return

    # After, need to down scale /4 and save as png

    def _DMSO_values(self, image_filename, plate, empty_wells):
        qc_list = self._read_well_list_file(plate)
        ctrl_means = np.zeros(shape=(3, len(negative_control_wells)), dtype=float)
        count_means = np.zeros(shape=(len(negative_control_wells)), dtype=float)
        i = 0

        values = [[] for k in range(3)]

        for well in negative_control_wells:
            if well not in qc_list and well not in empty_wells:
                im = cv2.imread(
                    os.path.join(scaledfolder, plate, bgs_folder, well, image_filename.format(plate, well, 0)), -1)
                for k in range(3):
                    values[k].extend(im[:, :, k].flatten())

                ctrl_means[:, i] = im.mean(0).mean(0)
                count_means[i] = im.size

                i += 1
        DMSO_pixel_value_mean = np.dot(ctrl_means[:, :i], count_means[:i]) / np.sum(count_means[:i]) / (
                2 ** 8 - 1)  # should be of shape 3

        values = [np.array(values[k]) for k in range(3)]
        perc1 = np.array([scoreatpercentile(values[k][values[k] > 0], 5) for k in range(3)])
        perc99 = np.array([scoreatpercentile(values[k][values[k] > perc1[k]], 99.9) for k in range(3)])
        print(perc1, perc99)
        perc1 /= (2 ** 8 - 1)
        perc99 /= (2 ** 8 - 1)

        self.save_control_values(plate, DMSO_pixel_value_mean, perc1, perc99)

        return DMSO_pixel_value_mean

    def _read_well_list_file(self, plate, folder='Caie_QC_step1'):
        # Reading the QC for wells that are out of focus or just completely empty or have a hair
        # This was manual QC by me

        f = open(os.path.join(Caie_metadata_folder, folder, '{}.txt'.format(plate)), 'r')
        lines = f.readlines();
        f.close()

        return [el.rstrip('\n') for el in lines]

    def _rename_channel_image(self, plate, well, channel, files):
        im_name = output_image_name.format(plateID=plate, well=well, channel=channel)
        im_channel = list(filter(lambda x: '{}.tif'.format(channel) in x, files))[0]
        shutil.move(os.path.join(outputfolder, plate, bgs_folder, well, im_channel),
                    os.path.join(outputfolder, plate, bgs_folder, well, im_name))

    def _image_name(self, plate, well, channel):
        #        if well[1]=='0':
        #           return output_image_name.format(plateID=plate, well=well.replace('0', ''), channel=3)
        im_name = output_image_name.format(plateID=plate, well=well, channel=channel)
        files = os.listdir(os.path.join(outputfolder, plate, bgs_folder, well))
        if im_name not in files:
            self._rename_channel_image(plate, well, 3, files)
            self._rename_channel_image(plate, well, 1, files)
            self._rename_channel_image(plate, well, 2, files)

        return im_name

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
                        print('ERRRRR (del)', plate, well, e)
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

        self._rename_well_folder(plate, well, images)


    def _rename_well_folder(self, plate, well, images):
        if well[1] == '0':
            folder = os.path.join(outputfolder, plate, bgs_folder)
            try:
                os.mkdir(os.path.join(outputfolder, plate, bgs_folder, well.replace('0', '')))
            except FileExistsError:
                return
            for image in images:
                shutil.copy(os.path.join(folder, well, image),
                            os.path.join(folder, well.replace('0', ''), image.replace(well, well.replace('0', ''))))

    def _delete_well_folders(self, plateList):
        for plate in plateList:
            wells = os.listdir(os.path.join(outputfolder, plate, bgs_folder))
            for well in wells:
                if len(well) == 2:
                    shutil.rmtree(os.path.join(outputfolder, plate, bgs_folder, well))
