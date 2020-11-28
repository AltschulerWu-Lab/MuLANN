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
import os, shutil
from optparse import OptionParser

import pandas

from pysrc.utils import england_plates

plate_col_name = "Image_Metadata_Plate_DAPI"
well_col_name = "Image_Metadata_Well_DAPI"
cpd_col_name = "Image_Metadata_Compound"
dose_col_name = "Image_Metadata_Concentration"

if __name__ == '__main__':
    code_folder = os.path.dirname(os.getcwd())
    # Working on Ubuntu 16.04, Python3.6.5 or 3.5.4
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder
    caie_folder = os.path.join(data_folder, 'RawCaie')

    if not os.path.isdir(caie_folder):
        os.mkdir(caie_folder)

    # Dl metadata
    cmd = 'wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv'
    os.system(cmd)
    df = pandas.read_csv('BBBC021_v1_image.csv')
    cols = list(df.columns)
    cols[cols.index(plate_col_name)]='Plate'
    cols[cols.index(well_col_name)]='Well'
    cols[cols.index(cpd_col_name)]='Compound'
    cols[cols.index(dose_col_name)]='Dose'
    df.columns = cols
    df.to_csv('BBBC021_v1_image.csv')
    shutil.move('BBBC021_v1_image.csv', os.path.join(code_folder, 'metadata', 'Caie_info'))

    # Now downloading data if not already there
    # You can launch this during your lunch break or something - it'll take some time
    cmd = 'wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_{}.zip'
    cmd2 = 'unzip BBBC021_v1_images_{}.zip'
    cmd3 = 'rm BBBC021_v1_images_{}.zip'

    for plate in england_plates:
        if not os.path.isdir(os.path.join(data_folder, plate)):
            # Downloading
            os.system(cmd.format(plate))
            # Unzipping
            os.system(cmd2.format(plate))
            # Moving images to data folder
            shutil.move(plate, caie_folder)
            # Removing zip file
            os.system(cmd3.format(plate))
