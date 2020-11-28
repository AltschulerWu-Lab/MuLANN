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

#In that case, the data is dl in the right format already. Nothing to do besides add data location to settings.lua
import os, shutil
from optparse import OptionParser

if __name__ == '__main__':
    code_folder = os.path.dirname(os.getcwd())
    # Working on Ubuntu 16.04, Python3.6.5
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder

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
    f.write("\nimage_folder = '{}'\n".format(data_folder))
    f.close()
