import os, shutil
from optparse import OptionParser

england_plates = ['Week10_40111',  'Week10_40115',  'Week10_40119',  'Week1_22123', 'Week1_22141',
 'Week1_22161', 'Week1_22361', 'Week1_22381', 'Week1_22401', 'Week2_24121', 'Week2_24141',
 'Week2_24161', 'Week2_24361', 'Week2_24381', 'Week2_24401', 'Week3_25421', 'Week3_25441',
 'Week3_25461', 'Week3_25681', 'Week3_25701', 'Week3_25721', 'Week4_27481', 'Week4_27521',
 'Week4_27542', 'Week4_27801', 'Week4_27821', 'Week4_27861', 'Week5_28901', 'Week5_28921',
 'Week5_28961', 'Week5_29301', 'Week5_29321', 'Week5_29341', 'Week6_31641', 'Week6_31661',
 'Week6_31681', 'Week6_32061', 'Week6_32121', 'Week6_32161', 'Week7_34341', 'Week7_34381',
 'Week7_34641', 'Week7_34661', 'Week7_34681', 'Week8_38203', 'Week8_38221', 'Week8_38241',
 'Week8_38341', 'Week8_38342', 'Week9_39206', 'Week9_39221', 'Week9_39222', 'Week9_39282',
 'Week9_39283', 'Week9_39301']


if __name__=='__main__':
    code_folder = os.path.dirname(os.getcwd())
    #Working on Ubuntu 16.04, Python3.6.5
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder

    #Dl metadata
    cmd = 'wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv'
    os.system(cmd)
    shutil.move('BBBC021_v1_image.csv', os.path.join(code_folder, 'metadata', 'Caie_info'))

    # Now downloading data if not already there
    #You can launch this during your lunch break or something - it'll take some time
    cmd = 'wget https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_{}.zip'
    cmd2 = 'unzip BBBC021_v1_images_{}.zip'
    for plate in england_plates:
        if not os.path.isdir(os.path.join(data_folder, plate)):
            os.system(cmd.format(plate))
            os.system(cmd2.format(plate))
            shutil.move(plate, data_folder)

