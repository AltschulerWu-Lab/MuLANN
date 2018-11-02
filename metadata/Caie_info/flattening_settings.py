
################### For image flattening
imagetype_Caie='tif'

subFolder = "bgs_images"

#png is better because non-lossy
image_filename = "FL_{}_{}_{}.png"

###Function for getting channel number from image name
channel_extractor = lambda x: int(x[-5])

###Function for getting site number from image name
site_extractor = lambda x: int(x.split('_')[-1][:5])