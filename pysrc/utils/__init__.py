
## VARS

england_plates = ['Week10_40111',  'Week10_40115',  'Week10_40119',  'Week1_22123', 'Week1_22141',
 'Week1_22161', 'Week1_22361', 'Week1_22381', 'Week1_22401', 'Week2_24121', 'Week2_24141',
 'Week2_24161', 'Week2_24361', 'Week2_24381', 'Week2_24401', 'Week3_25421', 'Week3_25441',
 'Week3_25461', 'Week3_25681', 'Week3_25701', 'Week3_25721', 'Week4_27481', 'Week4_27521',
 'Week4_27542', 'Week4_27801', 'Week4_27821', 'Week4_27861', 'Week5_28901', 'Week5_28921',
 'Week5_28961', 'Week5_29301', 'Week5_29321', 'Week5_29341', 'Week6_31641', 'Week6_31661',
 'Week6_31681', 'Week6_32061', 'Week6_32121', 'Week6_32161', 'Week7_34341', 'Week7_34381',
 'Week7_34641', 'Week7_34661', 'Week7_34681', 'Week8_38203', 'Week8_38221', 'Week8_38241',
 'Week8_38341', 'Week8_38342', 'Week9_39206', 'Week9_39221', 'Week9_39222', 'Week9_39282',
 'Week9_39283', 'Week9_39301'
                  ]

bgs_folder = 'bgs_images'
output_image_name = 'Caie_plate_{plateID}_20x_t48_{well}_0000-{channel}.tif'

compute_stitching_macro = '''run("Grid/Collection stitching", "type=[Grid: row-by-row] order=[Right & Down                ] grid_size_x=2 grid_size_y=2 tile_overlap=5 first_file_index_i=1 directory={outputdir} file_names=tile_{{ii}}.tif output_textfile_name=CorrectTileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap display_fusion computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory={outputdir}");'''
copy_stitching_macro = '''run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory={outputdir} layout_file=CorrectTileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory={outputdir}");'''

image_setting_file = 'Caie_info/flattening_settings.py'