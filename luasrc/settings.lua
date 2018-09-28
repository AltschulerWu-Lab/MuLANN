------------------------------------------------------------------------
-- Copyright 2018, University of California, San Francisco
-- Author: Alice Schoenauer Sebag for the Altschuler and Wu Lab
--
-- All rights reserved.
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, version 3 of the License.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details:
-- < http://www.gnu.org/licenses/ >.
------------------------------------------------------------------------

local currPath = paths.thisfile()
code_folder = paths.dirname(paths.dirname(currPath))
rundir = path.join(code_folder, 'results')
final_result_folder = rundir
metadata_dir = path.join(code_folder, 'metadata')
image_folder = ''

office_size =256
digits_size = 32
max_target_unlabelled = 50

office_casepicking_folder = "train_test_sets/office/%s/%s/%s" --name of domain, then source/target then train/test
office_casepicking_file = "%ssplit_0%d.txt" --if source, then '' and num_fold, elseif target 'same_category-' if supervised, 'diff_category-' if semi-supervised
normalization_setting = "global" -- other possibility: 'local' for local contrast normalization.
normalization_file = "mean_%s_X%s_p%s.png"
tsne_feature_file = "tsne_features_%s.csv"
feature_file = "features_%s.csv"

quanti_log_file = 'quanti_log.txt'
verbose_log_file = 'verbose_log.txt'
modelfilename = "model"

-- Computation characteristics
num_threads = 35

adapt_factor = 3e-7
adapt_alpha = 0.1
salera_lambda = 1
--Parameter epsilon to decide when the gradient is considered zero
epsilon = 1e-15

verbose = false
display = false

-- Testing parameters
testingBatch = 320.0
numTestingBatch =1
testSetPercentage = 0.35

--For labelled or unlabelled asymmetry, here are the classes which are removed from resp. domain 2 or 1
office_asym_removed_classes = {'projector',
                             'punchers',
                             'ring_binder',
                             'ruler',
                             'scissors',
                             'speaker',
                             'stapler',
                             'tape_dispenser',
                             'trash_can'}
--For full labelled AND unlabelled asymmetry, here are the classes which are removed from domain 1 and 2
--We have num_classes = 26
office_fullasym_removed_classes = {
    ["webcam"] = {'scissors', 'speaker', 'stapler', 'tape_dispenser'},
    ["amazon"] = {'projector', 'punchers', 'ring_binder', 'ruler'}
}
