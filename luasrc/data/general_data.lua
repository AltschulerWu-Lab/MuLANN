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

local function load_imagenet_mean(opt)
    local mean
    if opt.model=='VGG' then
        --Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].
        mean = torch.Tensor(3, office_size, office_size)
        mean[1]:fill(103.939)
        mean[2]:fill(116.779)
        mean[3]:fill(123.68)
    else
        print('Which model? Need to implement normalization')
        print(a+u)
    end

    return mean
end

local function cropImage(im, nr, nc)
    local ri = torch.random(1,im:size(2)-nr)
    local ci = torch.random(1,im:size(3)-nc)

    return im:narrow(2,ri,nr):narrow(3,ci,nc)
end


local function selectImagePatch(dataset, inputsize, label)
    local drug_class = opt.reverse_classes[label]
    local curr_num_images = dataset.label_image_num[drug_class]

    local i = torch.random(1, curr_num_images)
    local image_name = dataset.imageDict[drug_class][i]
    return dataset:_selectImagePatch(image_name, inputsize, inputsize),label
end

local general = {
    load_imagenet_mean = load_imagenet_mean,
    selectImagePatch = selectImagePatch,
    cropImage = cropImage,
}
return general
