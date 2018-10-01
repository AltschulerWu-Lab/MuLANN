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

require 'image'
require 'pl'
require 'cunn'
require 'gnuplot'

utils = require 'utils'
general_data = require 'general_data'
require 'settings'
local data_verbose = false
local X = office_size
local Y = office_size

total_num_images = {
    ["amazon"]=2817,
    ["dslr"]=498,
    ["webcam"]=795
}

local function getdata(dataset_name, mean, inputsize, dataset_role, set_name)
    local dataset ={}
    dataset.imagelist= {}
    dataset.mean = mean
    dataset.inputsize = inputsize
    dataset.label_image_num = {}

    local imageDict = utils.get_class_images(dataset_name, dataset_role,
                            (set_name=='unlabtrain' and 'test') or set_name)
    print('Loading ', dataset_name, dataset_role, set_name)

    local num_image_per_class
    local condition = function(index) return true end
    if not opt.fully_transductive and dataset_role=='target'
                    and (set_name=='unlabtrain' or set_name=='test') then
        if opt.target_supervision_level==50 then --semi-supervised setting
            num_image_per_class = 10
        else                        --supervised setting
            num_image_per_class = 7
        end

        if set_name=='unlabtrain' then
            condition = function(index) return index<num_image_per_class end
        elseif set_name=='test' then
            condition = function(index) return index>=num_image_per_class end
        end
    end

    --Gives me an iterator on the list of plates in the folder
    dataset.imageDict = {}
    for class,_ in pairs(opt.classes) do
        local curr_count = 0
        if not dataset.imageDict[class] then
           dataset.imageDict[class] = {}
        end
        local index = 0
        if imageDict[class] then
            for _, im_name in ipairs(imageDict[class]) do
                if condition(index) then
                    local image_file = path.join(office_folder, im_name)
                    table.insert(dataset.imageDict[class], image_file)
                    curr_count = curr_count+1
                end
                index = index+1
            end
        end
    dataset.label_image_num[class]=curr_count
    end

    if (opt.noise=='unlabelled' and dataset_role=='source' and set_name=='train')
      or (opt.noise=='labelled' and dataset_role=='target' and set_name=='test')
    then
        for _,class in ipairs(office_asym_removed_classes) do dataset.label_image_num[class]=0 end
    elseif opt.noise=='full' and
            ((dataset_role=='source' and set_name=='train')
          or (dataset_role=='target' and set_name=='test')) then
        for _,class in ipairs(office_fullasym_removed_classes[dataset_name]) do dataset.label_image_num[class]=0 end
    end

   local autoencoding = false

   --This will return for which classes i have images
   function dataset:classes()
       local result = torch.zeros(opt.num_classes)
       for k=1, opt.num_classes do
           local class = opt.reverse_classes[k]
           if self.label_image_num[class] and self.label_image_num[class]>0 then
              result[k]=1
           end
       end
       return result:view(1, -1)
   end

   function dataset:encoding_conversion()
       autoencoding = true
   end

-- RETURNS a NON-NORMALIZED PATCH, its label and the full image
   function dataset:selectPatch(nr,nc)
        --image index
        local label, class
        local label_len=0
        while not label_len or label_len<1 do
            label = torch.random(1, opt.num_classes)
            class = opt.reverse_classes[label]
            label_len = self.label_image_num[class]
        end
        local index = torch.random(1, label_len)

        return self:_selectImagePatch(self.imageDict[class][index], nr, nc),label
   end

    function dataset:_selectImagePatch(image_name, nr, nc)
        local im = image.load(image_name)
        --Getting the images Caffe ready:
        im:mul(255)
        im=im:index(1,torch.LongTensor{3,2,1})
        --Then normalizing the images
        im:add(-self.mean)
        return general_data.cropImage(im, nr, nc)
    end

    function dataset:_getClassifBatch(batchsize)
        local targets = torch.Tensor(batchsize)
        local inputs = torch.Tensor(batchsize,opt.nchannelsin, opt.inputsize,opt.inputsize)

        for i = 1,batchsize do
            -- load new sample
            local sample ,label = self:selectPatch(self.inputsize, self.inputsize)
            inputs[i] = sample:clone()
            targets[i]= label
        end

        return inputs, targets
    end

    function dataset:getGivenImageBatch(batchsize, class, index)
        local result = torch.Tensor(batchsize, 3, opt.inputsize, opt.inputsize)
        for i = 1,batchsize do
            -- load new sample
            local sample = self:_selectImagePatch(self.imageDict[class][index], self.inputsize, self.inputsize)
            result[i] = sample:clone()
        end
        return result
    end

    function dataset:getBatch(batchsize)
        return self:_getClassifBatch(batchsize)
    end
   return dataset
end


function getdataset(dataset_name, dataset_role, opt)
    local datasetgetter={}
    --amazon, webcam or dslr
    datasetgetter.name = dataset_name
    --source or target
    datasetgetter.role = dataset_role

    print('Getting ds ', datasetgetter.name)
    local mean = general_data.load_imagenet_mean(opt)

    function datasetgetter:trainDataset()
        local set_name = 'train'
        return getdata(self.name, mean, opt.inputsize, self.role, set_name)
    end

    function datasetgetter:testDataset()
        local set_name = 'test'
        return getdata(self.name, mean, opt.inputsize, self.role, set_name)
    end
    function datasetgetter:rest_testDataset()
        local set_name = 'resttest'
        return getdata(self.name, mean, opt.inputsize, self.role, set_name)
    end
    if datasetgetter.role=='target' and opt.target_unlabelled_presence then
        function datasetgetter:unlabelledTrainDataset()
            local set_name = 'unlabtrain'
            --Using images from test classes to learn DA part
            return getdata(self.name, mean, opt.inputsize, self.role, set_name)
        end
    end
    return datasetgetter
end

local data_preparation = {
    getdataset = getdataset,
}
return data_preparation
