--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 12/8/16
-- Time: 3:27 PM
-- To change this template use File | Settings | File Templates.
--

-- I need to open the images and just select subpatches where it is a cell or a cell context, write the patches to another folder
-- inspired from https://github.com/torch/demos/blob/master/train-autoencoder/autoencoder-data.lua

require 'image'
require 'pl'
require 'cunn'
require 'gnuplot'

utils = require 'utils'
general_data = require 'general_data'
require 'settings'
local data_verbose = false

local function getdata(dataset_name, DMSO_mean, inputsize,
                        dataset_role, set_name)

    --Gives me an iterator on the list of plates in the folder
    local dataset ={}
    dataset.DMSO_mean = DMSO_mean
    dataset.imagelist= {}
    dataset.label_image_num = {}
    local imageDict = utils.get_class_images(dataset_name, dataset_role,
                            (set_name=='unlabtrain' and 'test') or set_name, true)
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
                    local image_file = path.join(inputfolder, im_name)
                    table.insert(dataset.imageDict[class], image_file)
                    curr_count = curr_count+1
                end
                index = index+1
            end
        end
    dataset.label_image_num[class]=curr_count
    end

   --A little hack for the noise experiments
   if dataset_role=='target' and ((opt.noise=='unlabelled' and set_name=='train') or (opt.noise=='full')) then
       dataset.label_image_num[opt.reverse_classes[8]]=0
       dataset.label_image_num[opt.reverse_classes[9]]=0
       dataset.label_image_num[opt.reverse_classes[10]]=0
   end

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

   function dataset:image_normalization(im, plate)
       return utils.AlexNet_image_normalization(im, self.DMSO_mean[plate])
   end

   local dsample = torch.Tensor(3, inputsize,inputsize)
    if opt.nchannelsin ==3 then
        function dataset:loadImage(image_filename)
            local plate = utils.get_plate(image_filename)
            local im = image.load(image_filename)

            self:image_normalization(im, plate)

            return im
        end
    elseif opt.nchannelsin==2 then
        if dataset_name=='Caie' then
            --In that case we want 1,2 channels
            function dataset:loadImage(image_filename)
                local plate = utils.get_plate(image_filename)
                local im = image.load(image_filename)

                self:image_normalization(im, plate)
                im[3]:fill(0)
                return torch.cat({im[1]:reshape(1, im:size(2), im:size(3)),
                    im[3]:reshape(1, im:size(2), im:size(3)),
                    im[2]:reshape(1, im:size(2), im:size(3))}, 1)
            end
        else
            --In that case we want 1,3 channels
            function dataset:loadImage(image_filename)
                local plate = utils.get_plate(image_filename)
                local im = image.load(image_filename)

                self:image_normalization(im, plate)
                im[2]:fill(0)
                return im
            end
        end
    end

    function dataset:_selectImagePatch(image_name, nr, nc)
        --NORMALIZATION IS DONE HERE
        local im = self:loadImage(image_name)
        return general_data.cropImage(im, nr, nc)
    end

-- RETURNS a NORMALIZED PATCH, its label and the full image
    function dataset:selectPatch(nc, num_sampled_classes)
        --image index
        local label
        local drug_class
        local curr_num_images

        while not curr_num_images or curr_num_images<1 do
            label = torch.random(1, num_sampled_classes)
            drug_class = opt.reverse_classes[label]
            curr_num_images = self.label_image_num[drug_class]
        end

        return general_data.selectImagePatch(self, nc, label)
    end

   function dataset:_getClassifBatch(batchsize, num_sampled_classes)
      local targets = torch.Tensor(batchsize)
      local inputs = torch.Tensor(batchsize,dsample:size(1), dsample:size(2), dsample:size(3))

      for i = 1,batchsize do
            -- load new sample
          local sample ,label = self:selectPatch(inputsize, num_sampled_classes)
          inputs[i] = sample:clone()
          targets[i]= label
      end

      return inputs, targets
    end

   function dataset:getBatch(batchsize)
       return self:_getClassifBatch(batchsize, opt.num_classes)
   end

   return dataset
end


function getdataset(dataset_name, dataset_role, opt)
    --[[ This function is a dataset getter. It will return an object which can be called
    --as many times as one want, which will randomly select a given number of images for each class
    -- amongst a set of plates.
    --
    -- The first thing is to pick up available images, and save the list.
        -- If train and test come from the same plates, images are divided in train and test,
        -- and this is saved too.
    -- Then it estimates mean and std dev of images on all images, or just train images
                -- if they are picked from the same plates.
     -- Images are loaded from image_folder, and all outputs are saved in model_folder.

     ]]
    local datasetgetter={}
    datasetgetter.mean={}

    print('Loading DMSO means')
    local plate_files = paths.iterfiles(path.join(metadata_dir,string.format('%s_info', dataset_name), string.format('%s_DMSO_values_paper', dataset_name)))
    for file in plate_files do
        local plate = file:split('.')[1]
        datasetgetter.mean[plate], _,_ = utils.read_DMSO_values(plate, dataset_name)
    end

    function datasetgetter:trainDataset()
        return getdata(dataset_name, self.mean, opt.inputsize, dataset_role, 'train')
    end

    function datasetgetter:testDataset()
        return getdata(dataset_name, self.mean, opt.inputsize, dataset_role, 'test')
    end

    function datasetgetter:restTestDataset()
        return getdata(dataset_name, self.mean, opt.inputsize, dataset_role, 'resttest')
    end

    if dataset_role=='target' and opt.target_unlabelled_presence then
        function datasetgetter:unlabelledTrainDataset()
            local set_name = 'unlabtrain'
            --Using images from test classes to learn DA part
            return getdata(dataset_name, self.mean, opt.inputsize, dataset_role, 'unlabtrain')
        end
    end

    function datasetgetter:valDataset()
        return getdata(dataset_name, self.mean, opt.inputsize, dataset_role, 'val')
    end

    function datasetgetter:restValDataset()
        return getdata(dataset_name, self.mean, opt.inputsize, dataset_role, 'restval')
    end

    return datasetgetter
end

local data_preparation = {
    getdataset = getdataset,
    displayData = displayData
}
return data_preparation
