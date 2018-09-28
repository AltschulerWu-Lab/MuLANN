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

package.path = package.path
        .. string.format(';%s/luasrc/data/?.lua' , code_folder)
        .. string.format(';%s/luasrc/models/?.lua' , code_folder)
        .. string.format(';%s/luasrc/utilitaries/?.lua', code_folder)

require 'utils'
require 'settings'
require 'optim'
model_level = require 'office_model'
data_level = require 'office_data'
general_data = require 'general_data'
require 'pl'

torch.setnumthreads(num_threads)
redo = true

--Options for input data
opt.nchannelsin = 3
opt.inputsize = 224
opt.model='VGG'
opt.dataset = 'OFFICE'
opt.train_file = 'train_all.lua'

opt.num_patch_per_im = 10
opt.batchsize = 32
opt.seed = 5
opt.nDonkeys = 4
opt.testingBatch = testingBatch
opt.bottleneck_size  = 256
opt.num_disc_neurons = 1024
--Baselines are done with source by default
opt.incl_source=true
--Setting the proba to give unlabelled, inactive and active batches to the DA part while training
opt.proba = 0.5

--Details from DANN paper, for dynamic lambda setting and learning rate decay
opt.domain_method='DANN'
--Actually same details for MADA, except gamma which should be 0.0003
opt.lambda_schedule_gamma = 10
--Now setting parameters for learning rate schedule as in paper.
--If this is used the learning rate decay should be set to 0
opt.gamma = 0.001
opt.beta = 0.75

--Details for optimization
opt.weightDecay = 0
opt.SAGMA_lambda = 1
opt.lr_decay = 0

opt.common_class_percentage = 100
opt.target_supervision_level = 100
opt.target_unlabelled_presence = false
local drug_iter = 0
local shuffling_iter = 0

function produce_model(outputfolder)
    if not opt.domain_adaptation then
        opt.domainLambda = 0
        opt.domain_method = nil
    end

    opt.num_common_labelled_classes = opt.dataset=='OFFICE' and 15 or 7
    if opt.target=='Caie' or opt.source=='Caie' then
        opt.nchannelsin = 2
        opt.num_common_labelled_classes = 4
    end

    local train_parameters = utils.return_train_parameters(opt)
    local shuffling_iter = opt.fold
    inputfolder, outputfolder = utils.prepare_for_learning(outputfolder, opt, shuffling_iter)
    if inputfolder==-1 then return end

    opt.num_classes = opt.num_common_classes
    print('Going with ', outputfolder)
    hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
    if hasCudnn then cudnn.fastest=true; cudnn.benchmark =true end

    --Computation details
    opt.outputfolder = outputfolder
    paths.dofile('threader.lua')

    -- Getting datasetgetters
    source_data_getter = data_level.getdataset(opt.source, 'source', opt)
    target_data_getter = data_level.getdataset(opt.target, 'target', opt)

    print('Image div ', opt.image_div," - Done loading all data getters.")
    if opt.dataset=='BIO' then
        maxiter = 20002
        plotinterval =1000
        statinterval =10000

    elseif opt.dataset=='OFFICE' then
        maxiter = 15002 -- MADA: 10k iter
        plotinterval =1000
        statinterval =5000

    end

    --ii. Load model and start training
    opt.num_classwise_disc = opt.num_classes
    model = model_level.getmodel(opt.domain_method, train_parameters)
    print(model.feature_extractor, model.label_predictor, model.domain_predictor)

    if opt.domain_method=='MuLANN' then opt.train_file = 'train_asym.lua' end
    print(opt)
    local status, err = pcall(function ()
        paths.dofile(opt.train_file)
     end)
    if not status then
        print(err)
        utils.write_stop_reason(outputfolder, err)
    end

    return
end

function evaluate_all(redo, dataset, filter_str)
    local folders = paths.iterdirs(final_result_folder)
    local domains, domain_names
    local sets = {'test'}
    local filter_str = filter_str or 'unlab'
    opt.dataset = dataset or 'OFFICE'

    if opt.dataset=='BIO' then
        table.insert(sets, 'val')
        domains = {'source', 'target' , 'source'}
    elseif opt.dataset=='OFFICE' then
        domains = {'source','target'}
        if opt.target_supervision_level==0 then
            sets = {'train'} 
        end
    end
    local iterations = {10001, 15001,20001,30001,40001}

    for folder in folders do
        local bugfile = path.join(final_result_folder, folder, 'model_unopenable')
        if not paths.filep(bugfile) then
            if folder:find(string.lower(opt.dataset)) and folder:find('VGG') and folder:find(filter_str) then
                utils.get_source_target(folder)
                if not pcall(function ()
                    domain_names = opt.dataset=='BIO' and {opt.source, opt.target, opt.extra_domain} or {opt.source, opt.target}
                    for i, domain in ipairs(domain_names) do
                        for _, setname in ipairs(sets) do
                            for _, iteration in ipairs(iterations) do
                                evaluate_model(folder, iteration, domain, domains[i], setname, redo)
                            end
                        end
                    end
                end) then
                local file = io.open(bugfile, 'w'); file:close()
                print('Pb, ', folder) end
            end
        end
    end
end

local function set_num_predicted_classes(model_folder)
    local num_predicted_classes = opt.num_classes
    if model_folder:find('noise') then
        if opt.dataset=='BIO' and (model_folder:find('labelled') or model_folder:find('full')) then
                num_predicted_classes = 10
        elseif opt.dataset=='OFFICE' then
            opt.num_classes = 22
            if model_folder:find('noiseunlabelled') or model_folder:find('noisesym') then 
                num_predicted_classes = 22 
            elseif model_folder:find('noisefull') then
                num_predicted_classes = 26
            end
        end
    end
    return  num_predicted_classes
end

local function load_model(model_filename)
    return torch.load(model_filename)
end

function evaluate_model(model_folder, iteration, dataset, dataset_role, set_name, redo)
    --Dataset: amazon, webcam, dslr, UCSF, Caie or UTSW
    --Dataset_role: source or target
    --Set_name: train or test or val

    if dataset then
        local model_filename = path.join(final_result_folder, model_folder, string.format('model_%s.bin', tostring(iteration)) )
        if not paths.filep(model_filename) then return end

        local output_folder = path.join(final_result_folder, model_folder, 'full_features')
        local output_filename = path.join(output_folder, string.format('full_features_%s_%s_%s.csv',
                                                    tostring(iteration), dataset, set_name))

        if redo or not paths.filep(output_filename) then
            if not paths.dirp(output_folder) then paths.mkdir(output_folder) end

            local label_filename = path.join(output_folder, string.format('labels_%s_%s_%s.csv', tostring(iteration), dataset, set_name))
            local label_file = io.open(label_filename, 'a')

            opt.target_supervision_level= (model_folder:find('unsup') and 0) or (model_folder:find('supervised') and 100) or 50
            require 'cudnn'
            cudnn.fastest=true; cudnn.benchmark =true
            opt.model=(model_folder:find('MADA') and 'MADA') or (model_folder:find('AlexNet') and 'AlexNet') or (model_folder:find('VGG') and 'VGG') or -1
            ----------------------------------------
            ----------------------------------------
            --Loading model first, because some are corrupted
            --
            local model = load_model(model_filename)
            ----------------------------------------
            ----------------------------------------
            --Parameters
            --
            opt.fold = utils.get_fold(model_folder)
            opt.fully_transductive = not model_folder:find('nontransductive')
            opt.difficulty_level = model_folder:find('hard') and 'hard' or 'easy'
            utils.setSeed(opt.seed)
            utils.setClasses(opt)

            if opt.dataset=='OFFICE' then
                currimage_folder = office_folder
            elseif opt.dataset=='BIO' then
                --No need to set nchannelsin as we are not using this parameter. Normalization is achieved in utils.get_normalization_func
                opt.num_classes = opt.num_common_classes
                currimage_folder = path.join(image_folder, 'Cells_div4')
            end
            local num_predicted_classes = set_num_predicted_classes(model_folder)

            print(paths.basename(model_folder), dataset, dataset_role, set_name, ' fold', opt.fold, 'iteration ', iteration)
            print(opt.fully_transductive, opt.target_supervision_level, opt.model, opt.difficulty_level)

            ----------------------------------------
            ----------------------------------------
            --Pre-allocating space for Tensors on CPU and GPU
            --
            local s = torch.LongStorage({opt.num_patch_per_im, 3, opt.inputsize, opt.inputsize})
            local mb_cpu = torch.Tensor(s)
            local mb_gpu = torch.CudaTensor(s)

            ----------------------------------------
            ----------------------------------------
            --Loading image filename iterator
            --
            local class_images,total_number_images = utils.get_class_images(dataset, dataset_role, set_name, model_folder)
            if opt.model=='MADA' and opt.target_supervision_level == 0 then
                local comp_images= utils.get_class_images(dataset, dataset_role, 'test', model_folder)
                for class, image_list in pairs(comp_images) do
                   if not class_images[class] then class_images[class] = {} end
                   for _,image in ipairs(image_list) do
                       table.insert(class_images[class], image)
                   end
                end
            end

            --Filtering image list if some unlabelled images were used while training
            if dataset_role=='target' and set_name=='test' and not opt.fully_transductive then
                print('Non transductive, filtering target test')
                local num_images = 10
                if opt.target_supervision_level==100 then num_images=7 end

                local new_class_images = {}
                for class, image_list in pairs(class_images) do

                    if not new_class_images[class] then new_class_images[class]={} end
                    for i, image in ipairs(image_list) do
                        if i>num_images then
                            table.insert(new_class_images[class], image)
                        end
                    end
                end
                class_images = new_class_images
            end
            ----------------------------------------
            ----------------------------------------
            --Loading image normalization and reshape function
            --
            local normalization_func = utils.get_normalization_func(dataset, model_folder, opt)

            ----------------------------------------
            ----------------------------------------
            --Going for image per image classification
            --
            local result_confusion_matrix = torch.zeros(opt.num_classes, num_predicted_classes)
            local num_saved_points = dataset=='webcam' and 10 or 2
            local saved_features = torch.Tensor(total_number_images*num_saved_points, opt.bottleneck_size)
            local done_image = 0

            for class, image_list in pairs(class_images) do
                local true_label = opt.classes[class]
                if opt.dataset=='OFFICE' or (true_label and true_label<= opt.num_classes) then
                    for i,image_name in ipairs(image_list) do
                        local im = image.load(path.join(currimage_folder, image_name))
                        --Careful, here we need eventually to swap color dimensions (in comparison w/Python) AND normalize
                        im = normalization_func(im, path.join(currimage_folder, image_name))
                        for j=1,opt.num_patch_per_im do
                            mb_cpu[j]=general_data.cropImage(im, opt.inputsize, opt.inputsize)
                        end
                        --Transfering to GPU
                        mb_gpu:copy(mb_cpu)
                        --Forward prop through model
                        local features = model.feature_extractor:forward(mb_gpu)
                        local predictions = model.label_predictor:forward(features):mean(1)[1]
                        local _, predicted_label = torch.max(predictions, 1)

                        if true_label and true_label<= opt.num_classes then
                            result_confusion_matrix[{{true_label}, {predicted_label[1]}}] = 1+
                                            result_confusion_matrix[{{true_label}, {predicted_label[1]}}]
                        end

                        saved_features[{{1+done_image*num_saved_points, (done_image+1)*num_saved_points}}]:copy(features[{{1,num_saved_points}}])
                        for k=1,num_saved_points do label_file:write(class ..'\n') end

                        done_image = done_image +1
                        --Clearing state
                        model:clearState()
                    end
                end
            end
            print(result_confusion_matrix)
            print('Global acc \t\t', torch.sum(torch.diag(result_confusion_matrix))/torch.sum(result_confusion_matrix))
            local m = 0
            local c = 0
            for i=1,opt.num_classes do
                if torch.sum(result_confusion_matrix[i])>0 then
                    m = m + result_confusion_matrix[i][i]/torch.sum(result_confusion_matrix[i])
                    c = c +1
                end
            end
            print('Avg acc \t\t',m/c)
            ----------------------------------------
            ----------------------------------------
            --Saving result per image
            --
            csvigo.save({data = torch.totable(result_confusion_matrix), path=output_filename, verbose=false})
            --Saving features for tSNE plots
            output_filename = path.join(output_folder, string.format('features_%s_%s_%s.csv', tostring(iteration), dataset, set_name))
            csvigo.save({data = torch.totable(saved_features[{{1,done_image*num_saved_points}}]), path= output_filename, verbose=false})
            label_file:close()

        end

        if (opt.dataset=='BIO' or model_folder:find('noise')) and opt.target_supervision_level==50 and dataset_role=='target' and set_name~='train' and string.sub(set_name, 1, 4)~='rest' then
            evaluate_model(model_folder, iteration, dataset, dataset_role, 'rest'..set_name, redo)
        end
    end
    return
end
