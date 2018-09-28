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

    if opt.dataset=='BIO' then
        opt.num_classes = opt.num_common_classes
        if opt.noise then opt.num_classes = 10 end
        if opt.noise=='full' then opt.num_classes = 13 end
        if opt.noise=='unlabelled' and opt.target=='Caie' then
            --This will switch JK and Caie classes BEFORE setting up the data getters
            utils.set_unlabelled_noise_classes(opt)
        end
    elseif opt.dataset=='OFFICE' then
        if opt.noise=='symm' then opt.num_classes = 22 end
        if opt.noise=='full' then opt.num_classes = 30 end
    end

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
        if opt.noise=='unlabelled' then opt.num_classes = opt.num_common_classes end
        if opt.noise=='full' then opt.num_classes = 10 end

        maxiter = 40002
        plotinterval =1000
        statinterval =10000

    elseif opt.dataset=='OFFICE' then
        if opt.noise=='unlabelled' then opt.num_classes = 22 end
        if opt.noise=='full' then opt.num_classes = 26 end

        maxiter = 15002 -- MADA: 10k iter
        plotinterval =1000
        statinterval =5000

    end

    --ii. Load model and start training
    opt.num_classwise_disc = opt.num_classes
    model = model_level.getmodel(opt.domain_method, train_parameters)
    print(model.feature_extractor, model.label_predictor, model.domain_predictor)

    if opt.dataset=='BIO' then
        if opt.noise=='full' then opt.num_classes = 13 end
    elseif opt.dataset=='OFFICE' then
        if opt.noise=='unlabelled' then opt.num_classes = 31 end
        if opt.noise=='full' then opt.num_classes = 30 end
    end

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
