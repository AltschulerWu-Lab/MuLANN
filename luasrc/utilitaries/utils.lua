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

require 'csvigo'
require 'pl'
hasCuda, cuda = pcall(require, 'cutorch')

local test_inputs = hasCuda and torch.CudaTensor() or torch.Tensor()
local test_outputs = hasCuda and torch.CudaTensor() or torch.Tensor()

local function fill_in_confusion_matrix(true_labels, label_softmax_predictions, confusion)
    local testing_err = 0
    for i=1, label_softmax_predictions:size(1) do
        local currPred = label_softmax_predictions[i]
        _, currPred = torch.max(currPred, 1)
        currPred = currPred[1]
        confusion[{{ true_labels[i]},{currPred}}]=confusion[{{ true_labels[i]},{currPred}}]+1
        if currPred~= true_labels[i] then
            testing_err=testing_err+1
        end
    end
    return testing_err
end

local function batch_error(dataset, module, true_domain_label, num_predicted_classes, testingBatch, num_runs)
    local num_predicted_classes = num_predicted_classes or opt.num_classes
    local testing_err=0
    local confusion = torch.Tensor(opt.num_classes, num_predicted_classes):fill(0)
    local testingBatch = testingBatch or opt.testingBatch
    local num_runs = num_runs or 10
    local domain_err = 0
    if dataset then
        local test_batchsize = testingBatch/num_runs

        for k=1, num_runs do
            local testingInputs, testingLabels = dataset:getBatch(test_batchsize)
            test_inputs:resize(testingInputs:size()):copy(testingInputs)
            local features = module.feature_extractor:forward(test_inputs)
            local label_softmax_predictions = module.label_predictor:forward(features)

            --Computing domain error
            if (opt.domain_method=='DANN' or opt.domain_method=='MuLANN') and module.domain_predictor then
                local domain_labels = module.domain_predictor:forward(features)
                if opt.extra_domain then
                    _, domain_labels = domain_labels:max(2)
                else domain_labels = domain_labels:gt(0.5) end
                
                domain_err = domain_err + domain_labels:ne(true_domain_label):sum()
            end

            testing_err = testing_err + fill_in_confusion_matrix(testingLabels, label_softmax_predictions, confusion)

            module:clearState()
        end
        testing_err = testing_err/testingBatch
    end
    return testing_err, confusion, domain_err/testingBatch
end

local function full_test_error(dataset, module, true_domain_label, set_name, num_runs)
    --In that case I'm computing the full test error directly
    local test_err, resttest_err=0, 0
    local confusion = torch.Tensor(opt.num_classes, opt.num_classes):fill(0)
    local domain_err = 0
    local count = torch.zeros(2)
    if dataset then
        --Data is of shape 10,000x3x32x32 for MNIST and 26,032x3x32x32 for SVHN
        local data, labels = dataset:getAllData()
        local num_runs = num_runs or (opt.dataset=='SIGNS' and 20 or 10)
        local test_batchsize = math.ceil(data:size(1)/num_runs)
        for k=1,num_runs do
            local end_ = math.min(data:size(1), k*test_batchsize)
            local testingInputs = data[{{1 + (k-1)*test_batchsize, end_}}]
            test_inputs:resize(testingInputs:size()):copy(testingInputs)

            local testingLabels = labels[{{1 + (k-1)*test_batchsize, end_}}]
            local features = module.feature_extractor:forward(test_inputs)
            local label_softmax_predictions = module.label_predictor:forward(features)

            if module.domain_predictor then
                local domain_labels = module.domain_predictor:forward(features):gt(0.5)
                domain_err = domain_err + domain_labels:ne(true_domain_label):sum()
            end
            fill_in_confusion_matrix(testingLabels, label_softmax_predictions, confusion)

            module.feature_extractor:clearState()
            if module.domain_predictor then module.domain_predictor:clearState() end
        end
        if set_name=='target' then
            for k=1,opt.num_classes do
                --We're looking at test data so unlabelled classes are the ones which are counted as present
                --Was swapped before
                if dataset._classes:eq(k):sum()==0 then
                    resttest_err = resttest_err + confusion[k][k]
                    count[1] = count[1]+ confusion[k]:sum()
                else
                    test_err = test_err + confusion[k][k]
                    count[2] = count[2]+ confusion[k]:sum()
                end
            end
            --Inverting bc these are actually accuracies
            resttest_err = 1- resttest_err/count[1]
            test_err = 1- test_err/count[2]
            domain_err = domain_err/count:sum()
        else
            test_err = 1- confusion:diag():sum()/confusion:sum()
            domain_err = domain_err/confusion:sum()
        end
    end
    return test_err, confusion, domain_err, resttest_err
end

function get_all_errors(trainingsourcedata, testsourcedata, valsourcedata,
                        trainingtargetdata, testtargetdata, valtargetdata,
                        target_supervision_level,
                        module, outputfolder, timepoint)
    module:evaluate()
    local target_dom_label = opt.extra_domain and 3 or 0
    local val_source_dom_label = opt.extra_domain and 2 or 1
    local val_target_dom_label = opt.extra_domain and 2 or 0

    local domain_error_result = ''
    local msg = 'Source train error '
    local s_trainacc, s_trainerr, s_domain_err = batch_error(trainingsourcedata, module, 1)
    msg = msg.. s_trainacc
    domain_error_result = domain_error_result ..' '..s_domain_err

    local t_trainacc, t_trainerr, t_domain_err = batch_error(trainingtargetdata, module, target_dom_label)
    msg = msg.. ' Target train error '.. t_trainacc
    domain_error_result = domain_error_result ..' '..t_domain_err

    local s_testacc, s_testerr, s_domain_err
    if opt.dataset=='DIGITS' or opt.dataset=='SIGNS' then
        s_testacc, s_testerr, s_domain_err = full_test_error(testsourcedata, module, 1, 'source')
    else
        s_testacc, s_testerr, s_domain_err = batch_error(testsourcedata, module, 1)
    end
    msg = msg .. ' Source test error '.. s_testacc
    domain_error_result = domain_error_result ..' '..s_domain_err

    local t_testacc, t_testerr, t_domain_err, t_resttestacc
    if opt.dataset=='DIGITS' or opt.dataset=='SIGNS' then
        t_testacc, t_testerr, t_domain_err, t_resttestacc = full_test_error(testtargetdata, module, 0, 'target')
        local file = io.open(path.join(outputfolder, 't_Resttest_err.txt'), 'a')
        file:write(t_resttestacc ..'\n')
        file:close()
    else
        t_testacc, t_testerr, t_domain_err = batch_error(testtargetdata, module, target_dom_label)
    end
    msg = msg.. ' Target test error ' .. t_testacc
    domain_error_result = domain_error_result ..' '..t_domain_err

    local s_validacc, s_validErr, s_domain_err = batch_error(valsourcedata, module, val_source_dom_label)
    msg = msg.. ' Source validation error ' .. s_validacc
    domain_error_result = domain_error_result ..' '..s_domain_err

    local t_validacc, t_validErr, t_domain_err = batch_error(valtargetdata, module, val_target_dom_label)
    msg = msg.. ' Target validation error ' .. t_validacc
    domain_error_result = domain_error_result ..' '..t_domain_err

    print(msg)
    module.feature_extractor:clearState()
    module.label_predictor:clearState()
    module:training()
    utils.write_error_output(outputfolder, s_trainerr, t_trainerr,
                                            s_testerr, t_testerr,
                                            s_validErr, t_validErr)

    --Also writing binary error for domain classification. Remember we want this to be as close to 0.5 as possible
    if (opt.domain_method=='DANN' or opt.domain_method=='MuLANN') and module.domain_predictor then
        module.domain_predictor:clearState()
        print(domain_error_result)
        local file = io.open(path.join(outputfolder, 'Domain_err.txt'), 'a')
        file:write(domain_error_result ..'\n')
        file:close()
    end
    return {s_trainacc, t_trainacc,
            s_testacc, t_testacc,
            s_validacc, t_validacc}
end

local function compute_domain_error(dataset, module, true_domain_label, num_runs, AE_model)
    local num_runs = num_runs or 8
    local test_batchsize = opt.testingBatch/num_runs
    local domain_err = 0

    for k=1, num_runs do
        local testingInputs, _ = dataset:getBatch(test_batchsize)
        if AE_model then testingInputs = AE_model.feature_extractor:forward(testingInputs) end
        test_inputs:resize(testingInputs:size()):copy(testingInputs)

        local features = module.feature_extractor:forward(test_inputs)
        local domain_labels = module.label_predictor:forward(features):gt(0.5)
        domain_err = domain_err + domain_labels:ne(true_domain_label):sum()

    end
    return domain_err/opt.testingBatch
end

local function count_labels(labels, select)
    local result = {}
    for i=1, labels:size(1) do
        local label = labels[i]
        if not result[label] then result[label]=1
        else result[label] = result[label] +1
        end
    end
    if select then
        local cases = {1, 15, 16, 22, 31 }
        for _,i in ipairs(cases) do
            print(i, result[i])
        end
    else
        for i=1, opt.num_classes do
            print(i, result[i])
        end
    end
end

local function read_DMSO_values(plate, dataset)
    local file = path.join(metadata_dir,string.format('%s_info', dataset), string.format('%s_DMSO_values_paper', dataset), string.format('%s.txt', plate))
    local reader = csvigo.load({path=file, header=false, separator = ' ',verbose=false})

    local DMSO_mean = torch.zeros(3)--:double() it's double by default
    local perc1 = torch.zeros(3)--:double() it's double by default
    local perc99 = torch.zeros(3)--:double() it's double by default
    -- now very careful because Python is BGR and Torch is RGB

    DMSO_mean[1]=tonumber(reader['var_2'][3])
    DMSO_mean[3]=tonumber(reader['var_2'][1])
    DMSO_mean[2]=tonumber(reader['var_2'][2])

    if pcall(function()
    perc1[1]=tonumber(reader['var_3'][3])
    perc1[3]=tonumber(reader['var_3'][1])
    perc1[2]=tonumber(reader['var_3'][2])

    perc99[1]=tonumber(reader['var_4'][3])
    perc99[3]=tonumber(reader['var_4'][1])
    perc99[2]=tonumber(reader['var_4'][2])
    end) then end

    return DMSO_mean, perc1, perc99
end

local function get_plate(image_name)
    return paths.basename(paths.dirname(paths.dirname(paths.dirname(image_name))))
end

function write_stop_reason(outputfolder, text)
    local file = io.open(path.join(outputfolder, 'Reason.txt'), 'w')
    file:write(text ..'\n')
    file:close()
end

function string:split(sSeparator, nMax, bRegexp)
   assert(sSeparator ~= '')
   assert(nMax == nil or nMax >= 1)

   local aRecord = {}

   if self:len() > 0 then
      local bPlain = not bRegexp
      nMax = nMax or -1

      local nField, nStart = 1, 1
      local nFirst,nLast = self:find(sSeparator, nStart, bPlain)
      while nFirst and nMax ~= 0 do
         aRecord[nField] = self:sub(nStart, nFirst-1)
         nField = nField+1
         nStart = nLast+1
         nFirst,nLast = self:find(sSeparator, nStart, bPlain)
         nMax = nMax-1
      end
      aRecord[nField] = self:sub(nStart)
   end

   return aRecord
end
local function get_fold(folder)
    local l = string.split(folder, '_')
    if l[#l]:find('z') then
        return l[#l-5]
    else
        return l[#l-4]
    end
end

local function setClasses(opt)
    --Updated with correct phenoactivity for Caie
    if opt.target =='Caie' or opt.source=='Caie' then
        opt.num_classes = 14
        opt.num_common_classes = 7
        opt.classes = {--common drugs
                       ["DNA"]=1, ["HDAC"]=2, ["MT"]=3, 
                       ["Proteasome"]=4, ["Actin"]=5,
                       ["ER"]=6, ["Aurora"]=7,
                       --JK drugs
                       ["mTOR"]=8,
                       ["Hsp90"]=9, ["PLK"]=10,
            --Caie drugs
            ["Eg5inhibitor"]=11,
            ["Kinase"]=12,
            ["Proteinsynthesis"]=13,

            ["Inactive"]=14
        }

        opt.reverse_classes = {
            [1]="DNA", [2]="HDAC",[3]="MT",
            [4]="Proteasome", [5]="Actin",
            [6]="ER",[7]="Aurora",

            [8]="mTOR", [9]="Hsp90", [10]='PLK',

            [11]="Eg5inhibitor",
            [12]="Kinase",
            [13]="Proteinsynthesis",

            [14]='Inactive'
        }

    elseif opt.dataset=='DIGITS' then
        opt.num_common_classes = 10
        opt.num_classes = 10
        opt.classes = torch.totable(torch.range(1, 10))
        opt.reverse_classes = torch.totable(torch.range(1, 10))

    elseif opt.source =='amazon' or opt.source=='webcam' or opt.source=='dslr' then
        opt.num_common_classes = 31
        opt.num_classes = 31
        local class_dir = paths.iterdirs(path.join(office_folder, 'amazon'))
        opt.classes = {['back_pack']=1,
                        ['bike']=2,
                        ['bike_helmet']=3,
                        ['bookcase']=4,
                        ['bottle']=5,
                        ['calculator']=6,
                        ['desk_chair']=7,
                        ['desk_lamp']=8,
                        ['desktop_computer']=9,
                        ['file_cabinet']=10,
                        ['headphones']=11,
                        ['keyboard']=12,
                        ['laptop_computer']=13,
                        ['letter_tray']=14,
                        ['mobile_phone']=15,
                        ['monitor']=16,
                        ['mouse']=17,
                        ['mug']=18,
                        ['paper_notebook']=19,
                        ['pen']=20,
                        ['phone']=21,
                        ['printer']=22,
                        ['projector']=23,
                        ['punchers']=24,
                        ['ring_binder']=25,
                        ['ruler']=26,
                        ['scissors']=27,
                        ['speaker']=28,
                        ['stapler']=29,
                        ['tape_dispenser']=30,
                        ['trash_can']=31
        }
        opt.reverse_classes = {[1]='back_pack',
                        [2]='bike',
                        [3]='bike_helmet',
                        [4]='bookcase',
                        [5]='bottle',
                        [6]='calculator',
                        [7]='desk_chair',
                        [8]='desk_lamp',
                        [9]='desktop_computer',
                        [10]='file_cabinet',
                        [11]='headphones',
                        [12]='keyboard',
                        [13]='laptop_computer',
                        [14]='letter_tray',
                        [15]='mobile_phone',
                        [16]='monitor',
                        [17]='mouse',
                        [18]='mug',
                        [19]='paper_notebook',
                        [20]='pen',
                        [21]='phone',
                        [22]='printer',
                        [23]='projector',
                        [24]='punchers',
                        [25]='ring_binder',
                        [26]='ruler',
                        [27]='scissors',
                        [28]='speaker',
                        [29]='stapler',
                        [30]='tape_dispenser',
                        [31]='trash_can',
}
    else
        print('What dataset are you looking for')
        print(a+you)
    end

    opt.inactive_label = opt.classes["Inactive"]
end

local function set_unlabelled_noise_classes(opt)
    --Updated with correct phenoactivity for Caie
    if opt.target =='Caie' then
        opt.classes = {--common drugs
                       ["DNA"]=1, ["HDAC"]=2, ["MT"]=3,
                       ["Proteasome"]=4, ["Actin"]=5,
                       ["ER"]=6, ["Aurora"]=7,
                       --JK drugs
                       ["mTOR"]=11,
                       ["Hsp90"]=12, ["PLK"]=13,
            --Caie drugs
            ["Eg5inhibitor"]=8,
            ["Kinase"]=9,
            ["Proteinsynthesis"]=10,

            ["Inactive"]=14
        }

        opt.reverse_classes = {
            [1]="DNA", [2]="HDAC",[3]="MT",
            [4]="Proteasome", [5]="Actin",
            [6]="ER",[7]="Aurora",

            [11]="mTOR", [12]="Hsp90", [13]='PLK',

            [8]="Eg5inhibitor",
            [9]="Kinase",
            [10]="Proteinsynthesis",

            [14]='Inactive'
        }
    else
        print("How did you get here")
        print(a+youpi)
    end

end


local function setSeed(seed)
   torch.setnumthreads(num_threads)
   math.randomseed(seed)
   torch.manualSeed(seed)
   if cutorch then
       cutorch.manualSeed(seed)
   end
end

local function get_class_images(dataset, dataset_role, set_name, noise_images)
    local image_d = {}
    local comp_str = ''
    if string.sub(set_name, 1, 4)=='rest' then
        comp_str = 'rest'
        set_name = string.sub(set_name, 5)
    end
    local count= 0
    if opt.dataset=='OFFICE' then
        --Loading train/test images from the metadata folder
        if dataset_role=='source' then
            comp_str = ''
        else
            comp_str = (opt.target_supervision_level==50 and comp_str..'diff_category-') or 'same_category-'
        end

        local file = path.join(metadata_dir,
                            string.format(office_casepicking_folder, dataset, dataset_role, set_name),
                            string.format(office_casepicking_file, comp_str, opt.fold))
        print('Loading ', file)
        local image_list = csvigo.load({path =file, verbose=false, header=false})
        for _,image in ipairs(image_list['var_1']) do
            local filename = image:split(' ')[1]
            local class = paths.basename(paths.dirname(filename))
            if not image_d[class] then image_d[class] = {} end
            table.insert(image_d[class], path.join(dataset, class, paths.basename(filename)))
            count = count+1
        end

    elseif opt.dataset=='DIGITS' then
        local filename = path.join(office_folder, dataset, string.format('%s_%sx%s.t7', set_name, opt.inputsize, opt.inputsize))
        print('Loading ', filename)
        local loaded = torch.load(filename, 'ascii')
        filename = path.join(office_folder, dataset, string.format('%s_indices.t7', set_name))
        local indices = torch.load(filename)

        local loadedData
        if dataset=='mnistm' then
            loadedData = {
            --Already in the right format as I saved it using Torch
                data = loaded.X,
                labels = loaded.y,
                indices = indices
            }
        elseif dataset=='mnist' then
            loadedData = {
                --Repeating as if color image for MNIST
                data = loaded.data:double():repeatTensor(1,3,1,1) ,
                labels = loaded.new_labels,
                indices = indices
            }
        else
            print('Which dataset are you looking for?')
            print(a+youpi)
        end
        --Now we're returning some data that's of shape num x 3 x 32 x 32
        return loadedData

    elseif opt.dataset=='BIO' then
        --Loading train/test images from the metadata folder
        if dataset_role=='source' then
            comp_str = ''
        else
            if opt.target_supervision_level==100 then
                comp_str = 'same_category-' 
            elseif opt.target_supervision_level==50 then
                comp_str = comp_str .. ((opt.difficulty_level=='easy' and 'diffeasy_category-')
                                        or (opt.difficulty_level=='hard' and 'diffhard_category-')
                                        or '')
            end
        end

        local file = path.join(metadata_dir,
                            string.format(bio_casepicking_folder, dataset, dataset_role, set_name),
                            string.format(office_casepicking_file, comp_str, opt.fold))
        print('Loading ', file)
        local image_list = csvigo.load({path =file, verbose=false, header=false})
        for _,filename in ipairs(image_list['var_1']) do
            local class = not filename:find('Aurora') and filename:split('/')[2] or 'Aurora'
            if not image_d[class] then image_d[class] = {} end
            table.insert(image_d[class], filename)
            count = count+1
        end
        if noise_images==true and (opt.noise=='unlabelled' or opt.noise=='full') and dataset_role=='target' and set_name=='test' then
           --We are going to add the Eg5inhibitor, Kinase and Proteinsynthesis images
            file = path.join(metadata_dir,
                            string.format(bio_casepicking_folder, dataset, dataset_role, set_name),
                            string.format(office_casepicking_file, 'rest'..comp_str, opt.fold))
            print('ADDITIONAL NOISE DATA ', file)
            image_list = csvigo.load({path =file, verbose=false, header=false})
            for _,filename in ipairs(image_list['var_1']) do
                local class = not filename:find('Aurora') and filename:split('/')[2] or 'Aurora'
                if class=='Eg5inhibitor' or class=='Kinase' or class=='Proteinsynthesis' then
                    if not image_d[class] then image_d[class] = {} end
                    table.insert(image_d[class], filename)
                    count = count+1
                end
                if opt.noise_type==1 then
                    print('WTF')
                    print(a+youpi)
                    image_d[opt.reverse_classes[9]]=image_d[opt.reverse_classes[8]]
                    image_d[opt.reverse_classes[10]]=image_d[opt.reverse_classes[8]]
                end
            end
        end
    else
        print('What dataset are you looking for?')
        print(a+youpi)
    end
    return image_d, count
end

local function get_source_target(folder)
    local source, target, extra
    if folder:find('amazonwebcam') then
       source = 'amazon'
       target = 'webcam'
    elseif folder:find('webcamamazon') then
       source = 'webcam'
       target = 'amazon'
    elseif folder:find('dslrwebcam') then
       source = 'dslr'
       target = 'webcam'
    elseif folder:find('webcamdslr') then
       source = 'webcam'
       target = 'dslr'
    elseif folder:find('dslramazon') then
       source = 'dslr'
       target = 'amazon'
    elseif folder:find('amazondslr') then
       source = 'amazon'
       target = 'dslr'
    elseif folder:find('mnist_mnistm') then
       source = 'mnist'
       target = 'mnistm'
    elseif folder:find('mnistm_mnist') then
       source = 'mnistm'
       target = 'mnist'
    elseif folder:find('UCSF_UTSW') then
       source = 'UCSF'
       target = 'UTSW'
    elseif folder:find('UTSW_Caie') then
       source = 'UTSW'
       target = 'Caie'
    elseif folder:find('Caie_UCSF') then
       source = 'Caie'
       target = 'UCSF'
    elseif folder:find('both_Caie') then
       source ='UTSW'
       extra = 'UCSF'
       target ='Caie'
    else
       print('Domaines inconnus') print(a+youpi)
    end
    if opt.target=='Caie' or opt.source=='Caie' then
        opt.nchannelsin = 2
    end

    opt.source = source
    opt.target = target
    opt.extra_domain = extra
    return
end

local function get_normalization_func(dataset, folder, opt)
    if folder:find('office') then
        --Office normalization function
        local mean = general_data.load_imagenet_mean(opt)
        local function norm(image)
            image:mul(255)
            image=image:index(1,torch.LongTensor{3,2,1})
            image:add(-mean)
            return image
        end
        return norm
    end
    --Biological dataset normalization function
    local function norm(image, image_name)
        local plate = utils.get_plate(image_name)
        local mean_, _, _= utils.read_DMSO_values(plate, dataset)
        utils.AlexNet_image_normalization(image, mean_)
        if folder:find('Caie') then
            --This means that we need to re-align the channels between Caie and UTSW
            if dataset=='Caie' then
                image[3]:fill(0)
                image = torch.cat({image[1]:reshape(1, image:size(2), image:size(3)),
                    image[3]:reshape(1, image:size(2), image:size(3)),
                    image[2]:reshape(1, image:size(2), image:size(3))}, 1)
            else
                image[2]:fill(0)
            end
        end
        return image
    end
    return norm
end

local function save_net(t, module)
    print('Starting the process of saving nnet')
    module:clearState()
-- I save the model in evaluation mode, ie for example dropout layers will be identity
    module:evaluate()
    torch.save(opt.outputfolder .. '/'.. modelfilename ..'_' .. t .. '.bin', module)
    print('Saved nnet')
    module:training()
end

function write_error_output(outputfolder, s_train_error, t_train_error,
                                s_test_error,t_test_error,
                                s_valid_error, t_valid_error)

    local file = io.open(path.join(outputfolder, 'Trainerr.txt'), 'a')
    local error = s_train_error:__tostring()
    if t_train_error then error = error .. ' '.. t_train_error:__tostring() end
    file:write(error ..'\n')
    file:close()

    file = io.open(path.join(outputfolder, 'Testerr.txt'), 'a')
    error = s_test_error:__tostring()
    if t_test_error then error = error .. ' '.. t_test_error:__tostring() end
    file:write(error ..'\n')
    file:close()

    file = io.open(path.join(outputfolder, 'Validerr.txt'), 'a')
    error = s_valid_error:__tostring() .. ' '.. t_valid_error:__tostring()
    file:write(error ..'\n')
    file:close()

end

function norm_l2(vec)
   return math.sqrt(torch.sum(vec:clone():pow(2)))
end

local function AlexNet_image_normalization(im, DMSO_m)
    for i=1, 3 do
        im[i]:add(-DMSO_m[i])
    end

    im:mul(255)
    return
end

local function return_train_parameters(opt)
    local train_parameters = {}

--i. Getting architecture settings
    local architecture =(opt.model=='AlexNet' and 'AlexNet') or opt.architecture

    if architecture==3 then
        print('Pb, what are you trying to do?')
--        train_parameters = return_train_parameters_arch3(opt.architecture_setting)
    end

    train_parameters['loss']= opt.loss
    train_parameters['num_disc_neurons'] = opt.num_disc_neurons
    train_parameters['bottleneck_size'] = opt.bottleneck_size

    train_parameters['finetuning']=opt.finetuning
    train_parameters['architecture']=architecture

--ii. Getting training settings
    opt['optimizer_func']='sgd'
    if opt.train_setting ==0 then
        opt['lambda_func']= 'fixed'
    elseif opt.train_setting == 1 then
        opt['lambda_func']= 'schedule'
        opt['lambda_maxiter'] = 10000

    else
        print('What setting are you getting at')
        print(a+u)
    end

    if opt.domain_method=='MADA' then
        --Reproducing paper values
        opt.gamma = 0.0003
    end

    return train_parameters
end

local function prepare_for_learning(outputfolder, opt,
                                        shuffling_iter)
   --SETTING SEEDS
    print("Setting random seed for CPUs and GPUs, setting num threads on CPUs")
    setSeed(opt.seed)

   --SETTING CLASS NAMES LABELS AND NUMBER
    setClasses(opt)

    -- INPUT FOLDER SETTING
    local inputfolder = path.join(image_folder, 'Cells_div4')

    --OUTPUT FOLDER SETTING
    local outputfolder = outputfolder
    outputfolder = outputfolder ..'_'.. shuffling_iter

    if opt.train_setting then
        local comp_str
        if not opt.domain_adaptation and not opt.incl_source then
            comp_str = 'falseNoSource'
        else
            comp_str = tostring(opt.domain_adaptation)
        end
        outputfolder = outputfolder  .. "_D"..comp_str.."_e" .. opt.eta0..
                                        '_l'..opt.domainLambda.. '_s'..opt.train_setting
        if opt.zeta then outputfolder = outputfolder..'_z'..opt.zeta end
    end

    if not redo and paths.dirp(path.join(final_result_folder, path.basename(outputfolder))) then
        print('Run done already, stopping here')
        return -1 ,-1
    end

    if not paths.dirp(outputfolder) then
        paths.mkdir(outputfolder)
    else
        if paths.rmall(outputfolder, 'yes') then
            paths.mkdir(outputfolder)
        end
    end

    return inputfolder, outputfolder
end



local utils = {
    setSeed =setSeed,
    setClasses = setClasses,
    set_unlabelled_noise_classes = set_unlabelled_noise_classes,
    prepare_for_learning = prepare_for_learning,
    AlexNet_image_normalization = AlexNet_image_normalization,
    return_train_parameters = return_train_parameters,

    save_net = save_net,
    get_plate = get_plate,
    read_DMSO_values = read_DMSO_values,
    compute_error = batch_error,
    compute_full_error = full_test_error,

    get_fold = get_fold,
    get_class_images=get_class_images,
    get_normalization_func = get_normalization_func,
    fill_in_confusion_matrix = fill_in_confusion_matrix,
    get_source_target = get_source_target,
    count_labels = count_labels,

    write_error_output = write_error_output,
    write_stop_reason = write_stop_reason,
}
return utils
