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


local function evaluate(net)
    if net then
        local numModules = #net.modules
        for i=1,numModules do
            pcall(function () net.modules[i]:evaluate() return 1 end)
        end
    end
end

local function training(net)
    if net then
        local numModules = #net.modules
        for i=1,numModules do
            pcall(function () net.modules[i]:training() return 1 end)
        end
    end
end

local function weight_init(module, gaussian_std, bias_cst, gaussian_mean)
    --Init like in DANN code see https://github.com/ddtm/caffe/blob/grl/examples/adaptation/protos/train_val.prototxt
    local gaussian_mean = gaussian_mean or 0
    print('Init ', module)
    module.weight:normal(gaussian_mean, gaussian_std)
    module.bias:fill(bias_cst) -- Conv layers do have biases
end

local function dann_domain_predictor_weight_init(domain_predictor, index)
    weight_init(domain_predictor.modules[2-index], 0.01, 0)
    weight_init(domain_predictor.modules[5-index], 0.01, 0)
    weight_init(domain_predictor.modules[8-index], 0.3, 0)
end

local function tf_variance_initializer_fan_in(m, scale)
    if m.weight then
        --Doing fan_in mode corresponding to https://www.tensorflow.org/api_docs/python/tf/variance_scaling_initializer
        print('Init ', m)
        local scale = scale or 1
        local num_inputs 
        if not pcall(function() num_inputs = m.weight:size(2) end) then num_inputs = m.weight:size(1) end
        local std = math.sqrt(scale/num_inputs)
        m.weight:normal(0, std)
        m.bias:fill(0)
    end
end

local function fc_init(m, bias_cst)
    local u = math.sqrt(6/(m.weight:size(1)+m.weight:size(2)))
    m.weight:uniform(-u, u)
    m.bias:fill(bias_cst)
end

local function conv_init(m)
    m.weight:normal(0, 0.02)
    m.bias:fill(0) -- Conv layers do have biases
end

local function initialize_small_net(f_extractor, l_predictor)
    print('Initializing small network')
    conv_init(f_extractor.modules[1])
    conv_init(f_extractor.modules[4])
    fc_init(l_predictor.modules[1], 0)
    fc_init(l_predictor.modules[3], 0)
    fc_init(l_predictor.modules[5], 1/opt.num_classes)
    return
end

local function small_net(DA_method)
    local kernel_size = 5
    local num_dim = 1200
    local feature_extractor = nn.Sequential()
    --Reproducing architecture from DANN
    --No padding, stride of 1
    feature_extractor:add(nn.SpatialConvolution(opt.nchannelsin, 32, kernel_size, kernel_size, 1, 1, 0, 0))
    feature_extractor:add(nn.ReLU(true))
    feature_extractor:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    feature_extractor:add(nn.SpatialConvolution(32, 48, kernel_size, kernel_size, 1, 1, 0, 0))
    feature_extractor:add(nn.ReLU(true))
    feature_extractor:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    feature_extractor:add(nn.View(-1):setNumInputDims(3))

    local label_predictor = nn.Sequential()
    label_predictor:add(nn.Linear(num_dim, 100))
    label_predictor:add(nn.ReLU(true))

    label_predictor:add(nn.Linear(100, 100))
    label_predictor:add(nn.ReLU(true))

    label_predictor:add(nn.Linear(100, opt.num_classes))
    if DA_method~='MADA' and DA_method~='MuLANN' then label_predictor:add(nn.LogSoftMax()) end

    initialize_small_net(feature_extractor, label_predictor)

    return feature_extractor, label_predictor, num_dim
end

local function gtsrb_model(DA_method)
    --Je reprends l'architecture du papier DANN - le domain discriminator est le meme que pour Office
    --DANN took inspiration from http://people.idsia.ch/~juergen/nn2012traffic.pdf
    --Does not seem like they are using padding in conv layers
    local num_dim = 2304

    local feature_extractor = nn.Sequential()

    feature_extractor:add(nn.SpatialConvolution(opt.nchannelsin, 96, 5, 5, 1, 1, 0, 0))
    feature_extractor:add(nn.ReLU(true))
    feature_extractor:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    feature_extractor:add(nn.SpatialConvolution(96, 144, 3, 3, 1, 1, 0, 0))
    feature_extractor:add(nn.ReLU(true))
    feature_extractor:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    feature_extractor:add(nn.SpatialConvolution(144, 256, 5, 5, 1, 1, 0, 0))
    feature_extractor:add(nn.ReLU(true))
    feature_extractor:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    feature_extractor:add(nn.View(-1):setNumInputDims(3))

    local label_predictor = nn.Sequential()
    label_predictor:add(nn.Linear(num_dim, 512))
    label_predictor:add(nn.ReLU(true))

    label_predictor:add(nn.Linear(512, opt.num_classes))
    if DA_method~='MADA' and DA_method~='MuLANN' then label_predictor:add(nn.LogSoftMax()) end

    for k=1,#feature_extractor.modules do
        tf_variance_initializer_fan_in(feature_extractor.modules[k])
    end
    for k=1,#label_predictor.modules do
        tf_variance_initializer_fan_in(label_predictor.modules[k])
    end

    return feature_extractor, label_predictor, num_dim
end

local function full_large_net(bottleneck_neuron_num, DA_method)
    require 'loadcaffe'
    local feature_extractor
    local ind
    if opt.model=='VGG' then
    --Caffe VGG-16 from https://gist.github.com/ksimonyan/211839e770f7b538e2d8
        local kw =  hasCudnn and 'cudnn' or 'nn'
        local model = loadcaffe.load(path.join(metadata_dir,'vgg_16/deploy.prototxt'),
            path.join(metadata_dir,'vgg_16/VGG_ILSVRC_16_layers.caffemodel'), kw)
        feature_extractor = nn.Sequential()
        for k=1,38 do feature_extractor:add(model.modules[k]) end
        ind = 39
    else
        print('What is your model') print(a+youpi)
    end

    feature_extractor:add(nn.Linear(4096, bottleneck_neuron_num))
    if hasCudnn then feature_extractor:cuda() end

    local label_predictor = nn.Sequential()
    label_predictor:add(nn.Linear(bottleneck_neuron_num, opt.num_classes))
    if DA_method~='MADA' and DA_method~='MuLANN' then label_predictor:add(nn.LogSoftMax()) end

    weight_init(feature_extractor.modules[ind], 0.005, 0.1)
    weight_init(label_predictor.modules[1], 0.01, 0)

    return feature_extractor, label_predictor
end

local function large_domain_predict_model(local_input_size, domainLambda, num_neurons, multidomain)
    local domain_predictor = nn.Sequential()
    local index = 0
    if domainLambda then
        print('Using domain Lambda ', domainLambda)
        domain_predictor:add(nn.GradientReversal(domainLambda))
    else
        index = 1
        print('######### No gradient reversal layer. ##########')
    end
    domain_predictor:add(nn.Linear(local_input_size, num_neurons))
    domain_predictor:add(nn.ReLU(true))
    domain_predictor:add(nn.Dropout(0.5))

    domain_predictor:add(nn.Linear(num_neurons, num_neurons))
    domain_predictor:add(nn.ReLU(true))
    domain_predictor:add(nn.Dropout(0.5))

    local criterion
    if multidomain then
        domain_predictor:add(nn.Linear(num_neurons, 3))
        domain_predictor:add(nn.LogSoftMax())
        criterion = nn.ClassNLLCriterion()

    else
        domain_predictor:add(nn.Linear(num_neurons, 1))
        domain_predictor:add(nn.Sigmoid())
        criterion = nn.BCECriterion()
    end
    criterion.sizeAverage = true
    dann_domain_predictor_weight_init(domain_predictor, index)

    return domain_predictor, criterion
end

local function class_large_domain_predict_model(local_input_size, domainLambda, num_neurons, num_classwise_disc,
            multidomain, domain_pred_func)

    local classwise_domain_predictor = nn.Sequential()
    local prl = nn.ParallelTable()
    prl:add(nn.SoftMax())
    prl:add(nn.Identity())
    classwise_domain_predictor:add(prl)
    classwise_domain_predictor:add(nn.OuterProduct())
    --We're expecting a matrix, and we're going to split on the first axis to get to the K classwise dom disc
    classwise_domain_predictor:add(nn.GradientReversal(domainLambda))
    classwise_domain_predictor:add(nn.SplitTable(1, 2))

    local parallel = nn.ParallelTable()
    local domain_predictor, domain_criterion
    for k=1, num_classwise_disc do
        domain_predictor, domain_criterion = domain_pred_func(local_input_size, nil, num_neurons, multidomain)
        parallel:add(domain_predictor)
    end
    classwise_domain_predictor:add(parallel)
    classwise_domain_predictor:add(nn.JoinTable(1, 1))

    return classwise_domain_predictor, domain_criterion
end

local function small_domain_predict_model(input_size, domainLambda, num_neurons)

    local domain_predictor = nn.Sequential()
    if opt.target=='svhn' or opt.source=='svhn' then
        domain_predictor:add(nn.View(-1):setNumInputDims(3))
    end

    if domainLambda then
        print('Using domain Lambda ', domainLambda)
        domain_predictor:add(nn.GradientReversal(domainLambda))
    else
        print('######### No gradient reversal layer. ##########')
    end

    domain_predictor:add(nn.Linear(input_size, num_neurons))
    domain_predictor:add(nn.ReLU(true))

    domain_predictor:add(nn.Linear(num_neurons, 1))
    domain_predictor:add(nn.Sigmoid())

    local criterion = nn.BCECriterion()
    criterion.sizeAverage = true

    for k=1,#domain_predictor.modules do
        tf_variance_initializer_fan_in(domain_predictor.modules[k])
    end
    return domain_predictor, criterion
end

local models = {
    domain_predict_model = large_domain_predict_model,
    classwise_domain_predict_model = class_large_domain_predict_model,
    full_large_net = full_large_net,

    --Getting the Traffic signs network
    gtsrb_net = gtsrb_model,

    --Getting the digits networks
    small_net = small_net,
    small_domain_predict_model = small_domain_predict_model,

    weight_init = weight_init,
    evaluate = evaluate,
    training = training
}
return models

