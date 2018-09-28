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

local function get_gradient_dimensions(module)
    local gradient_dimensions = {}
    for i=1,#module.modules do
        if module.modules[i].weight then
            local dim = module.modules[i].weight:nElement()
            if module.modules[i].bias then dim = dim + module.modules[i].bias:nElement() end
            table.insert(gradient_dimensions, dim)
        end
    end
    return torch.Tensor(gradient_dimensions)
end

local function schedule_setting(lambdaConf, batchNumber)
    --This is made to go from an initial lambdaconf.lambda, small, to 1 at the 10k iter, and then
--1 until the end
    local p = math.min(1, batchNumber/lambdaConf.maxiter)
    return 2/(1+math.exp(-1*lambdaConf.gamma*p)) -1
end

local function fixed_setting(lambdaConf)
    return lambdaConf.lambda
end

local function compute_select_diff(label_predictions, class_presence)
    label_predictions:exp()
    label_predictions:cdiv(label_predictions:clone():cmul(class_presence:expandAs(label_predictions)):sum(2):expandAs(label_predictions))
    local dom_sorted_softmaxs = label_predictions:cmul(class_presence:expandAs(label_predictions)):sort(2)
    local res = dom_sorted_softmaxs[{{}, {-1}}] - dom_sorted_softmaxs[{{}, {-2}}]
    return res
end

local function compute_select_entropy(label_predictions, class_presence)
    label_predictions:exp()
    label_predictions:cdiv(label_predictions:clone():cmul(class_presence:expandAs(label_predictions)):sum(2):expandAs(label_predictions))
    label_predictions:add(1e-10)
    local infoLabels = - label_predictions:cmul(label_predictions:clone():log():cmul(class_presence:expandAs(label_predictions))):sum(2)
    infoLabels:div(torch.log(torch.sum(class_presence)))
    return infoLabels
end

local function sort_labels_diff(label_predictions, class_presence, knowledge_batch, batch_size)
    local res
    if knowledge_batch==batch_size then
        res = torch.ones(label_predictions:size(1))
    else
        local diff_predictions = compute_select_diff(label_predictions, class_presence)
    --This will return from small to big differences
        local _,sorting = diff_predictions:sort(1)
        local cutoff = diff_predictions[sorting[knowledge_batch+1][1]][1]
        res = diff_predictions:lt(cutoff)
    end
    return res:typeAs(label_predictions):view(-1,1)
end

local function sort_labels_entropy(label_predictions, class_presence, knowledge_batch, batch_size)
    local res
    if knowledge_batch==batch_size then
        res = torch.ones(label_predictions:size(1))
    else
        local entropy_predictions = compute_select_entropy(label_predictions, class_presence)
    --This will return from big to small entropies
        local _,sorting = entropy_predictions:mul(-1):sort(1)
        local cutoff = entropy_predictions[sorting[knowledge_batch+1][1]][1]
        res = entropy_predictions:lt(cutoff)
    end
    return res:typeAs(label_predictions):view(-1,1)
end


local function individual_learning_rates(nnet, featExtractorParams, where_bottleneck_starts, bottleneck_module_index,
                                        labelPredictorParams,domainClassifierParams, infoClassifierParams)
--        To exactly reproduce DANN/MADA paper results
--          Learning rate on fine-tuned part is 1,
                       -- on from scratch, weights, is 10
                       -- on from scratch, bias, is 20

    local learningRates = featExtractorParams.new(featExtractorParams:size()):fill(1)
    learningRates[{{where_bottleneck_starts, -1}}]:mul(10)
    learningRates[{{-nnet.feature_extractor.modules[bottleneck_module_index].bias:nElement(), -1}}]:mul(2)

    local l2 = labelPredictorParams.new(labelPredictorParams:size()):fill(10)
    l2[{{-nnet.label_predictor.modules[1].bias:nElement(), -1}}]:mul(2)
    learningRates = torch.cat(learningRates, l2)

    if nnet.domain_predictor then
        local l3 = domainClassifierParams.new(domainClassifierParams:size()):fill(10)
        local module_count = opt.domain_method=='MADA' and #nnet.domain_predictor.modules[5].modules or 1
        local start = 1
        for k=1, module_count do
            local currMod = opt.domain_method=='MADA' and nnet.domain_predictor.modules[5].modules[k].modules or nnet.domain_predictor.modules
            for j=1, #currMod do
                if currMod[j].weight then
                    start = start + currMod[j].weight:nElement()
                    local nbias = currMod[j].bias:nElement()
                    l3[{{start, start+nbias-1}}]:mul(2)
                    start = start +nbias
                end
            end
        end
        assert(start-1 == domainClassifierParams:nElement())
        learningRates = torch.cat(learningRates, l3)

        if nnet.info_predictor then
            l3 = infoClassifierParams.new(infoClassifierParams:size()):fill(10)
            start = 1
            local currMod = nnet.info_predictor.modules
            for j=1, #currMod do
                if currMod[j].weight then
                    start = start + currMod[j].weight:nElement()
                    local nbias = currMod[j].bias:nElement()
                    l3[{{start, start+nbias-1}}]:mul(2)
                    start = start +nbias
                end
            end
            assert(start-1 == infoClassifierParams:nElement())
            learningRates = torch.cat(learningRates, l3)
        end
    end
    return learningRates
end

local function get_bottleneck_start(feature_extractor_grad_dim)
    return feature_extractor_grad_dim[{{1,-2}}]:sum()
end

local function get_bottleneck_index()
    return (opt.model=='AlexNet' and 23) or 39
end

 --------------------------------------------------------------------
 -- enforces Max-norm constraint
 --
local function maxNormConstraint(module, constraint_value)
    for i=1,#module.modules do
        if module.modules[i].weight then
            --This should not take the biases into account (and it doesnt)
            --Documentation of fct https://github.com/torch/torch7/blob/master/doc/maths.md#torchrenormres-x-p-dim-maxnorm
            module.modules[i].weight:renorm(2, 1, constraint_value)
        end
    end
end

local learning_utils = {
    schedule_setting = schedule_setting,
    fixed_setting = fixed_setting,
    individual_learning_rates = individual_learning_rates,
    get_bottleneck_start = get_bottleneck_start,
    get_bottleneck_index = get_bottleneck_index,
    maxNormConstraint = maxNormConstraint,
    get_gradient_dimensions = get_gradient_dimensions,
    sort_labels_diff = sort_labels_diff,
    sort_labels_entropy = sort_labels_entropy,
}
return learning_utils
