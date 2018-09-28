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

require 'nn'
require 'cunn'
learning_utils = require 'learning_utils'

-- GPU inputs (preallocate)
local sourceInputs = torch.CudaTensor()
local sourceLabels = torch.CudaTensor()

local targetInputs = torch.CudaTensor()
local targetLabels = torch.CudaTensor()

local unlabTargetInputs = torch.CudaTensor()
local gradient_norm
--Logging details
local logger = optim.Logger(path.join(opt.outputfolder, quanti_log_file))

-- Other upvalues for the threads
local batchNumber
local sgdconf, domainLambdaConf
local terminate
local x, dl_dx
local batch_size
if opt.target_supervision_level>0 then
   batch_size = opt.batchsize/2
else
   batch_size = opt.batchsize
end
local domDfdo = torch.CudaTensor(batch_size, opt.num_classwise_disc)
local featExtractorParams,featExtractorGradParams
local labelPredictorParams,labelPredictorGradParams
local domainClassifierParams,domainClassifierGradParams
local trainingsourcedata, trainingtargetdata, unlabelledtraintargetdata
local testsourcedata, testtargetdata
local valsourcedata, valtargetdata
local optimizer, feval, lambda_setting_function

local error_function = function(trainingsourcedata, testsourcedata, valsourcedata,
                        trainingtargetdata, testtargetdata, valtargetdata,
                       nnet, opt)
return get_all_errors(trainingsourcedata, testsourcedata, valsourcedata,
                        trainingtargetdata, testtargetdata, valtargetdata,
                       opt.target_supervision_level,
                       nnet, opt.outputfolder)
end

logger:setNames{'setting', 'iteration', 'gradient_norm',
                        'last_gdt_norm', 'DANN_lambda',
                        's_trainerr', 't_trainerr',
                        's_testerr', 't_testerr',
                        's_valerr', 't_valerr', 'source_domcost',
                    'target_domcost', 'first_lr', 'first_gdt_norm' }

local nnet = model

if hasCuda then
    nnet:cuda()
    if hasCudnn then
        nnet:cudnn()
    end
end
print('############ Starting training, DA ', opt.domain_adaptation,
    'model ', opt.model, 'transductive ', opt.fully_transductive
)

local domCost = 0
local targetDomCost = 0

--nnet.feature_extractor.modules[23] is the 1024->256 bottleneck layer
-- get all parameters
featExtractorParams,featExtractorGradParams = nnet.feature_extractor:getParameters()
labelPredictorParams,labelPredictorGradParams = nnet.label_predictor:getParameters()
local feature_extractor_grad_dim = learning_utils.get_gradient_dimensions(nnet.feature_extractor)
local label_predictor_grad_dim = learning_utils.get_gradient_dimensions(nnet.label_predictor)

local where_bottleneck_starts = learning_utils.get_bottleneck_start(feature_extractor_grad_dim)
local bottleneck_module_index = learning_utils.get_bottleneck_index()
if opt.dataset~='DIGITS' and opt.dataset~='SIGNS' then
assert(feature_extractor_grad_dim:sum()-where_bottleneck_starts==nnet.feature_extractor.modules[bottleneck_module_index].weight:nElement()
            +nnet.feature_extractor.modules[bottleneck_module_index].bias:nElement()) end

if nnet.domain_predictor then
    domainClassifierParams,domainClassifierGradParams = nnet.domain_predictor:getParameters()
    --Order of parameters: feature extractor, then label predictor, then domain predictor

    x = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1)):typeAs(featExtractorParams)
    x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
    x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
    x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
    dl_dx = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1)):typeAs(featExtractorParams)
else
    --Order of parameters: feature extractor, then label predictor
    x = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)):typeAs(featExtractorParams)
    x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
    x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
    dl_dx = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)):typeAs(featExtractorParams)

end

local total_gradient_dim = dl_dx:size()[1] 
print('Weight dimension: ', total_gradient_dim)

--Preparing values for serialization in the thread
batchNumber =1
terminate = false
local momentum=0
local rho = 0
local learningRates

if opt.optimizer_func =='sgd' then
    optimizer =  optim.sgd
    momentum = 0.9
    if opt.dataset=='OFFICE' or opt.indiv_lr then
--        To exactly reproduce DANN/MADA paper results
--          Learning rate on fine-tuned part is 1,
                   -- on from scratch, weights, is 10
                   -- on from scratch, bias, is 20
            --If Digits dataset it's not fine-tuning anyway
        print('Using indiv lr')
        learningRates = learning_utils.individual_learning_rates(nnet, featExtractorParams,
                                where_bottleneck_starts,
                                bottleneck_module_index,
                                labelPredictorParams,domainClassifierParams) end
else
    print('Not implemented')
    print(a+u)
end

local loss_weight = 1
if opt.lambda_func=='schedule' then
    print('Using scheduled lambda')
    lambda_setting_function = learning_utils.schedule_setting
    opt.domainLambda = 0
    loss_weight = 0.1
elseif opt.lambda_func =='fixed' then
    print('Using fixed lambda')
    lambda_setting_function = learning_utils.fixed_setting
else
    print('What lambda schedule are you looking for')
    print(a+u)
end

sgdconf = {learningRate = opt.eta0,
          learningRates = learningRates,
          weightDecay = opt.weightDecay,
          --SGD w/ momentum
          momentum = momentum,
          dampening = 0,
          rho = rho,

          --If using Adam, taking from dirt-t implementation
          beta1 = 0.5,
          beta2 = 0.999,

          --If using salera
          SAGMA_lambda = opt.salera_lambda,
          alpha = opt.adapt_alpha,
          adapt_constant = opt.adapt_factor,
          mbratio = 0.001,

         gpu=gpu }
domainLambdaConf = {lambda = opt.domainLambda,
                    gamma = opt.lambda_schedule_gamma,
                    alpha = opt.lambda_alera_alpha,
                    factor = opt.lambda_alera_factor,
                    maxiter = opt.lambda_maxiter
}
show_progress=true
if string.find(paths.thisfile(), 'titanic') or string.find(paths.thisfile(), 'nfs') then
    show_progress=false end
----------------------------------------------------------------------
-- train model
--
local maxiter = maxiter
trainingsourcedata = source_data_getter:trainDataset()
trainingtargetdata = target_data_getter:trainDataset()
testsourcedata = source_data_getter:testDataset()
testtargetdata = target_data_getter:testDataset()

if opt.dataset=='BIO' then
    valsourcedata = source_data_getter:valDataset()
    valtargetdata = target_data_getter:valDataset()
end

if nnet.domain_predictor and opt.target_unlabelled_presence then
    print('Using unlabelled target, fully trans?', opt.fully_transductive)
    unlabelledtraintargetdata = (opt.fully_transductive and testtargetdata) or target_data_getter:unlabelledTrainDataset()
end

--------------------------------------------------------------------
-- Looking at baseline values
--
local res = error_function(trainingsourcedata, testsourcedata, valsourcedata,
                            trainingtargetdata, testtargetdata, valtargetdata,
                           nnet, opt)
local log = {opt.train_setting, 0, 0, 0, domainLambdaConf.lambda}
for _,el in ipairs(res) do table.insert(log, el) end
table.insert(log, 0)
table.insert(log, 0)
table.insert(log, sgdconf.learningRate)
table.insert(log, 0)
logger:add(log)

if not opt.incl_source then print ('ATTENTION ATTENTION PAS DE SOURCE DANS LA BASELINE OUI ?') end

for t = 1,maxiter do
   --------------------------------------------------------------------
   -- progress
   --
   if show_progress then xlua.progress(t, maxiter) end

   if terminate then break end

   donkeys:addjob(
   function()
       local p = torch.rand(1)[1]
       local sourceInputs=-1
       local sourceLabels=-1
       local unlabelledTargetInputs=-1
       local targetInputs=-1
       local targetLabels=-1

       if not nnet.domain_predictor then
           if opt.incl_source then
               sourceInputs, sourceLabels = trainingsourcedata:getBatch(batch_size)
           end
           if opt.target_supervision_level>0 then
               targetInputs, targetLabels = trainingtargetdata:getBatch(batch_size)
           end
       else
           if opt.target_supervision_level>0 then
                sourceInputs, sourceLabels = trainingsourcedata:getBatch(batch_size)
                targetInputs, targetLabels = trainingtargetdata:getBatch(batch_size)
                if opt.target_unlabelled_presence and 1-p<=opt.proba then
                    unlabelledTargetInputs,labs = unlabelledtraintargetdata:getBatch(batch_size)
                end
           else
               sourceInputs, sourceLabels = trainingsourcedata:getBatch(batch_size)
               if not opt.target_unlabelled_presence then print('Hmm. Pb??') print(a+yupi) end
               unlabelledTargetInputs = trainingtargetdata:getBatch(batch_size)
           end
       end
       return __threadid, sourceInputs, sourceLabels,
                            targetInputs, targetLabels,
                            unlabelledTargetInputs
   end,

   function (id, sourceInCPU, sourceLabelsCPU,
                 targetInCPU, targetLabelsCPU,
                 unlabTargetInCPU)
        -- transfer over to GPU
        if sourceInCPU~=-1 then
            sourceInputs:resize(sourceInCPU:size()):copy(sourceInCPU)
            sourceLabels:resize(sourceLabelsCPU:size()):copy(sourceLabelsCPU)
        end

        if targetInCPU~=-1 then
            targetInputs:resize(targetInCPU:size()):copy(targetInCPU)
            targetLabels:resize(targetLabelsCPU:size()):copy(targetLabelsCPU)
        end

        if unlabTargetInCPU~=-1 then
            unlabTargetInputs:resize(unlabTargetInCPU:size()):copy(unlabTargetInCPU)
        end
        --------------------------------------------------------------------
        -- define eval closure
        --
        if opt.dataset=='OFFICE' or opt.lr_decay then
            sgdconf.learningRate = opt.eta0/(1+opt.gamma*(batchNumber-1))^opt.beta
        end

        local function feval_noDA(x)
            featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
            labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
            -- reset gradient/f
            featExtractorGradParams:zero()
            labelPredictorGradParams:zero()
            local labelCost = 0

            --Source propagation
            --#i. Getting to source label cost
            if sourceInCPU~=-1 then
                local feats = nnet.feature_extractor:forward(sourceInputs)
                local preds = nnet.label_predictor:forward(feats)
                labelCost = labelCost + nnet.label_criterion:forward(preds,sourceLabels)

                local sourceLabelDfdo = nnet.label_criterion:backward(preds, sourceLabels)
                local sourceGradLabelPredictor = nnet.label_predictor:backward(feats, sourceLabelDfdo)
                nnet.feature_extractor:backward(sourceInputs, sourceGradLabelPredictor)
            end
            --#ii. Getting to target label cost
            if targetInCPU~=-1 then
                local feats = nnet.feature_extractor:forward(targetInputs)
                local preds = nnet.label_predictor:forward(feats)
                labelCost = labelCost+ nnet.label_criterion:forward(preds,targetLabels)

                local targetLabelDfdo = nnet.label_criterion:backward(preds, targetLabels)
                local targetGradLabelPredictor = nnet.label_predictor:backward(feats, targetLabelDfdo)
                nnet.feature_extractor:backward(targetInputs, targetGradLabelPredictor)
            end

            x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
            x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
            dl_dx:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)

            return labelCost, dl_dx
        end

        local function feval_DANN(x)
            featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
            labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
            domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))
        -- reset gradient/f
            featExtractorGradParams:zero()
            labelPredictorGradParams:zero()
            domainClassifierGradParams:zero()

        --Source propagation
            --#i. Getting to source label cost
            local feats = nnet.feature_extractor:forward(sourceInputs)
            local preds = nnet.label_predictor:forward(feats)
            local sourceLabelCost = nnet.label_criterion:forward(preds,sourceLabels)

            local sourceLabelDfdo = nnet.label_criterion:backward(preds, sourceLabels)
            local sourceGradLabelPredictor = nnet.label_predictor:backward(feats, sourceLabelDfdo)
            nnet.feature_extractor:backward(sourceInputs, sourceGradLabelPredictor)

            --#ii. Getting to source domain cost
            local domPreds = nnet.domain_predictor:forward(feats)
            local domainSourceLabels = torch.Tensor(domPreds:size(1)):fill(1):typeAs(domPreds)
            domCost = nnet.domain_criterion:forward(domPreds,domainSourceLabels)
            local domDfdo = nnet.domain_criterion:backward(domPreds,domainSourceLabels)
--Loss_weight Caffe parameter: value 0.1 in original DANN paper
            domCost = domCost*loss_weight
            domDfdo:mul(loss_weight)

            local gradDomainClassifier = nnet.domain_predictor:backward(feats,domDfdo)
            nnet.feature_extractor:backward(sourceInputs, gradDomainClassifier*domainLambdaConf.lambda)

        --- Target propagation
            local domainTargetLabels = domainSourceLabels:clone():fill(0)

            local currTargetIn
            local targetFeats
            local targetLabelCost = 0
            if targetInCPU~=-1 then
                --#i. Getting to target label cost
                targetFeats = nnet.feature_extractor:forward(targetInputs)
                local targetPreds = nnet.label_predictor:forward(targetFeats)
                targetLabelCost = nnet.label_criterion:forward(targetPreds,targetLabels)

                local targetLabelDfdo = nnet.label_criterion:backward(targetPreds, targetLabels)
                local targetGradLabelPredictor = nnet.label_predictor:backward(targetFeats, targetLabelDfdo)
                nnet.feature_extractor:backward(targetInputs, targetGradLabelPredictor)

                --#ii. Getting to target domain cost
                if unlabTargetInCPU==-1 then
                    currTargetIn = targetInputs
                end
            end
            if unlabTargetInCPU~=-1 then
                --#i. Getting to target features
                targetFeats = nnet.feature_extractor:forward(unlabTargetInputs)
                currTargetIn = unlabTargetInputs
            end

            --#ii. Getting to target domain cost
	    local targetDomPreds = nnet.domain_predictor:forward(targetFeats)
            targetDomCost = nnet.domain_criterion:forward(targetDomPreds,domainTargetLabels)
            local targetDomDfdo = nnet.domain_criterion:backward(targetDomPreds,domainTargetLabels)
--Loss_weight Caffe parameter: value 0.1 in original DANN paper
            targetDomCost = targetDomCost*loss_weight
            targetDomDfdo:mul(loss_weight)

            local targetGradDomainClassifier = nnet.domain_predictor:backward(targetFeats,targetDomDfdo)
            nnet.feature_extractor:backward(currTargetIn, targetGradDomainClassifier*domainLambdaConf.lambda)

            x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
            x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
            x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
            dl_dx:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)

            --Updating parameter lambda according to the paper/ALeRA/no update
            domainLambdaConf.lambda=lambda_setting_function(domainLambdaConf, batchNumber)
            --Saving domain error to note it
            return (sourceLabelCost+targetLabelCost) -domainLambdaConf.lambda*(domCost+targetDomCost), dl_dx
        end

        local function feval_MADA(x)
            featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
            labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
            domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))
        -- reset gradient/f
            featExtractorGradParams:zero()
            labelPredictorGradParams:zero()
            domainClassifierGradParams:zero()
            domCost = 0
            targetDomCost = 0

        --Source propagation
            --#i. Getting source softmax and feature
            local feats = nnet.feature_extractor:forward(sourceInputs)
            local preds = nnet.label_predictor:forward(feats)
            local sourceLabelCost = nnet.label_criterion:forward(preds, sourceLabels)
            local sourceLabelDfdo = nnet.label_criterion:backward(preds, sourceLabels)

            local domPreds = nnet.domain_predictor:forward({preds, feats})
            local domainSourceLabels = torch.Tensor(domPreds:size(1)):fill(1):typeAs(domPreds)
            for k=1, opt.num_classwise_disc do
                domCost = domCost + nnet.domain_criterion:forward(domPreds[{{}, {k}}], domainSourceLabels)
                domDfdo[{{}, {k}}] = nnet.domain_criterion:backward(domPreds[{{}, {k}}], domainSourceLabels)
            end
            --This is what is done when using loss_weight = 0.3 in the prototxt files of MADA paper
            domCost = domCost*2*loss_weight
            domDfdo:mul(2*loss_weight)
            local gradDomainClassifier = nnet.domain_predictor:backward({preds, feats}, domDfdo)
            --local sourceGradLabelPredictor = nnet.label_predictor:backward(feats, sourceLabelDfdo:add(domainLambdaConf.lambda, gradDomainClassifier[1]))
            local sourceGradLabelPredictor = nnet.label_predictor:backward(feats, sourceLabelDfdo)
            nnet.feature_extractor:backward(sourceInputs, sourceGradLabelPredictor:add(domainLambdaConf.lambda, gradDomainClassifier[2]))

        --- Target propagation
            local domainTargetLabels = domainSourceLabels:clone():fill(0)

            local currTargetIn, targetFeats, targetPreds
            local targetLabelCost = 0
            if targetInCPU~=-1 then
                --#i. Getting to target label cost
                targetFeats = nnet.feature_extractor:forward(targetInputs)
                targetPreds = nnet.label_predictor:forward(targetFeats)
                targetLabelCost = nnet.label_criterion:forward(targetPreds, targetLabels)
                local targetLabelDfdo = nnet.label_criterion:backward(targetPreds, targetLabels)

                local targetGradLabelPredictor = nnet.label_predictor:backward(targetFeats, targetLabelDfdo)
                nnet.feature_extractor:backward(targetInputs, targetGradLabelPredictor)

                --#ii. Getting to target domain cost
                if unlabTargetInCPU==-1 then
                    currTargetIn = targetInputs
                end
            end
            if unlabTargetInCPU~=-1 then
                --#i. Getting to target features
                targetFeats = nnet.feature_extractor:forward(unlabTargetInputs)
                targetPreds = nnet.label_predictor:forward(targetFeats)
                currTargetIn = unlabTargetInputs
            end

            --#ii. Getting to target domain cost
            local targetDomPreds = nnet.domain_predictor:forward({targetPreds, targetFeats})
            for k=1, opt.num_classwise_disc do
                targetDomCost = targetDomCost + nnet.domain_criterion:forward(targetDomPreds[{{}, {k}}], domainTargetLabels)
                domDfdo[{{}, {k}}] = nnet.domain_criterion:backward(targetDomPreds[{{}, {k}}], domainTargetLabels)
            end
            --This is what is done when using loss_weight = 0.3 in the prototxt files of MADA paper
            targetDomCost = targetDomCost*2*loss_weight
            domDfdo:mul(2*loss_weight)

            local targetGradDomainClassifier = nnet.domain_predictor:backward({targetPreds, targetFeats}, domDfdo)
            --local targetGradLabelPredictor = nnet.label_predictor:backward(targetFeats, targetGradDomainClassifier[1]:mul(domainLambdaConf.lambda))
            --nnet.feature_extractor:backward(currTargetIn, targetGradLabelPredictor:add(domainLambdaConf.lambda, targetGradDomainClassifier[2]))
            nnet.feature_extractor:backward(currTargetIn, domainLambdaConf.lambda*targetGradDomainClassifier[2])

            x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
            x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
            x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
            dl_dx:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)

            --Updating parameter lambda according to the paper/ALeRA/no update
            domainLambdaConf.lambda = lambda_setting_function(domainLambdaConf, batchNumber)
            --Saving domain error to note it
            return (sourceLabelCost+targetLabelCost) -domainLambdaConf.lambda*(domCost+targetDomCost), dl_dx
        end

        if not nnet.domain_predictor then
            feval = feval_noDA
        elseif opt.domain_method=='DANN' then
            feval = feval_DANN
        elseif opt.domain_method =='MADA' then
            feval = feval_MADA
        else
            print('I dont know what youre trying to do')
            print(a+youpi)
        end

       --------------------------------------------------------------------
       -- one SGD step
        _,fs = optimizer(feval, x, sgdconf)

       --fs is mean squared error
       gradient_norm = dl_dx:norm()
       if fs and math.fmod(batchNumber-1, 10000)==0 then print(batchNumber, fs[1], gradient_norm) end
       nnet.feature_extractor:clearState()
       if nnet.domain_predictor then nnet.domain_predictor:clearState() end

       if fs then
           --Checking if the gradient norm is NaN or learning rate very small
           local monitor_values = torch.Tensor({gradient_norm, fs[1]})
           --If gradient norm has become NaN, saving an error of 100% before stopping
           if monitor_values:ne(monitor_values):sum()>0 then
               print('Gradient or error becomes NaN, stopping run here')
               utils.write_stop_reason(opt.outputfolder, 'Gradient or error becomes NaN, stopping run here')
               terminate = true
           end
       --If learning rate has become too small as reported by the optim algorithm, see the test error and save it before stopping
       else
           print('Stopping because lr too small.')
           utils.write_stop_reason(opt.outputfolder, 'Stopping because lr too small.')
           terminate = true
       end
       batchNumber = batchNumber +1
   end
   )

   if math.fmod(t-1, 100)==0 then donkeys:synchronize() end

   if math.fmod(t-1, plotinterval)==0 or math.fmod(t-1, statinterval)==0 then
        --LOOKING AT TRAIN, TEST AND VALIDATION ERRORS
        local res = error_function(trainingsourcedata, testsourcedata, valsourcedata,
                                    trainingtargetdata, testtargetdata, valtargetdata,
                                   nnet, opt)

        local first_layer_dim = feature_extractor_grad_dim[1]
        local first_gradient_norm = torch.norm(dl_dx[{{1, first_layer_dim}}])

        local last_gradient_norm = 0

        local log = {opt.train_setting, t, gradient_norm, last_gradient_norm, domainLambdaConf.lambda}
        for _,el in ipairs(res) do table.insert(log, el) end

        table.insert(log, domCost)
        table.insert(log, targetDomCost)

        table.insert(log, sgdconf.learningRate)
        table.insert(log, first_gradient_norm)

        logger:add(log)
   end

   if t>1 and (math.fmod(t-1 , statinterval) == 0 or (opt.dataset=='DIGITS' and (t-1==10000 or t-1==20000))) then
       utils.save_net(t, nnet)
   end
end
donkeys:synchronize()
donkeys:terminate()
--Final export
--utils.save_net('end', nnet)
