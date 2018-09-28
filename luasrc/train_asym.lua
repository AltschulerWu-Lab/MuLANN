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
local rest_TargetInputs = torch.CudaTensor()
local gradient_norm
--Logging details
local logger = optim.Logger(path.join(opt.outputfolder, quanti_log_file))
local batch_size = opt.batchsize/2
local knowledge_batch = math.floor(opt.unknown_perc*batch_size)
local domDfdo = torch.CudaTensor(batch_size, opt.num_classwise_disc)

local infoLabel_func
if opt.iterative_labeling=='entropy' then
    infoLabel_func = learning_utils.sort_labels_entropy
else
    infoLabel_func = learning_utils.sort_labels_diff
end

-- Other upvalues for the threads
local batchNumber
local sgdconf, domainLambdaConf
local terminate
local x, dl_dx
local featExtractorParams,featExtractorGradParams
local labelPredictorParams,labelPredictorGradParams
local domainClassifierParams,domainClassifierGradParams
local infoClassifierParams, infoClassifierGradParams
local trainingsourcedata, trainingtargetdata, unlabelledtraintargetdata, trainingextradata
local testsourcedata, testtargetdata, testextradata, rest_testtargetdata
local valsourcedata, valtargetdata
local optimizer, lambda_setting_function

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
    'model ', opt.model, 'transductive ', opt.fully_transductive, 'using asymmetry file')

local domCost = 0
local targetDomCost = 0

--nnet.feature_extractor.modules[23] is the 1024->256 bottleneck layer
-- get all parameters
featExtractorParams,featExtractorGradParams = nnet.feature_extractor:getParameters()
labelPredictorParams,labelPredictorGradParams = nnet.label_predictor:getParameters()
domainClassifierParams,domainClassifierGradParams = nnet.domain_predictor:getParameters()
infoClassifierParams, infoClassifierGradParams = nnet.info_predictor:getParameters()

local feature_extractor_grad_dim = learning_utils.get_gradient_dimensions(nnet.feature_extractor)
local label_predictor_grad_dim = learning_utils.get_gradient_dimensions(nnet.label_predictor)
--Order of parameters: feature extractor, then label predictor, then domain predictor
local gradient_dimensions = torch.cat({feature_extractor_grad_dim, label_predictor_grad_dim})

x = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)
                +domainClassifierParams:size(1)+infoClassifierParams:size(1)):typeAs(featExtractorParams)
x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1),
                                    infoClassifierParams:size(1)):copy(infoClassifierParams)
dl_dx = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+
                                    domainClassifierParams:size(1)+infoClassifierParams:size(1)):typeAs(featExtractorParams)

local total_gradient_dim = dl_dx:size(1)
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
    if opt.dataset=='OFFICE' or opt.dataset=='BIO' or opt.indiv_lr then
        print('Using indiv lr')
        local where_bottleneck_starts = learning_utils.get_bottleneck_start(feature_extractor_grad_dim)
        local bottleneck_module_index = learning_utils.get_bottleneck_index()
        learningRates = learning_utils.individual_learning_rates(nnet, featExtractorParams,
                                where_bottleneck_starts,
                                bottleneck_module_index,
                                labelPredictorParams, domainClassifierParams, infoClassifierParams) 
    end
else
    print('Not implemented')
    print(a+u)
end

local lambda_loss_weight = 1
local zeta_loss_weight = 1
if opt.lambda_func=='schedule' then
    print('Using scheduled lambda')
    lambda_setting_function = learning_utils.schedule_setting
    opt.domainLambda = 0
    lambda_loss_weight = 0.1
    if opt.zeta_schedule then
        zeta_loss_weight = 0.1
        opt.zeta=0
    end

elseif opt.lambda_func =='fixed' then
    print('Using fixed lambda')
    lambda_setting_function = learning_utils.fixed_setting
else
    print('What lambda schedule are you looking for')
    print(a+u)
end
print('Using loss weights for lambda, zeta', lambda_loss_weight, zeta_loss_weight)

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
                    zeta = opt.zeta,
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

--This will return for which classes I have labeled images in train
local target_classes = trainingtargetdata:classes():typeAs(sourceInputs)
--Target classes are different than source classes when we have asymmetry, so 
  --need to change the vector size for computing the entropy/diff later
local num_source_classes = trainingsourcedata:classes():sum()
target_classes = target_classes[{{}, {1, num_source_classes}}]:contiguous()

if opt.dataset=='BIO' then
    valsourcedata = source_data_getter:valDataset()
    valtargetdata = target_data_getter:valDataset()
end

print('Using unlabelled target, fully trans?', opt.fully_transductive)
unlabelledtraintargetdata = (opt.fully_transductive and testtargetdata) or target_data_getter:unlabelledTrainDataset()

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
       local sourceInputs, sourceLabels = trainingsourcedata:getBatch(batch_size)
       local targetInputs, targetLabels = trainingtargetdata:getBatch(batch_size)
       local unlabelledTargetInputs = unlabelledtraintargetdata:getBatch(batch_size)

       return __threadid, sourceInputs, sourceLabels,
                            targetInputs, targetLabels,
                            unlabelledTargetInputs
   end,

   function (id, sourceInCPU, sourceLabelsCPU,
                 targetInCPU, targetLabelsCPU,
                 unlabTargetInCPU)
        -- transfer over to GPU
        sourceInputs:resize(sourceInCPU:size()):copy(sourceInCPU)
        sourceLabels:resize(sourceLabelsCPU:size()):copy(sourceLabelsCPU)

        targetInputs:resize(targetInCPU:size()):copy(targetInCPU)
        targetLabels:resize(targetLabelsCPU:size()):copy(targetLabelsCPU)

        unlabTargetInputs:resize(unlabTargetInCPU:size()):copy(unlabTargetInCPU)
        --------------------------------------------------------------------
        -- define eval closure
        --
        if opt.lr_decay then
            sgdconf.learningRate = opt.eta0/(1+opt.gamma*(batchNumber-1))^opt.beta
        end

        local function feval_MuLANN(x)
            featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
            labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
            domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))
            infoClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1),
                                                        infoClassifierParams:size(1)))
        -- reset gradient/f
            featExtractorGradParams:zero()
            labelPredictorGradParams:zero()
            domainClassifierGradParams:zero()
            infoClassifierGradParams:zero()

        --Source propagation
            --#i. Getting to source label cost
            local feats = nnet.feature_extractor:forward(sourceInputs)
            local preds = nnet.label_predictor:forward(feats)
            local labelCost = nnet.label_criterion:forward(preds,sourceLabels)

            local sourceLabelDfdo = nnet.label_criterion:backward(preds, sourceLabels)
            local sourceGradLabelPredictor = nnet.label_predictor:backward(feats, sourceLabelDfdo)
            nnet.feature_extractor:backward(sourceInputs, sourceGradLabelPredictor)

            --#ii. Getting to source domain cost
            local domPreds = nnet.domain_predictor:forward(feats)
            local domainSourceLabels = torch.Tensor(domPreds:size(1)):fill(1):typeAs(domPreds)
            domCost = nnet.domain_criterion:forward(domPreds,domainSourceLabels)
            local domDfdo = nnet.domain_criterion:backward(domPreds,domainSourceLabels)
            domCost = domCost*lambda_loss_weight
            domDfdo:mul(lambda_loss_weight)

            local gradDomainClassifier = nnet.domain_predictor:backward(feats,domDfdo)
            nnet.feature_extractor:backward(sourceInputs, gradDomainClassifier*domainLambdaConf.lambda)

        --- Target propagation
            --#i. Getting to target label cost
            local targetFeats = nnet.feature_extractor:forward(targetInputs)
            local targetLabPreds = nnet.label_predictor:forward(targetFeats)
            labelCost = labelCost + nnet.label_criterion:forward(targetLabPreds,targetLabels)

            local targetLabelDfdo = nnet.label_criterion:backward(targetLabPreds, targetLabels)
            local targetGradLabelPredictor = nnet.label_predictor:backward(targetFeats, targetLabelDfdo)

            --ii. Forward of labelled target into info disc
            local targetPreds = nnet.info_predictor:forward(targetFeats[{{1, knowledge_batch}}])
            local infoTargetLabels = domainSourceLabels[{{1, knowledge_batch}}]
            local infoCost = nnet.domain_criterion:forward(targetPreds, infoTargetLabels)
            local targetInfoDfdo = nnet.domain_criterion:backward(targetPreds, infoTargetLabels)
            targetInfoDfdo:mul(zeta_loss_weight)

            local targetGradInfoPredictor = nnet.info_predictor:backward(targetFeats[{{1, knowledge_batch}}], targetInfoDfdo)
            targetGradInfoPredictor:mul(domainLambdaConf.zeta)
            targetGradLabelPredictor[{{1, knowledge_batch}}]:add(targetGradInfoPredictor)
            nnet.feature_extractor:backward(targetInputs, targetGradLabelPredictor)

            --#iii. Getting to target domain cost
            local domainTargetLabels = domainSourceLabels:fill(0)
            local currTargetIn, unlab_targetLabPreds
            local p = torch.rand(1)[1]
            if p<=opt.proba then
                currTargetIn = targetInputs
            else
                currTargetIn = unlabTargetInputs
                targetFeats = nnet.feature_extractor:forward(unlabTargetInputs)
                unlab_targetLabPreds = nnet.label_predictor:forward(targetFeats)
            end
            local targetDomPreds = nnet.domain_predictor:forward(targetFeats)
            targetDomCost = nnet.domain_criterion:forward(targetDomPreds,domainTargetLabels)
            local targetDomDfdo = nnet.domain_criterion:backward(targetDomPreds,domainTargetLabels)
            targetDomCost = targetDomCost*lambda_loss_weight
            targetDomDfdo:mul(lambda_loss_weight)

            local targetGradDomainClassifier = nnet.domain_predictor:backward(targetFeats,targetDomDfdo)
            nnet.feature_extractor:backward(currTargetIn, targetGradDomainClassifier*domainLambdaConf.lambda)

            --iv. Forward of unlabelled target into info disc
            if p<=opt.proba then
                targetFeats = nnet.feature_extractor:forward(unlabTargetInputs)
                unlab_targetLabPreds = nnet.label_predictor:forward(targetFeats)
            end
            local infoTargetLabels_mask = infoLabel_func(unlab_targetLabPreds, target_classes, knowledge_batch, batch_size)

            targetPreds = nnet.info_predictor:forward(targetFeats)
            infoCost = infoCost + nnet.domain_criterion:forward(targetPreds, domainTargetLabels)
            targetInfoDfdo = nnet.domain_criterion:backward(targetPreds, domainTargetLabels)
            infoCost = infoCost*zeta_loss_weight
            targetInfoDfdo:mul(zeta_loss_weight)

            targetGradInfoPredictor = nnet.info_predictor:backward(targetFeats, targetInfoDfdo:cmul(infoTargetLabels_mask))
            nnet.feature_extractor:backward(unlabTargetInputs, targetGradInfoPredictor*domainLambdaConf.zeta)

            x:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
            x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
            x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
            x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1),
                                    infoClassifierParams:size(1)):copy(infoClassifierParams)

            dl_dx:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)
            dl_dx:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1)+domainClassifierGradParams:size(1),
                                    infoClassifierGradParams:size(1)):copy(infoClassifierGradParams)

            local err =labelCost - domainLambdaConf.lambda*(domCost+targetDomCost) - domainLambdaConf.zeta*infoCost
            --Updating parameter lambda according to DANN paper/ALeRA/no update
            domainLambdaConf.lambda = lambda_setting_function(domainLambdaConf, batchNumber)
            domainLambdaConf.zeta = lambda_setting_function(domainLambdaConf, batchNumber)
            --Saving domain error to note it
            return err, dl_dx
        end

       --------------------------------------------------------------------
       -- one SGD step
       -- fs is mean squared error
        _,fs = optimizer(feval_MuLANN, x, sgdconf)

       gradient_norm = dl_dx:norm()
       if fs and math.fmod(batchNumber-1, 10000)==0 then print(batchNumber, fs[1], gradient_norm) end
       nnet:clearState()

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

        local first_layer_dim = gradient_dimensions[1]
        --dl_dx is already squared
        local first_gradient_norm = torch.norm(dl_dx[{{1, first_layer_dim}}])
        local last_gradient_norm = torch.norm(dl_dx[{{ -infoClassifierParams:size(1), -1}}])

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
