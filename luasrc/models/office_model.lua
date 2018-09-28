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

general_model = require 'general_model'

function getmodel(DA_method, train_parameters)
    local module = {}
    module.name=opt.model
    module.inputsize = opt.inputsize
    print('Using model ', opt.model)

    local num_disc_neurons = train_parameters['num_disc_neurons']
    local bottleneck_size = train_parameters['bottleneck_size']
    local domain_pred_func= general_model.domain_predict_model

    if opt.dataset=='BIO' or opt.dataset=='OFFICE' then
        module.feature_extractor,module.label_predictor = general_model.full_large_net(bottleneck_size, DA_method)

    elseif opt.dataset=='DIGITS' then
        num_disc_neurons = 100
        bottleneck_size = 1200
        domain_pred_func= general_model.small_domain_predict_model
        module.feature_extractor,module.label_predictor = general_model.small_net(DA_method)

    elseif opt.dataset=='SIGNS' then
        module.feature_extractor,module.label_predictor, bottleneck_size = general_model.gtsrb_net(DA_method)

    else
        print('What is your dataset')
        print(a+youpi)
    end

    if DA_method then
        --Getting the base domain discriminator for DANNs
        if DA_method =='DANN' or DA_method =='MuLANN' then
            if DA_method =='MuLANN' then
                module.info_predictor = domain_pred_func(bottleneck_size, nil, num_disc_neurons)

                if opt.extra_domain then
                    --If we just have 2 domains then we can just use the domain_criterion as info_criterion
                    module.info_criterion = nn.BCECriterion()
                    module.info_criterion.sizeAverage = true
                end
            end
            --We are just using a general domain discriminator
            module.domain_predictor, module.domain_criterion = domain_pred_func(bottleneck_size,
                                                                    1, num_disc_neurons, opt.extra_domain)

        elseif DA_method =='MADA' then
            module.domain_predictor, module.domain_criterion = general_model.classwise_domain_predict_model(bottleneck_size,
                                    1, num_disc_neurons, opt.num_classwise_disc, opt.extra_domain, domain_pred_func)
        end
    end

    local label_criterion = nn.ClassNLLCriterion()
    if DA_method=='MADA' or DA_method=='MuLANN' then label_criterion = nn.CrossEntropyCriterion() end
    label_criterion.sizeAverage = true
    module.label_criterion = label_criterion
    --Initializing weights: DONE IN THE ORIGINAL FUNCTIONS

    --Defining a few methods on module. These will be copied when using module:clone()
    function module:name() return self.name end
    function module:inputsize() return self.inputsize end
    function module:evaluate()
        --Need to define this as it is not defined in nn
        general_model.evaluate(self.feature_extractor)
        general_model.evaluate(self.domain_predictor)
        general_model.evaluate(self.info_predictor)
        general_model.evaluate(self.label_predictor)
    end
    function module:training()
        --Need to define this as it is not defined in nn
        general_model.training(self.feature_extractor)
        general_model.training(self.domain_predictor)
        general_model.training(self.info_predictor)
        general_model.training(self.label_predictor)
    end
    function module:cuda()
        --Moving everything onto the GPU
        self.feature_extractor:cuda()

        if self.domain_predictor then
            self.domain_predictor:cuda()
            self.domain_criterion:cuda()
            if self.info_predictor then
                self.info_predictor:cuda()
            end
            if self.info_criterion then
                self.info_criterion:cuda()
            end
        end
        self.label_predictor:cuda()
        self.label_criterion:cuda()
    end

    function module:cudnn()
        self:cuda()
        cudnn.convert(self.feature_extractor, cudnn)
        cudnn.convert(self.label_predictor, cudnn)
        if self.domain_predictor then
            cudnn.convert(self.domain_predictor, cudnn)
            if self.info_predictor then
                cudnn.convert(self.info_predictor, cudnn)
            end
        end
    end

    function module:clearState()
        self.feature_extractor:clearState()
        self.label_predictor:clearState()
        if self.domain_predictor then
            self.domain_predictor:clearState()
            if self.info_predictor then
                self.info_predictor:clearState()
            end
        end
    end

    function module:clone()
        local clone={}
        clone.feature_extractor = self.feature_extractor:clone()
        clone.label_predictor = self.label_predictor:clone()
        clone.inputsize = self.inputsize
        clone.name = self.name

        function clone:evaluate()
            general_model.evaluate(self.feature_extractor)
            general_model.evaluate(self.label_predictor)
        end
        function clone:double()
            self.feature_extractor:double()
            self.label_predictor:double()
        end
        function clone:clearState()
            self.feature_extractor:clearState()
            self.label_predictor:clearState()
        end

        return clone
    end

    module:training()--It should already be the case but never too cautious
    return module
end

local result = {}
result.getmodel = getmodel
return result

