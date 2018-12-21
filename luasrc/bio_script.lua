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

package.path = package.path .. ';../?.lua'
opt = {}
require 'settings'
require 'main'
opt.dataset = 'BIO'
data_level = require 'bio_data'
opt.model = 'VGG'
opt.target_unlabelled_presence = true
opt.target_supervision_level = 50
opt.difficulty_level = 'easy'
opt.iterative_labeling = 'entropy'
opt.unknown_perc = 0.7

local folder = path.join (rundir, 'office')

opt.eta0 = tonumber(arg[1])
opt.source = arg[2]
if opt.source:lower()=='england' then opt.source="Caie" else opt.source = opt.source:lower()=='california' and 'UCSF' or 'UTSW' end
opt.target = arg[3]
if opt.target:lower()=='england' then opt.target="Caie" else opt.target = opt.target:lower()=='california' and 'UCSF' or 'UTSW' end

opt.domain_adaptation = false
opt.domainLambda = 0
if arg[4] and arg[4]~='-1' then
    opt.domain_adaptation = true
    opt.domainLambda = tonumber(arg[4])
end

opt.fold = (arg[5] and tonumber(arg[5])) or 0
--Meaning if we're using all train+test during training
opt.fully_transductive = (arg[6]=='true' and true) or false

opt.domain_method = arg[7] and arg[7] or opt.domain_method
if opt.domain_method=="MADA" then
    opt.lr_decay = true
    opt.indiv_lr = true
    opt.train_setting = 1
elseif opt.domain_method=="DANN" then
    if opt.source=='UCSF' then
        opt.lr_decay = true
        opt.indiv_lr = false
        opt.train_setting = 0
    else
        opt.lr_decay = false
        opt.indiv_lr = true
        opt.train_setting = 1
    end
elseif opt.domain_method=="MuLANN" then
    opt.train_setting = 0
    opt.indiv_lr = false
    opt.lr_decay = false
    if opt.source=="UTSW" then opt.lr_decay = true end
end

if arg[8] and arg[8]~='nil' then
   opt.zeta = tonumber(arg[8])
end
if arg[9] and arg[9]~='nil' then
    opt.unknown_perc = arg[9] and tonumber(arg[9]) or nil
end
if arg[10] then
    --This can be symm, labelled, unlabelled, full
    opt.noise = arg[10]~='nil' and arg[10] or nil
    require 'main_asym_experiments'
end

local kw = ''
if opt.target_supervision_level==50 then
    if opt.target_unlabelled_presence then kw = kw..'unlab' else kw = kw..'semisupPureLab' end
    if opt.noise then kw = kw..'noise'..opt.noise end
else
    kw = kw..'supervised'
end
if not opt.fully_transductive then kw = kw..'_nontransductive' end
local kw2 = (opt.domain_method=='MuLANN' and 'Mu') or (opt.domain_method=='DANN' and 'DANN') or (opt.domain_method=='MADA' and 'MADA') or ''
if opt.domain_method=='MuLANN' then
    kw2 = (opt.iterative_labeling=='entropy' and kw2..'entropy' or kw2..'diff') ..string.format('%s', opt.unknown_perc) end

folder = folder..kw2..opt.model..string.format('_%s_', kw)..opt.source..'_'..opt.target
if opt.source=='both' then opt.source = 'UTSW' opt.extra_domain ='UCSF' end
produce_model(folder)
