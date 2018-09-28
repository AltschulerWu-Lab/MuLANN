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
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      print('Setting random seed on donkeys as well')
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function(threadid)
            require 'torch'
            hasCuda, cuda = pcall(require, 'cutorch')
            hasCudnn, cudnn = pcall(require, 'cudnn')
            require 'pl'
            require 'image'
            require 'nn'
            package.path = package.path..';../?.lua'
            require 'settings'
            package.path = package.path
        .. string.format(';%s/luasrc/data/?.lua' , code_folder)
        .. string.format(';%s/luasrc/models/?.lua' , code_folder)
        .. string.format(';%s/luasrc/utilitaries/?.lua', code_folder)
            general_data = require 'general_data'
            utils = require 'utils'
         end,

         function(idx)
            opt = options -- pass to all donkeys via upvalue
            utils.setSeed(opt.seed+__threadid)
         end
      );
   else -- single threaded data loading. useful for debugging
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

