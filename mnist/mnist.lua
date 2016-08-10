--[[
Modified from example mnist execution code in torchnet.
Copyright (c) 2016-present, Facebook, Inc.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
]]--

--torch.setdefaulttensortype('torch.FloatTensor')

--TODO: generalize training procedure to pairs of images for siamese 2AFC training
   -- TODO: remove note for Metacurriculum learning project

--TODO: given a batch, figure out which pairs of images to use for the comparison. Basically, the siamese thing only requires
-- across-image comparisons at the very end.

-- load torchnet:
local tnt = require 'torchnet'
local debugger = require('fb.debugger')

local architectures = require('../utils/architectures')
local weight_init = require('../utils/weight-init')

-- Command line options
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-perm_invar', false, 'permutation invariant: reshape to 1d')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-mu', 0.0, 'SGD Momentum') -- Set to zero for the moment
cmd:option('-maxepoch', 5, 'Maximum number of epochs to run')
cmd:option('-batchsize',128, 'Batch size')
cmd:option('-augMode','','Data augmentation mode')
cmd:option('-normMode','mnistZeroMean','Input data normalization')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

-- TODO: migrate this to a separate .lua file
local function spatialTransform()
   -- Returns the spatial transformation function used to augment the dataset
   -- on the fly. Allows either batch- or instance-level randomization of augmentation
   -- parameters.
   local transformFunction

   if config.augMode == 'Cir2010' then
      -- TODO: implement this augmentation strategy
      -- Epochwise augmentation, so initialize the random parameters outside of the function
      -- All parameters sampled uniformly over the specified range
      -- Elastic deformation params sigma=[5.0,6.0], alpha=[36.0,38.0] (see Simard et al 2003)
      -- Rotation/horiz shearing: beta=[-7.5 deg, 7.5 deg] for 1 and 7, beta=[-15 deg, 15 deg] for others
      -- Horiz/vert scaling: gamma = [15,20], given as [1-gamma/100,1+gamma/100], independent scaling in x and y
      -- transformFunction =
   elseif config.augMode == 'Cir2012' then
      -- TODO: implement this augmentation strategy
      -- More elaborate: changes bounding box on samples before transforming,
      -- and returns each different-sized one to a different subnetwork.
      -- Low implement priority.
   elseif config.augMode == 'Wan2013' then
      -- TODO: implement this augmentation strategy
      -- Random Cropping to 24x24
      -- Rotate/scale up to 15%
   else
      -- Identity transformation
      -- e.g. for Goodfellow et al Maxout Nets
      transformFunction = function(sample) return sample end
   end

   return transformFunction
end

local function dataNormTransform()
   -- Returns the normalizing transformation used on data.
   local normFunction

   if config.normMode == 'mnistZeroMean' then
      normFunction = function(sample)
         return {
            input = sample.input/127.5 - 1.0,
            target = sample.target,
         }
      end
   else
      -- Identity transformation
      normFunction = function(sample) return sample end
   end

   return normFunction
end

-- function that sets of dataset iterator:
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      -- transform = GetTransforms(config.transformMode), -- Use tnt.TransformDataset instead of this
      init    = function() require 'torchnet' end,
      closure = function() -- will repeatedly call dataset:get()

         -- load MNIST dataset:
         local mnist = require 'mnist'
         local dataset = mnist[mode .. 'dataset']()
         local augmentFunction = spatialTransform()
         local normFunction = dataNormTransform()

         -- Make images 1d: full dataset
         if config.perm_invar then
            dataset.data = dataset.data:reshape(dataset.data:size(1),
               dataset.data:size(2) * dataset.data:size(3)):float()
         else
            dataset.data = dataset.data:float()
         end

         -- Duplicate labels as doubles for regression
         -- TODO: remove for Metacurriculum learning project
         --dataset.intlabel = torch.FloatTensor(dataset.label:size()):copy(dataset.label)

         -- return batches of data:
         return tnt.BatchDataset{
            batchsize = config.batchsize, -- Can get this > 10k with no trouble
            dataset = tnt.TransformDataset { -- apply transform closure
               transform = function(sample)
                  sample = augmentFunction(normFunction(sample))
                  return {
                     input = sample.input,
                     target = sample.target,
                  }
               end, -- closure for transformation
               dataset = tnt.ShuffleDataset { -- Always shuffle w/ replacement each epoch
                  dataset = tnt.ListDataset{  -- replace this by your own dataset
                     list = torch.range(1, dataset.data:size(1)):long(),
                     load = function(idx)

                     local input
                     if config.perm_invar then
                        input = dataset.data[idx]
                     else
                        input = dataset.data[{{idx},{},{}}]
                     end
                        return {
                           input = input,
                           target = torch.LongTensor{dataset.label[idx] + 1},
                        }  -- sample contains input and target
                     end,
                  }
               }
            }
         }
      end,
   }
end

-- set up logistic regressor:
local xDim = 28 -- Note, will be 24x24 for Wan2013
local nLabels = 10
local nImChans = 1
local net
if config.perm_invar then
   net = architectures.Cir2010_4_ReLU(xDim^2,nLabels)
else
   net = architectures.Wan2013_CNN(nImChans,nLabels,xDim)
end

net = weight_init(net,'kaiming')

local criterion = nn.CrossEntropyCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter() -- Low better
local clerr  = tnt.ClassErrorMeter{topk = {1}} -- Low better
engine.hooks.onStartEpoch = function(state)
   -- Add epoch-wise evaluation here.
   meter:reset()
   clerr:reset()
end
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
         meter:value(), clerr:value{k = 1}))
   end
end

-- set up GPU training:
if config.usegpu then

   -- copy model to GPU:
   require 'cunn'
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- train the model:
engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion,
   lr        = config.lr,
   maxepoch  = config.maxepoch,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}
print(string.format('test loss: %2.4f; test error: %2.4f',
   meter:value(), clerr:value{k = 1}))
