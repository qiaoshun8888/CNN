require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique

if not opt then
    print '==> processing options'
    cmd = torch.CmdLine()
    cmd:text('Options:')
    cmd:option('-type', 'float', 'float or cuda')
    cmd:option('model', 'convnet', 'type of model to construct: linear | mlp | convnet')
    cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:text()
    opt = cmd:parse(arg or {})
    imageWidth = 128
    imageHeight = 128
end

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Define parameters')

-- 10-class problem: faces!
local noutputs = 10

-- input dimensions
nfeats = 1
width = imageWidth
height = imageHeight
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {32,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

-- Input dimensions
-- local nfeats = numberOfChannel
-- local width = imageWidth
-- local height = imageHeight

-- -- Hidden units, filter sizes (for ConvNet only)
-- local nstates = {16,32}
-- local filtsize = {5, 7}
-- local poolsize = 4

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Construct model')

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet' then
   -- a typical convolutional network, with locally-normalized hidden
   -- units, and L2-pooling

   -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
   -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
   -- the use of LP-pooling (with P=2) has a very positive impact on
   -- generalization. Normalization is not done exactly as proposed in
   -- the paper, and low-level (first layer) features are not fed to
   -- the classifier.

   model = nn.Sequential()
   --[[ Kaggle Score: 2.41928 ]]--
   -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
   -- (W−F+2P)/S+1
   model:add(nn.SpatialConvolutionMM(1, 32, 3, 3))  -- (32-3+2*0)/1 + 1 = 30
   model:add(nn.ReLU())
   -- W2=(W1−F)/S+1
   -- H2=(H1−F)/S+1
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  -- (30-2)/2 + 1 = 15

   model:add(nn.SpatialConvolutionMM(32, 64, 2, 2))  -- (15-2+2*0)/1 + 1 = 14
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(64, 128, 3, 3))  -- (14-3+2*0)/1 + 1 = 12
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  -- (12-2)/2 + 1 = 6

   model:add(nn.SpatialConvolutionMM(128, 256, 2, 2))  -- (6-2+2*0)/1 + 1 = 5
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(256, 512, 2, 2))  -- (5-2+2*0)/1 + 1 = 4
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  -- (4-2)/2 + 1 = 2

   -- stage 3 : standard 2-layer MLP:
   model:add(nn.View(512*2*2))
   -- model:add(nn.Reshape(128*2*2))
   model:add(nn.Linear(512*2*2, 256*2*2))
   model:add(nn.ReLU())
   model:add(nn.Linear(256*2*2, noutputs))

   -- -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
   -- model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
   -- model:add(nn.Tanh())
   -- model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- model:add(nn.Dropout(0.25))

   -- model:add(nn.SpatialConvolutionMM(64, 128, 2, 2))
   -- model:add(nn.Tanh())
   -- -- model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- model:add(nn.Dropout(0.25))

   



   --[[ Kaggle Score: 1.98149 ]]--
   --[[
   -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
   -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- stage 3 : standard 2-layer MLP:
   model:add(nn.Reshape(64*2*2))
   model:add(nn.Linear(64*2*2, 200))
   model:add(nn.Tanh())
   model:add(nn.Linear(200, noutputs))
   ]]--

   --[[
   -- https://github.com/lisabug/Cifar10-Torch7/blob/master/model.lua
   -- convolution layers
   model:add(nn.SpatialConvolutionMM(1, 128, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   model:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   model:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1))
   model:add(nn.ReLU())

   -- fully connected layers
   model:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1))
    
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())
   ]]--


   --[[
   -- https://github.com/nagadomi/kaggle-ndsb/blob/master/cnn_96x96.lua
   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(1, 64, 5, 5))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
   model:add(nn.Dropout(0.25))

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(64, 64, 5, 5))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
   model:add(nn.Dropout(0.25))
   
   -- stage 3 : standard 2-layer neural network
   model:add(nn.View(64*5*5))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(64*5*5, 128))
   model:add(nn.ReLU())
   model:add(nn.Linear(128, 10))
   ]]--

   --[[
   -- stage 1 : 
   model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
   model:add(nn.ReLU())  -- model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))  -- model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))

   -- stage 2 : 
   model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
      
   -- stage 3 : 
   model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize, filtsize))
   model:add(nn.ReLU())
   -- model:add(nn.SpatialConvolutionMM(nstates[3], nstates[4], filtsize, filtsize))
   -- model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
   model:add(nn.Dropout(0.25))

   -- model:add(nn.Reshape(nstates[2]*2*2))
   -- model:add(nn.Linear(nstates[2]*2*2, nstates[3]))
   -- model:add(nn.Tanh())
   -- model:add(nn.Linear(nstates[3], noutputs))
   local size = nstates[3] * 2 * 2
   model:add(nn.View(size))
   model:add(nn.Linear(size, size))  -- nstates[4]))
   -- model:add(nn.ReLU())
   -- model:add(nn.Dropout(0.5))
   -- model:add(nn.Linear(nstates[4], nstates[4]))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(size, noutputs))
   -- model:add(nn.Reshape(noutputs))

   model:add(nn.SoftMax())

   ]]--
else

   error('unknown -model')

end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Model:')
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
    print '==> visualizing ConvNet filters'
    print('Layer 1 filters:')
    itorch.image(model:get(1).weight)
    print('Layer 2 filters:')
    itorch.image(model:get(5).weight)
      else
    print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
