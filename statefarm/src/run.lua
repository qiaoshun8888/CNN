require 'pl'
require 'trepl'
require 'torch'
require 'image'
require 'nn'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Processing options')

opt = lapp[[
   -r,--learningRate       (default 1e-1)        learning rate
   -d,--learningRateDecay  (default 1e-5)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-3)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full
   -o,--save               (default results)     save directory
      --optimization       (default SGD)         'optimization method: SGD | ASGD | CG | LBFGS')
      --t7Path             (default t7_images)   the path of t7 files for training and test images
      --maxIters           (default 23)           maximum number of iterations
      --patches            (default all)         percentage of samples to use for testing'
      --model              (default convnet)     type of model to construct: linear | mlp | convnet
      --loss               (default nll)         type of loss function to minimize: nll | mse | margin
      --randRTDA           (default false)        randomly do real time data argumentation (RTDA) during training
      --visualize          (default false)        visualize dataset
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> Switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red .. '==> Using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> executing all')

dofile 'data.lua'
dofile 'model.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Training!')

for i=1,opt.maxIters do
   train(data.trainData)
   test(data.testData)
   if i > 20 then
      correct_rate_eval()
   end
end
