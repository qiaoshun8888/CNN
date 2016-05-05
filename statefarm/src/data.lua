require 'torch'
require 'image'
require 'nn'
require 'cunn'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
    print '==> processing options'
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('StateFarm Dataset Preprocessing')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-size', 'small', 'how many samples do we load: small | full')
    cmd:option('-batchSize', 128, 'batch size')
    cmd:option('-type', 'float', 'float or cuda')
    cmd:option('-t7Path', 't7_images', 'the path of t7 files for training and test images')
    cmd:option('-visualize', false, 'visualize input data and weights during training')
    cmd:option('-testSample', false, 'use pretrained model to test one sample data')
    cmd:option('-pretrainedModel', 'results/model.net', 'pretrained model')
    cmd:option('-kaggleSubmission', false, 'whether to use a pretrained model to predicate kaggle test data')
    cmd:option('-kaggleTestData', 'imgs/test', 'the path of kaggle test data')
    cmd:text()
    opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- Define variables and flags --
----------------------------------------------------------------------
opt.size = 'small'
opt.type = 'float'

portionTrain = 0.8 -- 80% is train data, rest is test data
imageWidth = 32
imageHeight = 32
numberOfChannel = 1

classes = {
    'normal driving',                -- c0
    'texting - right',               -- c1
    'talking on the phone - right',  -- c2
    'texting - left',                -- c3
    'talking on the phone - left',   -- c4
    'operating the radio',           -- c5
    'drinking',                      -- c6
    'reaching behind',               -- c7
    'hair and makeup',               -- c8
    'talking to passenger'           -- c9
}
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Load files and util functions --
----------------------------------------------------------------------
function ls(path) return sys.split(sys.ls(path),'\n') end
function max(a, b) if a > b then return a else return b end end
function min(a, b) if a < b then return a else return b end end
local pl = require('pl.import_into')()
dofile 'skin_filter.lua'
dofile 'data_augmentation_utils.lua'
dofile 'correct_rate_eval.lua'
----------------------------------------------------------------------

trainFile = paths.concat(opt.t7Path, 'train_' .. opt.size .. '_' .. imageWidth .. 'x' .. imageHeight .. '.t7')
testFile = paths.concat(opt.t7Path, 'test_' .. opt.size .. '_' .. imageWidth .. 'x' .. imageHeight .. '.t7')

if opt.size == 'small' then
    print '==> Using small size training data for fast experiments'
    dataSize = 1000  -- Data size for each class
else
    print '==> Using full size training data'
    dataSize = 9999999  -- Data size for each class
end

----------------------------------------------------------------------
-- test sample
opt.pretrainedModel = 'results/model.net'
-- opt.pretrainedModel = 'results/backup/1.98149/model20160412_1_ 1.98149.net'
opt.testSample = false
opt.kaggleSubmission = false
if opt.testSample or opt.kaggleSubmission then
    if opt.size == 'small' then 
        trainedMean = 0.31424253774215
        trainedStd = 0.27076773411736
    else
        trainedMean = 0.31421060906608
        trainedStd = 0.27087070537414
    end
    
    model = torch.load(opt.pretrainedModel)
    
    if opt.type == 'cuda' then
        model = model:cuda()
    elseif opt.type == 'float' then
        model = model:float()
    end
    
    model:evaluate()
    local inputs = torch.Tensor(opt.batchSize, numberOfChannel, imageWidth, imageHeight):float()
    
    if opt.testSample then
        correct_rate_eval()
--         print("==> prediction:")
--         for i=1,confidences:size(2) do
--             print(indices[1][i] .. ' - ' .. confidences[1][i] * 100 .. '%' .. ' ' .. classes[indices[1][i]])
--         end
    elseif opt.kaggleSubmission then
        if not Csv then dofile 'Csv.lua' end
        local separator = ','  -- optional; use if not a comma
        local csv = Csv("results/submission.csv", "w", separator)
        csv:write({'img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'}) -- write header
        local total = #ls(opt.kaggleTestData)
        local fileTable = {}
        for i,file in ipairs(pl.dir.getallfiles(opt.kaggleTestData, '*.jpg')) do
            smallTestImage = image.load(file)
            smallTestImage = image.scale(smallTestImage,imageWidth,imageHeight)[1]:float()
            smallTestImage:add(-trainedMean)
            smallTestImage:div(trainedStd)
            
            -- 17: 5 5 5 2
            -- total - i > opt.batchSize - i % opt.batchSize
            if total -i > opt.batchSize - i % opt.batchSize or i % opt.batchSize == 0 then
                -- batch predication
                if i % opt.batchSize == 0 then
                    fileTable[opt.batchSize] = file:match( "([^/]+)$" )
                    inputs[opt.batchSize] = smallTestImage
                    if opt.type == 'cuda' then
                        inputs = inputs:cuda()
                    elseif opt.type == 'float' then
                        inputs = inputs:float()
                    end
                    local prediction = model:forward(inputs):exp()
                    for j=1,prediction:size(1) do
                        local resultTable = {}
                        resultTable[1] = fileTable[j]
                        for k=1,#classes do
                            resultTable[k + 1] = prediction[j][k]
                        end
                        csv:write(resultTable)
                    end
                    print('batch')
                else
                    fileTable[i % opt.batchSize] = file:match( "([^/]+)$" )
                    inputs[i % opt.batchSize] = smallTestImage
                end
            else
                -- single predication
                print('single')
                for j = 1,opt.batchSize do
                    inputs[j] = smallTestImage
                end
                
                if opt.type == 'cuda' then
                    inputs = inputs:cuda()
                elseif opt.type == 'float' then
                    inputs = inputs:float()
                end
                
                local prediction = model:forward(inputs):exp()[{1, {}}]
                prediction = torch.reshape(prediction, 1, 10)
                local resultTable = {}
                resultTable[1] = file:match( "([^/]+)$" )
                for j=1,#classes do
                    resultTable[j+1] = prediction[1][j]
                end
                csv:write(resultTable)
                
                print(i .. '/' .. total)
            end
            
            if i % 1000 == 0 then
                print(i .. '/' .. total)
                collectgarbage()
            end
        end
        csv:close() 
    end
    
    do return end  -- exit the script and skip all the code after this line
end

----------------------------------------------------------------------
-- Load training file and test file
-- If .t7 file doesn't exist, create a new one.
trainData = {}
testData = {}
print(sys.COLORS.red .. '==> Loading data set from ' .. sys.dirname(trainFile))
if not paths.filep(trainFile) or not paths.filep(testFile) then
    print '==> Create .t7 files for training data...'
    local allImagePaths = {}
    local allLabels = {}
    local count = 1
    for i = 0,#classes - 1 do
        print('==> class ' .. i)
        local dataPath = 'imgs/train/c' .. i
        if dataSize < #ls(dataPath) then
            local allImagePathsForOneClass = {}
            for j,file in ipairs(pl.dir.getallfiles(dataPath, '*.jpg')) do
                allImagePathsForOneClass[j] = file
            end
            -- Shuffle the index and copy the first dataSize data into allImagePaths
            local shuffledIndices = torch.randperm((#allImagePathsForOneClass))
            for j = 1,dataSize do
                allImagePaths[count] = allImagePathsForOneClass[shuffledIndices[j]]
                allLabels[count] = i + 1
                count = count + 1
            end
        else
            for j,file in ipairs(pl.dir.getallfiles(dataPath, '*.jpg')) do
                allImagePaths[count] = file
                allLabels[count] = i + 1
                count = count + 1
            end
        end
    end
    
    -- shuffle dataset: get shuffled indices in this variable:
    local shuffledIndices = torch.randperm(#allImagePaths)
    local trainingSize = torch.floor(shuffledIndices:size(1)*portionTrain)
    local testSize = shuffledIndices:size(1) - trainingSize
    
    print '==> step 1'
    
    -- Note: the data, in trainData or testData is 4-d: 
    -- the 1st dim indexes the samples, 
    -- the 2nd dim indexes the color channels (RGB)
    -- the last two dims index the height and width of the samples.
    
    -- create train set:
    trainData = {
       data = torch.Tensor(trainingSize, numberOfChannel, imageWidth, imageHeight),
       labels = torch.Tensor(trainingSize),
       size = function() return trainingSize end
    }
    --create test set:
    testData = {
        data = torch.Tensor(testSize, numberOfChannel, imageWidth, imageHeight),
        labels = torch.Tensor(testSize),
        size = function() return testSize end
    }
    
    print '==> step 2'
    
    for i=1,trainingSize do
        local img = image.load(allImagePaths[shuffledIndices[i]])
        img = image.scale(img,imageWidth,imageHeight):float()  -- 3x32x32 -> 1x32x32
        trainData.data[i] = img[1]  -- three channels are the same, either one is fine
        trainData.labels[i] = allLabels[shuffledIndices[i]]
        if i % 100 == 0 then
            collectgarbage()
        end
    end
    
    print '==> step 3'
    
    for i=trainingSize+1,trainingSize+testSize do
        local img = image.load(allImagePaths[shuffledIndices[i]])
        img = image.scale(img,imageWidth,imageHeight):float()  -- 3x32x32 -> 1x32x32
        testData.data[i-trainingSize] = img[1]
        testData.labels[i-trainingSize] = allLabels[shuffledIndices[i]]
        if i % 100 == 0 then
            collectgarbage()
        end
    end
    
    print '==> step 4'
    
    -- Store the data into files
    os.execute('mkdir -p ' .. sys.dirname(trainFile))
    print('==> Saving training data in ' .. trainFile)
    torch.save(trainFile, trainData, 'binary')
    print('==> Saving test data in ' .. testFile)
    torch.save(testFile, testData, 'binary')
    
    collectgarbage()
else
    trainData = torch.load(trainFile)
    testData = torch.load(testFile)
end

print('training data size: ' .. trainData:size())
print('test data size: ' .. testData:size())

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Preprocessing data')

trainData.data = trainData.data:float()
testData.data = testData.data:float()

--[[
-- The data set is already in grayscale, so there is no need to convert RGB -> YUV.
-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end
]]--

-- Name channels for convenience
-- local channels = {'y','u','v'}
local channels = {'R'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red .. '==> preprocessing data: normalize each feature (channel) globally')

mean = {}
std = {}

for i,channel in ipairs(channels) do
    -- normalize each channel globally
    mean[i] = trainData.data[{ {},i,{},{} }]:mean()
    std[i] = trainData.data[{ {},i,{},{} }]:std()
    trainData.data[{ {},i,{},{} }]:add(-mean[i])
    trainData.data[{ {},i,{},{} }]:div(std[i])
end

trainedMean = mean[1]
trainedStd = std[1]
print('mean: ' .. trainedMean)
print('std: ' .. trainedStd)

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print(sys.COLORS.red .. '==> preprocessing data: normalize all three channels locally')

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(19)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> Verify statistics')

-- It's always good practice to verify that data is properly
-- normalized.
for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, ' .. channel .. '-channel, mean: ' .. trainMean)
   print('training data, ' .. channel .. '-channel, standard deviation: ' .. trainStd)

   print('test data, ' .. channel .. '-channel, mean: ' .. testMean)
   print('test data, ' .. channel .. '-channel, standard deviation: ' .. testStd)
end

trainDataClassified1 = {}
trainDataClassified2 = {}
trainDataClassified3 = {}
trainDataClassified4 = {}
trainDataClassified5 = {}
trainDataClassified6 = {}
trainDataClassified7 = {}
trainDataClassified8 = {}
trainDataClassified9 = {}
trainDataClassified10 = {}
for i=1,#classes do
    for j=1,trainData:size() do
        local data = trainData.data[j] 
        local label = trainData.labels[j]
        if i == label then
            if label == 1 then
                trainDataClassified1[#trainDataClassified1+1] = data
            elseif label == 2 then
                trainDataClassified2[#trainDataClassified2+1] = data
            elseif label == 3 then
                trainDataClassified3[#trainDataClassified3+1] = data
            elseif label == 4 then
                trainDataClassified4[#trainDataClassified4+1] = data
            elseif label == 5 then
                trainDataClassified5[#trainDataClassified5+1] = data
            elseif label == 6 then
                trainDataClassified6[#trainDataClassified6+1] = data
            elseif label == 7 then
                trainDataClassified7[#trainDataClassified7+1] = data
            elseif label == 8 then
                trainDataClassified8[#trainDataClassified8+1] = data
            elseif label == 9 then
                trainDataClassified9[#trainDataClassified9+1] = data
            elseif label == 10 then
                trainDataClassified10[#trainDataClassified10+1] = data
            end
        end
    end
end

print(#trainDataClassified1)
print(#trainDataClassified2)
print(#trainDataClassified3)
print(#trainDataClassified4)
print(#trainDataClassified5)
print(#trainDataClassified6)
print(#trainDataClassified7)
print(#trainDataClassified8)
print(#trainDataClassified9)
print(#trainDataClassified10)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().
if opt.visualize then
    print(sys.COLORS.red .. '==> Visualizing data')
    if itorch then
    first256Samples_y = trainData.data[{ {1,256},1 }]
--     first256Samples_u = trainData.data[{ {1,256},2 }]
--     first256Samples_v = trainData.data[{ {1,256},3 }]
    itorch.image(first256Samples_y)
--     itorch.image(first256Samples_u)
--     itorch.image(first256Samples_v)
    else
       print("For visualization, run this script in an itorch notebook")
    end
end
