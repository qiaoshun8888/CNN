local pl = require('pl.import_into')()
local inputs = torch.Tensor(opt.batchSize, numberOfChannel, imageWidth, imageHeight):float()

function correct_rate_eval()
    print(sys.COLORS.green .. '==> Evaluate correct rate')
    local avgCorrect = 0
    for i=1,#classes do
        local correctCount = 0
        local totalCount = 0
        for j,f in ipairs(pl.dir.getfiles('imgs/small_tests/c' .. (i-1), '*.jpg')) do
            smallTestImage = image.load(f)
            smallTestImage = image.scale(smallTestImage,imageWidth,imageHeight)
            smallTestImage = smallTestImage[1]
            smallTestImage:add(-trainedMean)
            smallTestImage:div(trainedStd)
            for k = 1,opt.batchSize do
                inputs[k] = smallTestImage
            end
            
            if opt.type == 'cuda' then
                inputs = inputs:cuda()
            elseif opt.type == 'float' then
                inputs = inputs:float()
            end
            
            local prediction = model:forward(inputs):exp()[{1, {}}]
            prediction = torch.reshape(prediction, 1, 10)
            local confidences, indices = torch.sort(prediction, true)

            totalCount = totalCount + 1
            if indices[1][1] == i then
                correctCount = correctCount + 1
            end
        end
        print('c' .. (i - 1) .. ' correct rate: ' .. (correctCount/totalCount))
        avgCorrect = avgCorrect + correctCount/totalCount
        collectgarbage()
    end
    avgCorrect = avgCorrect/#classes
    print('Average correct rate: ' .. avgCorrect)
    return avgCorrect
end