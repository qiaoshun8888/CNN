require 'torch'
require 'image'
require 'qt'
local pl = require('pl.import_into')()

local answer
for i,f in ipairs(pl.dir.getfiles('imgs/small_tests', '*.jpg')) do
    image.display(image.load(f))
    io.write("Input class id 0-9: ")
    io.flush()
    print('File: ' .. f)
    answer=io.read()
    cmd = 'mv ' .. f .. ' imgs/small_tests/c' .. answer .. '/'
    print(sys.COLORS.green .. cmd)
    os.execute(cmd)
end
