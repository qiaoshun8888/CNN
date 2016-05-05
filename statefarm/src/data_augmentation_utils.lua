-- copy m2 horizontal right half to m1
-- m1: source image
-- m2: taraget image
-- nc: number of the channels
function horizontal_r_cp(m1, m2, nc)
	local range = imageWidth/2
	nc = nc or 1
	for i=1,range do
		for j=1,nc do
			m1[{{j},{},{range + i}}] = m2[{{j},{},{range + i}}]
		end
	end
end

-- copy m2 vertical bottom half to m1
-- m1: source image
-- m2: taraget image
-- nc: number of the channels
function vertical_b_cp(m1, m2, nc)
	local range = imageHeight/2
	nc = nc or 1
	for i=1,range do
		for j=1,nc do
			m1[{{j},{range + i},{}}] = m2[{{j},{range + i},{}}]
		end
	end
end

-- copy m2 vertical top half to m1
-- m1: source image
-- m2: taraget image
-- nc: number of the channels
function vertical_t_cp(m1, m2, nc)
	local range = imageHeight/2
	nc = nc or 1
	for i=1,range do
		for j=1,nc do
			m1[{{j},{i},{}}] = m2[{{j},{i},{}}]
		end
	end
end

-- copy m2 top-left half of diagonal(/) to m1
-- m1: source image
-- m2: taraget image
-- nc: number of the channels
function diagonal_tl_cp(m1, m2, nc)
	nc = nc or 1
	for i=1,imageHeight do  -- top -> bottom
		for j=1,imageWidth-i+1 do  -- left -> right
			for k=1,nc do
				m1[{{k},{i},{j}}] = m2[{{k},{i},{j}}]
			end
		end
	end
end

-- copy m2 top-right half of diagonal(\) to m1
-- m1: source image
-- m2: taraget image
-- nc: number of the channels
function diagonal_tr_cp(m1, m2, nc)
	nc = nc or 1
	for i=1,imageHeight do  -- top -> bottom
		for j=1,i do  -- left -> right
			for k=1,nc do
				m1[{{k},{i},{j}}] = m2[{{k},{i},{j}}]
			end
		end
	end
end

function da_randomize(input, target)
	-- randomly augment data set -- BEGIN
    local trainDataClassified
    if target == 1 then
    	trainDataClassified = trainDataClassified1
    elseif target == 2 then
    	trainDataClassified = trainDataClassified2
	elseif target == 3 then
        trainDataClassified = trainDataClassified3
    elseif target == 4 then
		trainDataClassified = trainDataClassified4
	elseif target == 5 then
		trainDataClassified = trainDataClassified5
	elseif target == 6 then
		trainDataClassified = trainDataClassified6
	elseif target == 7 then
		trainDataClassified = trainDataClassified7
	elseif target == 8 then
		trainDataClassified = trainDataClassified8
	elseif target == 9 then
		trainDataClassified = trainDataClassified9
	elseif target == 10 then
		trainDataClassified = trainDataClassified10
	end
	math.randomseed(os.time())

	local enable_set = {math.random(1,2),2,2,2,2}  -- 50% apply one DA function
	local shuffledIndices = torch.randperm((#enable_set))
	for i=1,#enable_set do
		local enable = enable_set[shuffledIndices[i]] == 1
		if enable then
			if i == 1 then
				horizontal_r_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
			elseif i == 2 then
				vertical_b_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
			elseif i == 3 then
				vertical_t_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
			elseif i == 4 then
				diagonal_tl_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
			elseif i == 5 then
				diagonal_tr_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
			end
		end
	end

	--[[
	local horizontal_r_cp_enable = math.random(1, 2)
	if horizontal_r_cp_enable == 1 then
		horizontal_r_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
	end

	local vertical_b_cp_enable = math.random(1, 2)
	if vertical_b_cp_enable == 1 then
		vertical_b_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
	end

	local vertical_t_cp_enable = math.random(1, 2)
	if vertical_t_cp_enable == 1 then
		vertical_t_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
	end

	local diagonal_tl_cp_enable = math.random(1, 2)
	if diagonal_tl_cp_enable == 1 then
		diagonal_tl_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
	end

	local diagonal_tr_cp_enable = math.random(1, 2)
	if diagonal_tr_cp_enable == 1 then
		diagonal_tr_cp(input, trainDataClassified[math.random(1, #trainDataClassified)])
	end
	]]--
end
