skinRGMaxGap = 20/255  -- maximum gap between R and G
skinRBMaxGap = 20/255 -- maximum gap between R and B
skinRMin = 40/255  -- minimum R value
skinRGBWriteFilter = 200/255  -- white filter for R, G and B

function skinFilter(img)
    -- img[{{3}, {}, {}}]:fill(0)
    -- img[{{2}, {}, {}}]:fill(0)

    -- skin filter
    img2 = img:clone()
    img2[{{2}, {}, {}}] = img[{{1}, {}, {}}] - img[{{2}, {}, {}}]  -- R - G
    img2[{{3}, {}, {}}] = img[{{1}, {}, {}}] - img[{{3}, {}, {}}]  -- R - B
    -- img2[{{1}, {}, {}}]:fill(0)
    -- img2[{{1}, {}, {}}] = img[{{1}, {}, {}}]:fill(0)

    -- img[{{2}, {}, {}}] = img[{{1}, {}, {}}] - img[{{2}, {}, {}}]  -- R - G
    -- img[{{3}, {}, {}}] = img[{{1}, {}, {}}] - img[{{3}, {}, {}}]  -- R - B
    -- img[{{1}, {}, {}}] = img[{{1}, {}, {}}]:fill(0)
        
    -- filter colors for G,B channels

    img2[{{2}, {}, {}}] = torch.clamp(img2[{{2}, {}, {}}], 0, 1)  -- remove negative
    img2[{{3}, {}, {}}] = torch.clamp(img2[{{3}, {}, {}}], 0, 1)  -- remove negative

    -- print(img2)
        
    for i=1,imageHeight do
        for j=1,imageWidth do
            local r = torch.abs(img2[1][i][j])
            local r_g = torch.abs(img2[2][i][j])
            local r_b = torch.abs(img2[3][i][j])

            if 
                r   < skinRMin or 
                r_g > skinRGMaxGap or
                r_g == 0 or
                r_b == 0 or
                (img[1][i][j] > skinRGBWriteFilter and img[2][i][j] > skinRGBWriteFilter and img[3][i][j] > skinRGBWriteFilter)
                then
                -- r_b > skinRBMaxGap then


            -- if img2[2][i][j] > skinRGBMaxGap 
                -- or img2[3][i][j] > skinRGBMaxGap
                -- or img2[2][i][j] == 0 
                -- or img2[3][i][j] == 0 then
                    img[{{}, {i}, {j}}] = 0
--                     img[1][i][j] = 0
--                     img[2][i][j] = 0
--                     img[3][i][j] = 0
                else
                    -- img[{{}, {i}, {j}}] = 1
            end
        end
    end
end