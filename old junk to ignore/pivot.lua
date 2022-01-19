local pivot_x = nil
local pivot_y = nil
local pivot_z = nil
local pivot_index = mainmemory.read_s32_be(0x37C9E8)
if pivot_index >= 0 then
    pivot_x = mainmemory.readfloat(mainmemory.read_u32_be(0x37DFB4 + 8*pivot_index) - 0x80000000, true)
    pivot_y = mainmemory.readfloat(mainmemory.read_u32_be(0x37DFB4 + 8*pivot_index) - 0x80000000 + 4, true)
    pivot_z = mainmemory.readfloat(mainmemory.read_u32_be(0x37DFB4 + 8*pivot_index) - 0x80000000 + 8, true)
end
print(pivot_x, pivot_y, pivot_z)