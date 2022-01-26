-- Most info "borrowed" from ScriptHawk

local Memory = { -- Version order: Europe, Japan, US 1.1, US 1.0
	x_position = {0x37CF70, 0x37D0A0, 0x37B7A0, 0x37C5A0}, -- Float
	z_position = {0x37CF78, 0x37D0A8, 0x37B7A8, 0x37C5A8}, -- Float
	x_velocity = {0x37CE88, 0x37CFB8, 0x37B6B8, 0x37C4B8}, -- Float, divide by framerate (to match dXZ,dY scale)
	z_velocity = {0x37CE90, 0x37CFC0, 0x37B6C0, 0x37C4C0}, -- Float, divide by framerate (to match dXZ,dY scale)
	moving_angle = {0x37D064, 0x37D194, 0x37B894, 0x37C694}, -- Float
	current_movement_state = {0x37DB34, 0x37DC64, 0x37C364, 0x37D164},
	
	camera_x_position = {0x37E328, 0x37E458, 0x37CB58, 0x37D958}, -- Float
	camera_z_position = {0x37E330, 0x37E460, 0x37CB60, 0x37D960}, -- Float
	camera_x_velocity = {0x37E3B0, 0x37E4E0, 0x37CBE0, 0x37D9E0}, -- Float
	camera_z_velocity = {0x37E3B8, 0x37E4E8, 0x37CBE8, 0x37D9E8}, -- Float
	camera_y_rotation = {0x37E33C, 0x37E46C, 0x37CB6C, 0x37D96C}, -- Float
	camera_y_angular_velocity = {0x37E39C, 0x37E4CC, 0x37CBCC, 0x37D9CC}, -- Float
	
	camera_region_index = {0x37C9E8, nil, nil, nil},
	camera_region_table = {0x37DFB0, nil, nil, nil},
}

local movementStates = {
	[0] = "Null",
	[1] = "Idle",
	[2] = "Walking", -- Slow
	[3] = "Walking",
	[4] = "Walking", -- Fast
	[5] = "Jumping",
	[6] = "Bear punch",
	[7] = "Crouching",
	[8] = "Jumping", -- Talon Trot
	[9] = "Shooting Egg",
	[10] = "Pooping Egg",

	[12] = "Skidding",

	[14] = "Damaged",
	[15] = "Beak Buster",
	[16] = "Feathery Flap",
	[17] = "Rat-a-tat rap",
	[18] = "Backflip", -- Flap Flip
	[19] = "Beak Barge",
	[20] = "Entering Talon Trot",
	[21] = "Idle", -- Talon Trot
	[22] = "Walking", -- Talon Trot
	[23] = "Leaving Talon Trot",
	[24] = "Knockback", -- Flying

	[26] = "Entering Wonderwing",
	[27] = "Idle", -- Wonderwing
	[28] = "Walking", -- Wonderwing
	[29] = "Jumping", -- Wonderwing
	[30] = "Leaving Wonderwing",
	[31] = "Creeping",
	[32] = "Landing", -- After Jump
	[33] = "Charging Shock Spring Jump",
	[34] = "Shock Spring Jump",
	[35] = "Taking Flight",
	[36] = "Flying",
	[37] = "Entering Wading Boots",
	[38] = "Idle", -- Wading Boots
	[39] = "Walking", -- Wading Boots
	[40] = "Jumping", -- Wading Boots
	[41] = "Leaving Wading Boots",
	[42] = "Beak Bomb",
	[43] = "Idle", -- Underwater
	[44] = "Swimming (B)",
	[45] = "Idle", -- Treading water
	[46] = "Paddling",
	[47] = "Falling", -- After pecking
	[48] = "Diving",
	[49] = "Rolling",
	[50] = "Slipping",

	[52] = "Jig", -- Note door
	[53] = "Idle", -- Termite
	[54] = "Walking", -- Termite
	[55] = "Jumping", -- Termite
	[56] = "Falling", -- Termite
	[57] = "Swimming (A)",
	[58] = "Idle", -- Carrying object (eg. Orange)
	[59] = "Walking", -- Carrying object (eg. Orange)

	[61] = "Falling", -- Tumbling, will take damage
	[62] = "Damaged", -- Termite

	[64] = "Locked", -- Pumpkin: Pipe
	[65] = "Death",
	[66] = "Dingpot",
	[67] = "Death", -- Termite
	[68] = "Jig", -- Jiggy
	[69] = "Slipping", -- Talon Trot

	[72] = "Idle", -- Pumpkin
	[73] = "Walking", -- Pumpkin
	[74] = "Jumping", -- Pumpkin
	[75] = "Falling", -- Pumpkin
	[76] = "Landing", -- In water
	[77] = "Damaged", -- Pumpkin
	[78] = "Death", -- Pumpkin
	[79] = "Idle", -- Holding tree, pole, etc.
	[80] = "Climbing", -- Tree, pole, etc.
	[81] = "Leaving Climb",
	[82] = "Tumblar", -- Standing on Tumblar
	[83] = "Tumblar", -- Standing on Tumblar
	[84] = "Death", -- Drowning
	[85] = "Slipping", -- Wading Boots
	[86] = "Knockback", -- Successful enemy damage
	[87] = "Beak Bomb", -- Ending
	[88] = "Damaged", -- Beak Bomb
	[89] = "Damaged", -- Beak Bomb
	[90] = "Loading Zone",
	[91] = "Throwing", -- Throwing object (eg. Orange)

	[94] = "Idle", -- Croc
	[95] = "Walking", -- Croc
	[96] = "Jumping", -- Croc
	[97] = "Falling", -- Croc
	[99] = "Damaged", -- Croc
	[100] = "Death", -- Croc

	[103] = "Idle", -- Walrus
	[104] = "Walking", -- Walrus
	[105] = "Jumping", -- Walrus
	[106] = "Falling", -- Walrus
	[107] = "Locked", -- Bee, Mumbo Transform Cutscene
	[108] = "Knockback", -- Walrus
	[109] = "Death", -- Walrus
	[110] = "Biting", -- Croc
	[111] = "EatingWrongThing", -- Croc
	[112] = "EatingCorrectThing", -- Croc
	[113] = "Falling", -- Talon Trot
	[114] = "Recovering", -- Getting up after taking damage, eg. fall famage
	[115] = "Locked", -- Cutscene
	[116] = "Locked", -- Jiggy pad, Mumbo transformation, Bottles
	[117] = "Locked", -- Bottles
	[118] = "Locked", -- Flying
	[119] = "Locked", -- Water Surface
	[120] = "Locked", -- Underwater
	[121] = "Locked", -- Holding Jiggy, Talon Trot
	[122] = "Creeping", -- In damaging water etc
	[123] = "Damaged", -- Talon Trot
	[124] = "Locked", -- Sled in FP sliding down scarf
	[125] = "Idle", -- Walrus Sled
	[126] = "Jumping", -- Walrus Sled
	[127] = "Damaged", -- Swimming
	[128] = "Locked", -- Walrus Sled losing race
	[129] = "Locked", -- Walrus Sled
	[130] = "Locked", -- Walrus Sled In Air when losing race

	[133] = "Idle", -- Bee
	[134] = "Walking", -- Bee
	[135] = "Jumping", -- Bee
	[136] = "Falling", -- Bee
	[137] = "Damaged", -- Bee
	[138] = "Death", -- Bee

	[140] = "Flying", -- Bee
	[141] = "Locked", -- Mumbo transformation, Mr. Vile
	[142] = "Locked", -- Jiggy podium, Bottles' text outside Mumbo's
	[143] = "Locked", -- Pumpkin
	[145] = "Damaged", -- Flying
	[146] = "Locked", -- Termite
	[147] = "Locked", -- Pumpkin?
	[148] = "Locked", -- Mumbo transformation
	[149] = "Locked", -- Walrus?
	[150] = "Locked", -- Paddling
	[151] = "Locked", -- Swimming
	[152] = "Locked", -- Loading zone, Mumbo transformation
	[153] = "Locked", -- Flying
	[154] = "Locked", -- Talon Trot
	[155] = "Locked", -- Wading Boots
	--[156] = "Locked??", -- In WalrusSled Set
	[157] = "Locked", -- Bee?
	[158] = "Locked", -- Climbing
	[159] = "Knockback", -- Termite, not damaged
	[160] = "Knockback", -- Pumpkin, not damaged
	[161] = "Knockback", -- Croc, not damaged
	[162] = "Knockback", -- Walrus, not damaged
	[163] = "Knockback", -- Bee, not damaged
	--[164] = "???", -- Wonderwing
	[165] = "Locked", -- Wonderwing
};

local supportedGames = {
	["90726D7E7CD5BF6CDFD38F45C9ACBF4D45BD9FD8"] = {moduleName="games.bk", friendlyName="Banjo to Kazooie no Daibouken (Japan)", version=2},
	["BB359A75941DF74BF7290212C89FBC6E2C5601FE"] = {moduleName="games.bk", friendlyName="Banjo-Kazooie (Europe) (En,Fr,De)", version=1},
	["DED6EE166E740AD1BC810FD678A84B48E245AB80"] = {moduleName="games.bk", friendlyName="Banjo-Kazooie (USA) (Rev A)", version=3},
	["1FE1632098865F639E22C11B9A81EE8F29C75D7A"] = {moduleName="games.bk", friendlyName="Banjo-Kazooie (USA)", version=4},
	["6A81FE9C1C9059275A2C5E64D608BAA91F22C14C"] = {moduleName="games.bk", friendlyName="Banjo-Dreamie", version=4},
	["7BCA8A32E83823F9230A92DA14F26E74744B0CEC"] = {moduleName="games.bk", friendlyName="BK Hidden Lair", version=4},
	["4B5CEA82AE7BD0951E02BFBE6D1665626A03BA20"] = {moduleName="games.bk", friendlyName="Banjo-Kazooie Worlds Collide", version=4},
	["3D1C4371EFA16E325E981C35CF56A91D37B4BC25"] = {moduleName="games.bk", friendlyName="BK Legend of the Crystal Jiggy", version=4},
	["EB3863AE260CC6B248AA3F69DC4F5D712799E09E"] = {moduleName="games.bk", friendlyName="BK Nightbear Before Christmas", version=4},
	["19D2B030BD1E7D91D1A25614C330638007660235"] = {moduleName="games.bk", friendlyName="Banjo-Kazooie Fort Fun v5", version=4},
	["7C50845E42C9B7B2BE48BDDC89D8D70BE95C5324"] = {moduleName="games.bk", friendlyName="Banjo Kazooie: How The Gruntch stole Christmas (1.2)", version=4},
}

local v = supportedGames[gameinfo.getromhash()].version

local x_pos = mainmemory.readfloat(Memory.x_position[v], true)
local z_pos = mainmemory.readfloat(Memory.z_position[v], true)
local x_speed = mainmemory.readfloat(Memory.x_velocity[v], true)
local z_speed = mainmemory.readfloat(Memory.z_velocity[v], true)
local angle = mainmemory.readfloat(Memory.moving_angle[v], true)
local movement_state = movementStates[mainmemory.read_s32_be(Memory.current_movement_state[v])]
local is_moving
if movement_state == 'Idle' then
	is_moving = 0
else
	is_moving = 1
end

local cam_x_pos = mainmemory.readfloat(Memory.camera_x_position[v], true)
local cam_z_pos = mainmemory.readfloat(Memory.camera_z_position[v], true)
local cam_x_speed = mainmemory.readfloat(Memory.camera_x_velocity[v], true)
local cam_z_speed = mainmemory.readfloat(Memory.camera_z_velocity[v], true)
local cam_angle = mainmemory.readfloat(Memory.camera_y_rotation[v], true)
local cam_angle_speed = mainmemory.readfloat(Memory.camera_y_angular_velocity[v], true)

local cam_region_index = mainmemory.read_s32_be(Memory.camera_region_index[v])
if cam_region_index < 0 then
	print("Free camera not yet supported!!")
	return
end

local cam_region_type = mainmemory.read_u32_be(Memory.camera_region_table[v] + 8*cam_region_index)
local cam_region_ptr = mainmemory.read_u32_be(Memory.camera_region_table[v] + 8*cam_region_index + 4)

if cam_region_type ~= 0x00000301 then
	print("Only camera region type 00000301 is currently supported!")
	return
end

local cam_region_info = {}
for i=0,8 do
	cam_region_info[i+1] = mainmemory.readfloat(cam_region_ptr - 0x80000000 + 4*i, true)
end

local cmd = string.format('python bk_searcher.py %s %s %s %s %s %s  %s %s %s %s %s %s  %s %s %s %s %s %s %s %s %s 2>stderr.log',
	x_pos, z_pos, x_speed, z_speed, angle, is_moving,
	cam_x_pos, cam_z_pos, cam_x_speed, cam_z_speed, cam_angle, cam_angle_speed,
	cam_region_info[1], cam_region_info[2], cam_region_info[3], cam_region_info[4], cam_region_info[5], cam_region_info[6], cam_region_info[7], cam_region_info[8], cam_region_info[9])

print(cmd)
os.remove('stdout.log')
local ret = os.execute(cmd)
if ret ~= 0 then
	print('--- Failed to execute python script: ---')
	
	local f = io.open('stdout.log', 'r')
	if f ~= nil then
		print(f:read("*all"))
		f:close()
	end
	
	f = io.open('stderr.log', 'r')
	print(f:read("*all"))
	f:close()
	
	return
end

print('--- Python script output: ---')

local f = io.open('stdout.log', 'r')
local stdout = f:read("*all")
print(stdout)
f:close()

local joypad_script = stdout:gmatch("joypad\.[^\r\n]+")()
if joypad_script ~= nil then
	client.unpause()
	loadstring(joypad_script)()
	client.pause()
end

print("done.")