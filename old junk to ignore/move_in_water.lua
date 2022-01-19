function round(f)
	return math.floor(f+0.5)
end

function sign(f)
	if f > 0 then
		return 1
	elseif f < 0 then
		return -1
	else
		return 0
	end
end

function value_as_float32(f)
	mainmemory.writefloat(0x7FFFFC, f, true)
	return mainmemory.readfloat(0x7FFFFC, true)
end

speed_first_swim_frame = 25.2/25 -- move on surface of water from rest
speed_first_jump_frame = 80/25 -- jump out of water
speed_first_midair_frame = 42/25 -- move in midair from rest
speed_first_flap_frame = 500/25 -- flap in midair

function xpos() return mainmemory.readfloat(0x37CF70, true) end
function ypos() return mainmemory.readfloat(0x37CF74, true) end
function zpos() return mainmemory.readfloat(0x37CF78, true) end
function camera() return mainmemory.readfloat(0x37E33C, true) end

function get_input_towards_coordinate(diff_x, diff_z, camera_angle)

	stickmag_x = -diff_z*math.sin(math.rad(camera_angle)) + diff_x*math.cos(math.rad(camera_angle))
	stickmag_y = -diff_z*math.cos(math.rad(camera_angle)) - diff_x*math.sin(math.rad(camera_angle))

	scale = math.max(math.abs(stickmag_x), math.abs(stickmag_y))

	input_x = round(stickmag_x * 52 / scale + 7*sign(stickmag_x))
	input_y = round(stickmag_y * 54 / scale + 7*sign(stickmag_y))

	return input_x, input_y
end

function get_input_on_triangular_path_to_coordinate(diff_x, diff_z, first_frame_speed, second_frame_speed, camera_angle)

	a_goal = math.atan2(diff_x, diff_z)
	a_triangle = math.acos((diff_x*diff_x + diff_z*diff_z + first_frame_speed*first_frame_speed - second_frame_speed*second_frame_speed) / (2*math.sqrt(diff_x*diff_x + diff_z*diff_z)*first_frame_speed))

	a1 = a_goal + a_triangle
	a2 = a_goal - a_triangle
	
	intermediate_x1 = math.sin(a1)
	intermediate_z1 = math.cos(a1)

	intermediate_x2 = math.sin(a2)
	intermediate_z2 = math.cos(a2)

	input_x1, input_y1 = get_input_towards_coordinate(intermediate_x1, intermediate_z1, camera_angle)
	input_x2, input_y2 = get_input_towards_coordinate(intermediate_x2, intermediate_z2, camera_angle)
	
	return input_x1, input_y1, input_x2, input_y2
end

function jump_to_coord(goal_x, goal_z, use_second_triangle, surface_ypos)

	-- Move for one frame
	for i=1,2 do
		input_x1, input_y1, input_x2, input_y2 = get_input_on_triangular_path_to_coordinate(goal_x-xpos(), goal_z-zpos(), speed_first_midair_frame, speed_first_midair_frame, camera())
		if use_second_triangle then
			if input_x2 ~= input_x2 or input_y2 ~= input_y2 then
				return false
			end
			joypad.setanalog({["P1 X Axis"]=input_x2, ["P1 Y Axis"]=input_y2})
		else
			if input_x1 ~= input_x1 or input_y1 ~= input_y1 then
				return false
			end
			joypad.setanalog({["P1 X Axis"]=input_x1, ["P1 Y Axis"]=input_y1})
		end
		emu.frameadvance()
	end
	joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})
	
	-- Feather flap to halt speed and land sooner
	for i=1,2 do
		joypad.set({["A"]=1}, 1)
		emu.frameadvance()
	end
	
	-- Wait until resurfaced
	while ypos() ~= surface_ypos do
		emu.frameadvance()
	end
	
	---- Wait for camera to settle
	--prev_camera = mainmemory.readfloat(0x37E33C, true)
	--emu.frameadvance()
	--emu.frameadvance()
	--while mainmemory.readfloat(0x37E33C, true) ~= prev_camera do
	--	prev_camera = mainmemory.readfloat(0x37E33C, true)
	--	emu.frameadvance()
	--	emu.frameadvance()
	--end
	
	-- Jump out of water
	for i=1,2 do
		joypad.set({["A"]=1}, 1)
		emu.frameadvance()
	end
	
	-- Move for one frame
	for i=1,2 do
		input_x, input_y = get_input_towards_coordinate(goal_x-xpos(), goal_z-zpos(), camera())
		joypad.setanalog({["P1 X Axis"]=input_x, ["P1 Y Axis"]=input_y})
		emu.frameadvance()
	end
	joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})
	
	---- Make sure the input registered
	--joypad.set({["A"]=1}, 1)
	--emu.frameadvance()
	
	reached_goal = (xpos() == value_as_float32(goal_x) and zpos() == value_as_float32(goal_z))
	return reached_goal
end

--max_magnitude_inputs = {{-59,-61},{-59,-60},{-59,-59},{-59,-58},{-59,-57},{-59,-56},{-59,-55},{-59,-54},{-59,-53},{-59,-52},{-59,-51},{-59,-50},{-59,-49},{-59,-48},{-59,-47},{-59,-46},{-59,-45},{-59,-44},{-59,-43},{-59,-42},{-59,-41},{-59,-40},{-59,-39},{-59,-38},{-59,-37},{-59,-36},{-59,-35},{-59,-34},{-59,-33},{-59,-32},{-59,-31},{-59,-30},{-59,-29},{-59,-28},{-59,-27},{-59,-26},{-59,-25},{-59,-24},{-59,-23},{-59,-22},{-59,-21},{-59,-20},{-59,-19},{-59,-18},{-59,-17},{-59,-16},{-59,-15},{-59,-14},{-59,-13},{-59,-12},{-59,-11},{-59,-10},{-59,-9},{-59,-8},{-59,0},{-59,8},{-59,9},{-59,10},{-59,11},{-59,12},{-59,13},{-59,14},{-59,15},{-59,16},{-59,17},{-59,18},{-59,19},{-59,20},{-59,21},{-59,22},{-59,23},{-59,24},{-59,25},{-59,26},{-59,27},{-59,28},{-59,29},{-59,30},{-59,31},{-59,32},{-59,33},{-59,34},{-59,35},{-59,36},{-59,37},{-59,38},{-59,39},{-59,40},{-59,41},{-59,42},{-59,43},{-59,44},{-59,45},{-59,46},{-59,47},{-59,48},{-59,49},{-59,50},{-59,51},{-59,52},{-59,53},{-59,54},{-59,55},{-59,56},{-59,57},{-59,58},{-59,59},{-59,60},{-59,61},{-58,-61},{-58,61},{-57,-61},{-57,61},{-56,-61},{-56,61},{-55,-61},{-55,61},{-54,-61},{-54,61},{-53,-61},{-53,61},{-52,-61},{-52,61},{-51,-61},{-51,61},{-50,-61},{-50,61},{-49,-61},{-49,61},{-48,-61},{-48,61},{-47,-61},{-47,61},{-46,-61},{-46,61},{-45,-61},{-45,61},{-44,-61},{-44,61},{-43,-61},{-43,61},{-42,-61},{-42,61},{-41,-61},{-41,61},{-40,-61},{-40,61},{-39,-61},{-39,61},{-38,-61},{-38,61},{-37,-61},{-37,61},{-36,-61},{-36,61},{-35,-61},{-35,61},{-34,-61},{-34,61},{-33,-61},{-33,61},{-32,-61},{-32,61},{-31,-61},{-31,61},{-30,-61},{-30,61},{-29,-61},{-29,61},{-28,-61},{-28,61},{-27,-61},{-27,61},{-26,-61},{-26,61},{-25,-61},{-25,61},{-24,-61},{-24,61},{-23,-61},{-23,61},{-22,-61},{-22,61},{-21,-61},{-21,61},{-20,-61},{-20,61},{-19,-61},{-19,61},{-18,-61},{-18,61},{-17,-61},{-17,61},{-16,-61},{-16,61},{-15,-61},{-15,61},{-14,-61},{-14,61},{-13,-61},{-13,61},{-12,-61},{-12,61},{-11,-61},{-11,61},{-10,-61},{-10,61},{-9,-61},{-9,61},{-8,-61},{-8,61},{0,-61},{0,61},{8,-61},{8,61},{9,-61},{9,61},{10,-61},{10,61},{11,-61},{11,61},{12,-61},{12,61},{13,-61},{13,61},{14,-61},{14,61},{15,-61},{15,61},{16,-61},{16,61},{17,-61},{17,61},{18,-61},{18,61},{19,-61},{19,61},{20,-61},{20,61},{21,-61},{21,61},{22,-61},{22,61},{23,-61},{23,61},{24,-61},{24,61},{25,-61},{25,61},{26,-61},{26,61},{27,-61},{27,61},{28,-61},{28,61},{29,-61},{29,61},{30,-61},{30,61},{31,-61},{31,61},{32,-61},{32,61},{33,-61},{33,61},{34,-61},{34,61},{35,-61},{35,61},{36,-61},{36,61},{37,-61},{37,61},{38,-61},{38,61},{39,-61},{39,61},{40,-61},{40,61},{41,-61},{41,61},{42,-61},{42,61},{43,-61},{43,61},{44,-61},{44,61},{45,-61},{45,61},{46,-61},{46,61},{47,-61},{47,61},{48,-61},{48,61},{49,-61},{49,61},{50,-61},{50,61},{51,-61},{51,61},{52,-61},{52,61},{53,-61},{53,61},{54,-61},{54,61},{55,-61},{55,61},{56,-61},{56,61},{57,-61},{57,61},{58,-61},{58,61},{59,-61},{59,-60},{59,-59},{59,-58},{59,-57},{59,-56},{59,-55},{59,-54},{59,-53},{59,-52},{59,-51},{59,-50},{59,-49},{59,-48},{59,-47},{59,-46},{59,-45},{59,-44},{59,-43},{59,-42},{59,-41},{59,-40},{59,-39},{59,-38},{59,-37},{59,-36},{59,-35},{59,-34},{59,-33},{59,-32},{59,-31},{59,-30},{59,-29},{59,-28},{59,-27},{59,-26},{59,-25},{59,-24},{59,-23},{59,-22},{59,-21},{59,-20},{59,-19},{59,-18},{59,-17},{59,-16},{59,-15},{59,-14},{59,-13},{59,-12},{59,-11},{59,-10},{59,-9},{59,-8},{59,0},{59,8},{59,9},{59,10},{59,11},{59,12},{59,13},{59,14},{59,15},{59,16},{59,17},{59,18},{59,19},{59,20},{59,21},{59,22},{59,23},{59,24},{59,25},{59,26},{59,27},{59,28},{59,29},{59,30},{59,31},{59,32},{59,33},{59,34},{59,35},{59,36},{59,37},{59,38},{59,39},{59,40},{59,41},{59,42},{59,43},{59,44},{59,45},{59,46},{59,47},{59,48},{59,49},{59,50},{59,51},{59,52},{59,53},{59,54},{59,55},{59,56},{59,57},{59,58},{59,59},{59,60},{59,61}}

function search_for_jump_to_coord(savestate_slot, goal_x, goal_z)
	for x=-10,10 do
		if x > 7 or x < -7 or x == 0 then
			for y=-10,10 do
				if y > 7 or y < -7 or y == 0 then
					for triangle_to_use=1,2 do
						
						savestate.loadslot(savestate_slot)
						
						for j=1,4 do
							joypad.setanalog({["P1 X Axis"]=x, ["P1 Y Axis"]=y})
							emu.frameadvance()
						end
						joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})
						
						surface_ypos = ypos()

						-- Jump out of water
						for i=1,2 do
							joypad.set({["A"]=1}, 1)
							emu.frameadvance()
						end

						reached_goal = jump_to_coord(goal_x, goal_z, triangle_to_use==2, surface_ypos)
						print(x, y, triangle_to_use, xpos(), zpos())
						if reached_goal then
							client.pause()
							return true
						end
						
					end
				end
			end
		end
	end
	return false
end


best_distance = 99999

function search_for_jump_to_any_coord(savestate_slot, surface_ypos)

	savestate.loadslot(5)

	for i=1,#gap_coords do
		for triangle_to_use=1,2 do
			framecount = emu.framecount()
			reached_goal = jump_to_coord(gap_coords[i][1], gap_coords[i][2], triangle_to_use==2, surface_ypos)
			if framecount ~= emu.framecount() then
				dist = math.abs(xpos()-gap_coords[i][1]) + math.abs(zpos()-gap_coords[i][2])
				if dist <= best_distance then
					print(gap_coords[i][1], gap_coords[i][2], triangle_to_use, xpos(), zpos(), dist, reached_goal)
					best_distance = dist
				end
				if reached_goal then
					client.pause()
					return true
				end
				savestate.loadslot(5)
			end
		end
	end
	return false
end

search_for_jump_to_any_coord(5, 1380)
print("done")