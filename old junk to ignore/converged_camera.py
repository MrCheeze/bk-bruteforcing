import math

import numpy as np
from numpy import float32

import re

def toFloat(x):
    return np.frombuffer(np.array(x, dtype=np.uint32).tobytes(), dtype=float32)[0]

def fromFloat(f):
    return hex(np.frombuffer(np.array(f, dtype=float32).tobytes(), dtype=np.uint32)[0])

def fromFloats(fs):
    return [fromFloat(f) for f in fs]

camera_pivot = (toFloat(0xC419EE01), toFloat(0x44D0F2EF))
banjo_pos = (toFloat(0x4554E000), toFloat(0x43AF8000))

def bk_round(f):
    return float32(round((f-0.009) / 0.018) * 0.018 + 0.009)

def getCamRot(camera_pivot, pos):
    return bk_round(math.degrees(math.atan2((camera_pivot[0]-pos[0]), (camera_pivot[1]-pos[1])))) % 360

banjo_cam_rot = getCamRot(camera_pivot, banjo_pos)
#print(banjo_cam_rot)

#print(getCamRot(camera_pivot, (toFloat(0x455D2FB4), toFloat(0x4399AD13))))
#print(getCamRot(camera_pivot, (toFloat(0x4558701D), toFloat(0x43A624F3))))
#print(getCamRot(camera_pivot, (toFloat(0x454848E2), toFloat(0x43D09155))))
#print(getCamRot(camera_pivot, (toFloat(0x453F3656), toFloat(0x43E863D5))))
#print(getCamRot(camera_pivot, (toFloat(0x4549AC29), toFloat(0x43CCEA73))))
#print(getCamRot(camera_pivot, (toFloat(0x4541C64B), toFloat(0x43E1A76B))))


'''
outfile = open('260_door_gap_cameras.txt','w')
outfile_lines = []

for line in open('gap_results_260.txt'):
    match = re.search(r'([0-9][0-9\.]*).* ([0-9][0-9\.]*)', line)
    if match:
        gap_coords = [float(s) for s in match.groups()]
        cam_rot = getCamRot(camera_pivot, gap_coords)
        cam_rot_int = round(cam_rot / 0.009)
        outfile_lines.append('%d\t%g\t%s\n' % (cam_rot_int, cam_rot, gap_coords))

        #if cam_rot == banjo_cam_rot:
        #    print(gap_coords)

for line in sorted(outfile_lines):
    outfile.write(line)

outfile.close()
'''

def move(initial_pos, camera_rot, speed):
    initial_pos = tuple(float32(x) for x in initial_pos)
    movement_x = speed*math.sin(math.radians(camera_rot))
    movement_z = speed*math.cos(math.radians(camera_rot))
    return (float32(initial_pos[0] + movement_x), float32(initial_pos[1] + movement_z))

speed_first_walk_frame = float32(174.00017)/25 # walk on ground
speed_first_jump_frame = 500/25 # jump from land or flap in midair
speed_first_air_adjust_frame = float32(42.00004)/25 # move in midair from rest

print(banjo_pos)
print(fromFloats(move(banjo_pos, banjo_cam_rot, speed_first_walk_frame)))
print(fromFloats(move(banjo_pos, banjo_cam_rot, -speed_first_walk_frame)))
print(fromFloats(move(banjo_pos, banjo_cam_rot, speed_first_jump_frame)))
print(fromFloats(move(banjo_pos, banjo_cam_rot, -speed_first_jump_frame)))
print(fromFloats(move(banjo_pos, banjo_cam_rot, speed_first_air_adjust_frame)))
print(fromFloats(move(banjo_pos, banjo_cam_rot, -speed_first_air_adjust_frame)))

def can_move_to(initial_pos, target_pos, camera_rot):
    return bk_round(math.degrees(math.atan2(initial_pos[0]-target_pos[0], initial_pos[1]-target_pos[1]))%180) == bk_round(camera_rot%180)

print(can_move_to(banjo_pos, (3631.559326171875, 277.49859619140625), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.599609375, 277.5003967285156), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.63671875, 277.5020446777344), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.63720703125, 277.5020751953125), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.65234375, 277.50274658203125), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.676513671875, 277.5038146972656), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.71484375, 277.5055236816406), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.716796875, 277.505615234375), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.740966796875, 277.5066833496094), banjo_cam_rot))
print(can_move_to(banjo_pos, (3631.75439453125, 277.5072937011719), banjo_cam_rot))
print(can_move_to(banjo_pos, (toFloat(0x4561F073), toFloat(0x438D322A)), banjo_cam_rot))
print(can_move_to(banjo_pos, (3188.752, 422.3069), banjo_cam_rot))

def move_input(initial_pos, camera_rot, speed, x_input, y_input):
    initial_pos = tuple(float32(x) for x in initial_pos)
    if x_input < 0:
        x_input = min(x_input+7, 0)
        x_input = max(x_input, -52)
    if x_input > 0:
        x_input = max(x_input-7, 0)
        x_input = min(x_input, 52)
    if y_input < 0:
        y_input = min(y_input+7, 0)
        y_input = max(y_input, -54)
    if y_input > 0:
        y_input = max(y_input-7, 0)
        y_input = min(y_input, 54)

    x_input = round(x_input*80/52)
    y_input = round(y_input*80/54)
    
    movement_x = speed*math.sin(math.radians(bk_round(math.degrees(math.atan2(x_input, -y_input)))))
    movement_z = speed*math.cos(math.radians(bk_round(math.degrees(math.atan2(x_input, -y_input)))))
    return (float32(initial_pos[0] + movement_x), float32(initial_pos[1] + movement_z))

print(move_input((3406,351), 0, speed_first_walk_frame, 0, 127))
print(move_input((3406,351), 0, speed_first_walk_frame, -128, 0))
print(move_input((3406,351), 0, speed_first_walk_frame, 127, 0))
print(move_input((3406,351), 0, speed_first_walk_frame, 0, -128))

x = 10000

while x > 0:
    print(hex(x))
    x //= 2
