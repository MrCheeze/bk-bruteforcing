import numpy as np
from numpy import float32, float64
import math
import copy
import re
from sys import argv


from bk_math_lib import *
from bk_movement_sim import *

distinct_x_inputs = [-59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
distinct_y_inputs = [-61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

useless_inputs = [(-59, -61), (-59, -58), (-59, -56), (-59, -48), (-59, -42), (-59, -41), (-59, -40), (-59, -34), (-59, 34), (-59, 40), (-59, 41), (-59, 42), (-59, 48), (-59, 56), (-59, 58), (-59, 61), (-58, -60), (-58, -51), (-58, 51), (-58, 60), (-57, -60), (-57, -58), (-57, -46), (-57, 46), (-57, 58), (-57, 60), (-56, -58), (-56, -48), (-56, -41), (-56, -38), (-56, -33), (-56, 33), (-56, 38), (-56, 41), (-56, 48), (-56, 58), (-55, -56), (-55, -51), (-55, 51), (-55, 56), (-54, -60), (-54, -59), (-54, -56), (-54, -50), (-54, -44), (-54, 44), (-54, 50), (-54, 56), (-54, 59), (-54, 60), (-53, -61), (-53, -59), (-53, -58), (-53, -54), (-53, -50), (-53, -48), (-53, -39), (-53, 39), (-53, 48), (-53, 50), (-53, 54), (-53, 58), (-53, 59), (-53, 61), (-52, -57), (-52, -54), (-52, 54), (-52, 57), (-50, -60), (-50, -56), (-50, -52), (-50, 52), (-50, 56), (-50, 60), (-49, -61), (-49, -56), (-49, -50), (-49, 50), (-49, 56), (-49, 61), (-48, -50), (-48, 50), (-46, -60), (-46, -59), (-46, -58), (-46, -56), (-46, 56), (-46, 58), (-46, 59), (-46, 60), (-41, -60), (-41, 60), (-40, -58), (-40, 58), (-37, -59), (-37, 59), (-33, -61), (-33, 61), (-28, -61), (-28, 61), (-13, -9), (-13, -8), (-13, 0), (-13, 8), (-13, 9), (-12, -11), (-12, -10), (-12, -9), (-12, -8), (-12, 0), (-12, 8), (-12, 9), (-12, 10), (-12, 11), (-11, -12), (-11, -11), (-11, -10), (-11, -9), (-11, -8), (-11, 0), (-11, 8), (-11, 9), (-11, 10), (-11, 11), (-11, 12), (-10, -13), (-10, -12), (-10, -11), (-10, -10), (-10, -9), (-10, -8), (-10, 0), (-10, 8), (-10, 9), (-10, 10), (-10, 11), (-10, 12), (-10, 13), (-9, -13), (-9, -12), (-9, -11), (-9, -10), (-9, -9), (-9, -8), (-9, 0), (-9, 8), (-9, 9), (-9, 10), (-9, 11), (-9, 12), (-9, 13), (-8, -13), (-8, -12), (-8, -11), (-8, -10), (-8, -9), (-8, -8), (-8, 0), (-8, 8), (-8, 9), (-8, 10), (-8, 11), (-8, 12), (-8, 13), (0, -13), (0, -12), (0, -11), (0, -10), (0, -9), (0, -8), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (8, -13), (8, -12), (8, -11), (8, -10), (8, -9), (8, -8), (8, 0), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, -13), (9, -12), (9, -11), (9, -10), (9, -9), (9, -8), (9, 0), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (10, -13), (10, -12), (10, -11), (10, -10), (10, -9), (10, -8), (10, 0), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (11, -12), (11, -11), (11, -10), (11, -9), (11, -8), (11, 0), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (12, -11), (12, -10), (12, -9), (12, -8), (12, 0), (12, 8), (12, 9), (12, 10), (12, 11), (13, -9), (13, -8), (13, 0), (13, 8), (13, 9), (28, -61), (28, 61), (33, -61), (33, 61), (37, -59), (37, 59), (40, -58), (40, 58), (41, -60), (41, 60), (46, -60), (46, -59), (46, -58), (46, -56), (46, 56), (46, 58), (46, 59), (46, 60), (48, -50), (48, 50), (49, -61), (49, -56), (49, -50), (49, 50), (49, 56), (49, 61), (50, -60), (50, -56), (50, -52), (50, 52), (50, 56), (50, 60), (52, -57), (52, -54), (52, 54), (52, 57), (53, -61), (53, -59), (53, -58), (53, -54), (53, -50), (53, -48), (53, -39), (53, 39), (53, 48), (53, 50), (53, 54), (53, 58), (53, 59), (53, 61), (54, -60), (54, -59), (54, -56), (54, -50), (54, -44), (54, 44), (54, 50), (54, 56), (54, 59), (54, 60), (55, -56), (55, -51), (55, 51), (55, 56), (56, -58), (56, -48), (56, -41), (56, -38), (56, -33), (56, 33), (56, 38), (56, 41), (56, 48), (56, 58), (57, -60), (57, -58), (57, -46), (57, 46), (57, 58), (57, 60), (58, -60), (58, -51), (58, 51), (58, 60), (59, -61), (59, -58), (59, -56), (59, -48), (59, -42), (59, -41), (59, -40), (59, -34), (59, 34), (59, 40), (59, 41), (59, 42), (59, 48), (59, 56), (59, 58), (59, 61)]

# (x, y) inputs that will have distinct results when input
distinct_inputs = []
for x in distinct_x_inputs:
    for y in distinct_y_inputs:
        if (x, y) not in useless_inputs:
            distinct_inputs.append((x, y))

# (x, 0) and (0, y) inputs that will have distinct results when input
distinct_x_inputs = [x for x in distinct_x_inputs if abs(x) > 13 or x == 0]
distinct_y_inputs = [y for y in distinct_x_inputs if abs(y) > 13 or y == 0]

# For when angle is irrelevant but need to move by a low magnitude
distinct_low_magnitudes = [(13, 10), (12, 12), (11, 13), (8, 14), (14, 9), (13, 11), (9, 14), (12, 13), (10, 14), (8, 15), (14, 11), (9, 15), (11, 14), (10, 15), (13, 13), (15, 9), (12, 14), (11, 15), (15, 10), (14, 13), (15, 11), (8, 16), (16, 9), (9, 16), (13, 14), (10, 16), (15, 12), (16, 11), (8, 17), (14, 14), (13, 15), (9, 17), (15, 13), (10, 17), (12, 16), (14, 15), (17, 8), (17, 9), (11, 17), (16, 13), (17, 10), (15, 14), (12, 17), (13, 16)]

# Sequences of y inputs in midair that will displace you but leave you with speed 0
zero_speed_y_inputs = [
    (-59,-35,26,55),
    (-59,-26,51,40),
    (-59,39,51,-17),
    (-58,20,23,37),
    (-58,54,-26,33),
    (-57,52,-20,33),
    (-57,54,-44,46),
    (-56,17,-36,56),
    (-55,19,28,24),
    (-55,49,-42,49),
    (-54,-18,47,30),
    (-53,53,-45,38),
    (-52,-35,14,53),
    (-52,-16,20,44),
    (-52,46,-27,37),
    (-52,50,28,-23),
    (-51,19,18,19),
    (-51,33,-39,50),
    (-51,42,-19,32),
    (-50,39,41,-26),
    (-49,52,-39,22),
    (-48,-28,27,38),
    (-48,16,-18,43),
    (-48,35,-24,32),
    (-48,44,36,-32),
    (-48,50,18,-28),
    (-47,31,-24,35),
    (-46,-14,37,15),
    (-46,34,-51,52),
    (-45,40,-20,21),
    (-44,14,14,22),
    (-44,38,-30,31),
    (-42,-40,36,35),
    (-42,44,-35,27),
    (-42,56,-24,-19),
    (-41,-19,57,-38),
    (-40,-19,-35,54),
    (-39,23,-20,30),
    (-37,-38,44,19),
    (-37,24,47,-37),
    (-37,51,14,-34),
    (-37,59,-42,-19),
    (-36,55,-55,34),
    (-36,56,-57,39),
    (-36,57,-40,-17),
    (-35,-43,57,-22),
    (-35,18,23,-15),
    (-35,22,-37,41),
    (-35,31,46,-42),
    (-34,-44,39,28),
    (-34,32,-23,20),
    (-33,27,-32,32),
    (-33,43,31,-40),
    (-31,-48,56,-17),
    (-30,-21,15,32),
    (-29,22,-54,53),
    (-29,33,-45,38),
    (-28,-28,29,19),
    (-28,27,26,-25),
    (-28,30,42,-42),
    (-27,-43,37,24),
    (-26,-14,57,-50),
    (-26,38,-54,50),
    (-25,-26,-28,51),
    (-25,15,43,-30),
    (-25,21,-27,26),
    (-25,25,-39,35),
    (-24,-51,32,43),
    (-24,-14,54,-45),
    (-24,27,-39,33),
    (-24,54,31,-53),
    (-23,-29,-31,52),
    (-23,-21,14,28),
    (-23,14,32,-18),
    (-22,-50,35,36),
    (-22,-18,42,-14),
    (-21,47,-41,16),
    (-21,50,43,-53),
    (-20,-52,43,33),
    (-20,-42,57,-35),
    (-20,-26,19,20),
    (-20,14,35,-23),
    (-20,28,-25,17),
    (-19,-41,32,21),
    (-18,-42,54,-22),
    (-18,15,-18,23),
    (-18,40,-46,24),
    (-17,-57,-17,59),
    (-17,-48,57,-33),
    (-17,-20,40,-15),
    (-17,23,-39,29),
    (-17,36,24,-40),
    (-16,19,-48,40),
    (-16,36,-57,51),
    (-16,59,-39,-40),
    (-15,-49,38,20),
    (15,49,-38,-20),
    (16,-59,39,40),
    (16,-36,57,-51),
    (16,-19,48,-40),
    (17,-36,-24,40),
    (17,-23,39,-29),
    (17,20,-40,15),
    (17,48,-57,33),
    (17,57,17,-59),
    (18,-40,46,-24),
    (18,-15,18,-23),
    (18,42,-54,22),
    (19,41,-32,-21),
    (20,-28,25,-17),
    (20,-14,-35,23),
    (20,26,-19,-20),
    (20,42,-57,35),
    (20,52,-43,-33),
    (21,-50,-43,53),
    (21,-47,41,-16),
    (22,18,-42,14),
    (22,50,-35,-36),
    (23,-14,-32,18),
    (23,21,-14,-28),
    (23,29,31,-52),
    (24,-54,-31,53),
    (24,-27,39,-33),
    (24,14,-54,45),
    (24,51,-32,-43),
    (25,-25,39,-35),
    (25,-21,27,-26),
    (25,-15,-43,30),
    (25,26,28,-51),
    (26,-38,54,-50),
    (26,14,-57,50),
    (27,43,-37,-24),
    (28,-30,-42,42),
    (28,-27,-26,25),
    (28,28,-29,-19),
    (29,-33,45,-38),
    (29,-22,54,-53),
    (30,21,-15,-32),
    (31,48,-56,17),
    (33,-43,-31,40),
    (33,-27,32,-32),
    (34,-32,23,-20),
    (34,44,-39,-28),
    (35,-31,-46,42),
    (35,-22,37,-41),
    (35,-18,-23,15),
    (35,43,-57,22),
    (36,-57,40,17),
    (36,-56,57,-39),
    (36,-55,55,-34),
    (37,-59,42,19),
    (37,-51,-14,34),
    (37,-24,-47,37),
    (37,38,-44,-19),
    (39,-23,20,-30),
    (40,19,35,-54),
    (41,19,-57,38),
    (42,-56,24,19),
    (42,-44,35,-27),
    (42,40,-36,-35),
    (44,-38,30,-31),
    (44,-14,-14,-22),
    (45,-40,20,-21),
    (46,-34,51,-52),
    (46,14,-37,-15),
    (47,-31,24,-35),
    (48,-50,-18,28),
    (48,-44,-36,32),
    (48,-35,24,-32),
    (48,-16,18,-43),
    (48,28,-27,-38),
    (49,-52,39,-22),
    (50,-39,-41,26),
    (51,-42,19,-32),
    (51,-33,39,-50),
    (51,-19,-18,-19),
    (52,-50,-28,23),
    (52,-46,27,-37),
    (52,16,-20,-44),
    (52,35,-14,-53),
    (53,-53,45,-38),
    (54,18,-47,-30),
    (55,-49,42,-49),
    (55,-19,-28,-24),
    (56,-17,36,-56),
    (57,-54,44,-46),
    (57,-52,20,-33),
    (58,-54,26,-33),
    (58,-20,-23,-37),
    (59,-39,-51,17),
    (59,26,-51,-40),
    (59,35,-26,-55),
]

def printwrite(f, *args):
    s = ' '.join([str(arg) for arg in args])
    print(s)
    if f:
        f.write(s+'\n')

def do_search(game, f=None):

    inputstring = []

    game_copy = copy.copy(game)
    game_copy._update_camera()
    if game_copy.camera.angle != game.camera.angle:
        printwrite(f, 'Camera has not yet converged! The sim will not work.')
        return
    
    gaps = {}
    for line in open('old junk to ignore/gap_results_260.txt'):
        match = re.search(r'([0-9][0-9\.]*).* ([0-9][0-9\.]*)', line)
        if match:
            gap_coords = tuple(float32(s) for s in match.groups())
            cam_angle = ml_vec3f_yaw_between(gap_coords, game._get_target_camera_pos(gap_coords))
            if cam_angle not in gaps:
                gaps[cam_angle] = []
            gaps[cam_angle].append(gap_coords)
    game_orig = copy.copy(game)

    y_inputs = [y for y in distinct_y_inputs if y==0 or abs(y) > 13]

    zero_speed_input_displacements = {}

    for ys in zero_speed_y_inputs:
        game = copy.copy(game_orig)
        game._update_banjo('jump', 0, 0)
        game._update_banjo('midair', 0, ys[0])
        game._update_banjo('midair', 0, ys[1])
        game._update_banjo('midair', 0, ys[2])
        game._update_banjo('midair', 0, ys[3])
        pos = copy.copy(game.banjo.pos)
        game.update('midair', 0, 0)
        assert np.array_equal(game.banjo.pos, pos)
        zero_speed_input_displacements[ys] = pos-game_orig.banjo.pos

    target_cam_angle = ml_vec3f_yaw_between(game.banjo.pos, game._get_target_camera_pos(game.banjo.pos))

    if target_cam_angle not in gaps:
        printwrite(f, 'There are no known gaps straight forwards/backwards from here with this camera value.')
        return
    
    target_gaps = gaps[target_cam_angle]
    printwrite(f, len(target_gaps))

    nearest_gap_coord = None
    nearest_gap_dist = 999999999
    for gap in target_gaps:
        dist = np.linalg.norm(gap-game.banjo.pos)
        if dist < nearest_gap_dist:
            nearest_gap_dist = dist
            nearest_gap_coord = gap
            
    assert nearest_gap_coord is not None

    goal_coords = {}

    printwrite(f, 'Computing goal coords...')
    for gap_coords in target_gaps:
        for y1234 in zero_speed_y_inputs:
            for y5678 in zero_speed_y_inputs:
                goal_coord = gap_coords - zero_speed_input_displacements[y5678] - zero_speed_input_displacements[y1234]
                goal_coords[tuple(goal_coord)] = y1234 + y5678
                    
    printwrite(f, len(goal_coords))

    game = copy.copy(game_orig)
    printwrite(f, game)

    dist = np.linalg.norm(nearest_gap_coord - game.banjo.pos)
    printwrite(f, 'Walking towards', nearest_gap_coord, '...')

    walkforward_inputs = []
    
    is_first = True

    while dist > 5:
        game_copy = copy.copy(game)
        _, y_input = game.calculate_input_for_desired_pos_no_overshoot('walk', nearest_gap_coord)
        y_input = round(y_input)
        if abs(y_input) < 8:
            y_input = 0
        x_input = 0
        game_copy.update('walk', x_input, y_input)
        dist_new = np.linalg.norm(nearest_gap_coord - game_copy.banjo.pos)
        if dist_new+1 >= dist and not is_first:
            break
        if game_copy.camera.angle != game_orig.camera.angle:
            printwrite(f, 'Camera desync, will probably fail.')
            break
        dist = dist_new
        game = game_copy
        printwrite(f, game, x_input, y_input)
        inputstring.append({'x':x_input, 'y':y_input, 'A':0})
        walkforward_inputs.append((x_input, y_input))
        is_first = False

    if game.camera.angle != game_orig.camera.angle:
        printwrite(f, 'Camera desync, giving up')
        return

    printwrite(f, 'Searching for landing on a goal coord...')
    found = None
    
    for x2, y2 in distinct_low_magnitudes:
        for x1, y1 in distinct_inputs:
            game_copy1 = copy.copy(game)
            game_copy1.update('walk', x1, y1)
            if game_copy1.camera.angle != game_orig.camera.angle:
                continue
            angle_diff = (game.banjo.angle - game_copy1.banjo.angle)%360
            if 165 < angle_diff < 195:
                continue # prevent skidding which the sim doesn't support properly
            game_copy2 = copy.copy(game_copy1)
            game_copy2.update('walk', x2, y2)
            if game_copy2.camera.angle != game_orig.camera.angle:
                continue
            if tuple(game_copy2.banjo.pos) in goal_coords:
                printwrite(f, '!!!! FOUND !!!!')
                printwrite(f, game_copy1, x1, y1)
                inputstring.append({'x':x1, 'y':y1, 'A':0})
                printwrite(f, game_copy2, x2, y2)
                inputstring.append({'x':x2, 'y':y2, 'A':0})
                found = tuple(game_copy2.banjo.pos)
                game = game_copy2
                break
        if found:
            break

    if not found:
        printwrite(f, 'Found no results')
        return

    y_inputs = goal_coords[found]

    game.update('jump', 0, 0)
    printwrite(f, game, 'jump')
    inputstring.append({'x':0, 'y':0, 'A':1})
    for i in range(len(y_inputs)):
        game.update('midair', 0, y_inputs[i])
        printwrite(f, game, 0, y_inputs[i])
        inputstring.append({'x':x_input, 'y':y_inputs[i], 'A':1})

    if tuple(game.banjo.pos) in target_gaps:
        printwrite(f, 'Success!')
    else:
        printwrite(f, 'Failed for some totally unknown reason!')
        return
        
    s = ''
    for inputs in inputstring:
        for _ in range(2):
            s += 'joypad.setanalog({["P1 X Axis"]=%d, ["P1 Y Axis"]=%d}) ' % (inputs['x'], inputs['y'])
            if inputs['A']:
                s += 'joypad.set({["P1 A"]="True"}) '
            else:
                s += 'joypad.set({["P1 A"]="False"}) '
            s += 'emu.frameadvance() '
            
    s += 'joypad.setanalog({["P1 X Axis"]=0, ["P1 Y Axis"]=0})'
    printwrite(f, s)

    
if __name__ == '__main__':

    # Intended to be trigged via lua, not run directly
    
    banjo = Banjo(pos=(argv[1], argv[2]), speed=(argv[3], argv[4]), angle=argv[5], moving=bool(int(argv[6])))
    camera = Camera(pos=(argv[7], argv[8]), speed=(argv[9], argv[10]), angle=argv[11], angular_momentum=argv[12])
    camera_pivot = CameraPivot(argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21])
    game = Game(banjo, camera, camera_pivot)

    outfile = open('stdout.log','w')
    do_search(game, f=outfile)
    outfile.close()

