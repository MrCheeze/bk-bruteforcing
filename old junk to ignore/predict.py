import math

import numpy as np
from numpy import float32

np.set_printoptions(floatmode='unique')

round_amount1 = 0.15
round_amount2 = 0.0375
round_amount3 = None

angles = [round(round_amount1*x, 5) for x in range(round(360/round_amount1))]
speed = 1.5466682910919189453125
speed_2squared = 2*speed*speed

angles_sin = [speed*math.sin(math.radians(a)) for a in angles]
angles_cos = [speed*math.cos(math.radians(a)) for a in angles]

def toFloat(x):
    return np.frombuffer(np.array(x, dtype=np.uint32).tobytes(), dtype=float32)[0]

def mag(tup):
    return abs(tup[0]) + abs(tup[1])

def roundAngle(a, round_amount):
    if round_amount:
        return round(round((math.degrees(a)%360)/round_amount)*round_amount,5)
    else:
        return float32(math.degrees(a)%360)

def angle_towards(pos0, pos1, round_amount):
    a = math.atan2(pos1[0]-pos0[0], pos1[1]-pos0[1])
    return roundAngle(a, round_amount)

def angle_indirect_towards(pos0, pos1, round_amount):
    a_goal = math.atan2(pos1[0]-pos0[0], pos1[1]-pos0[1])
    cos_a_triangle = 1 - (pos1[0]-pos0[0])**2/speed_2squared - (pos1[1]-pos0[1])**2/speed_2squared
    a_triangle = (math.pi - math.acos(cos_a_triangle))/2
    a1 = a_goal + a_triangle
    a2 = a_goal - a_triangle
    return roundAngle(a1, round_amount), roundAngle(a2, round_amount)
    

start = float32(13), float32(13)
goal = float32(13), float32(13.12312)

min_diff = 1000
min_angles = None

for i1,a1 in enumerate(angles):
    sum1 = float32(start[0] + angles_sin[i1]), float32(start[1] + angles_cos[i1])

    a_to_indirect = angle_indirect_towards(sum1, goal, round_amount=round_amount2)

    if round_amount2:
        a2_list = [round(a_to_indirect[0]-round_amount2,5), a_to_indirect[0], round(a_to_indirect[0]+round_amount2,5), round(a_to_indirect[1]-round_amount2,5), a_to_indirect[1], round(a_to_indirect[1]+round_amount2,5)]
    else:
        a2_list = a_to_indirect
    for a2 in a2_list:
        sum2 = float32(sum1[0] + speed*math.sin(math.radians(a2))), float32(sum1[1] + speed*math.cos(math.radians(a2)))

        angle_to = angle_towards(sum2, goal, round_amount=round_amount3)
        if round_amount3:
            a3_list = [round(angle_to-round_amount3,5), angle_to, round(angle_to+round_amount3,5)]
        else:
            a3_list = [angle_to]
        for a3 in a3_list:
            result = float32(sum2[0] + speed*math.sin(math.radians(a3))), float32(sum2[1] + speed*math.cos(math.radians(a3)))
            
            mag_result = mag((result[0] - goal[0], result[1] - goal[1]))
            if mag_result < min_diff or mag_result == 0:
                min_diff = mag_result
                min_angles = (a1, a2, a3)
                print(min_diff, result, result[0]==toFloat(0x41500000), result[1]==toFloat(0x415FDC26), min_angles)
