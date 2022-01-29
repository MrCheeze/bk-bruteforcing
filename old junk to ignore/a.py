import numpy as np
from numpy import float32
import struct

def toFloat(x):
    return np.frombuffer(np.array(x, dtype=np.uint32).tobytes(), dtype=float32)[0]

def fromFloat(f):
    return np.frombuffer(np.array(f, dtype=float32).tobytes(), dtype=np.uint32)[0]

def read_speeds(f):
    speed = np.frombuffer(f.read(8), dtype=f32)
    if len(speed) != 2:
        return None, None
    inputs = struct.unpack('bbbb', f.read(4))

    ret = (speed, [inputs])
    return ret ######
    #while True:
    #    speed = np.frombuffer(f.read(8), dtype=f32)
    #    if len(speed) != 2:
    #        return ret
    #    inputs = struct.unpack('bbbb', f.read(4))
    #    if speed[0] == ret[0][0] and speed[1] == ret[0][1]:
    #        ret[1].append(inputs)
    #    else:
    #        f.seek(-12, 1)
    #        return ret

f32 = np.dtype(np.float32).newbyteorder('>')
base_pos = np.array((3000, 300), dtype=float32)
twentyfive = float32(25)
thirty = float32(30)
drag_factor = float32(0.07)
pal_factor = float32(1.2000011)

distance = float32(0)
speed = float32(1)

print(distance)

while distance != distance+speed:
    distance += speed/twentyfive
    speed += (-speed * float32(0.29) * pal_factor)
    #print(speed, distance)

print(distance)
print(float32(distance))

# 0.11494242337174414
# 0.114942424
# 0.11494243

distance = float32(0)
speed = float32(1)

print(distance)

while distance != distance+speed:
    distance += speed/thirty
    speed += (-speed * float32(0.29) * float32(1.0000011))
    #print(speed, distance)

print(distance)
print(float32(distance))
1/0

# 0.11494253204306404
# 0.114942536
# 0.11494253

outfile = open('outfile.txt','w')

f_pos = open('speeds_for_angle_288.17105_positive.bin','rb')
f_neg = open('speeds_for_angle_288.17105_negative.bin','rb')

pos_speed, pos_inputs = read_speeds(f_pos)
neg_speed, neg_inputs = read_speeds(f_neg)

while pos_speed is not None and neg_speed is not None:

    decayed_pos_speed = pos_speed + (-pos_speed * drag_factor * pal_factor)
    decayed_pos_speed = decayed_pos_speed + (-decayed_pos_speed * drag_factor * pal_factor)
    decayed_pos_speed = decayed_pos_speed + (-decayed_pos_speed * drag_factor * pal_factor)
    decayed_pos_speed = decayed_pos_speed + (-decayed_pos_speed * drag_factor * pal_factor)

    sum_speed = decayed_pos_speed + neg_speed
    result_pos = base_pos + sum_speed/twentyfive

    if result_pos[0] == 3000 and result_pos[1] == 300:
        s = '%s\t%s\t%s\t%s' % (pos_speed, neg_speed, pos_inputs, neg_inputs)
        print(s)
        outfile.write(s+'\n')
        pos_speed, pos_inputs = read_speeds(f_pos)
    elif result_pos[0] < 3000:
        pos_speed, pos_inputs = read_speeds(f_pos)
    elif result_pos[0] > 3000:
        neg_speed, neg_inputs = read_speeds(f_neg)
    elif result_pos[1] < 300:
        pos_speed, pos_inputs = read_speeds(f_pos)
    elif result_pos[1] > 300:
        neg_speed, neg_inputs = read_speeds(f_neg)
    else:
        1/0

f_pos.seek(0)
f_neg.seek(0)

pos_speed, pos_inputs = read_speeds(f_pos)
neg_speed, neg_inputs = read_speeds(f_neg)

while pos_speed != None and neg_speed != None:

    decayed_neg_speed = neg_speed + (-neg_speed * drag_factor * pal_factor)
    decayed_neg_speed = decayed_neg_speed + (-decayed_neg_speed * drag_factor * pal_factor)
    decayed_neg_speed = decayed_neg_speed + (-decayed_neg_speed * drag_factor * pal_factor)
    decayed_neg_speed = decayed_neg_speed + (-decayed_neg_speed * drag_factor * pal_factor)

    sum_speed = pos_speed + decayed_neg_speed
    result_pos = base_pos + sum_speed/twentyfive

    if result_pos[0] == 3000 and result_pos[1] == 300:
        s = '%s\t%s\t%s\t%s' % (neg_speed, pos_speed, neg_inputs, pos_inputs)
        print(s)
        outfile.write(s+'\n')
        neg_speed, neg_inputs = read_speeds(f_neg)
    elif result_pos[0] < 3000:
        pos_speed, pos_inputs = read_speeds(f_pos)
    elif result_pos[0] > 3000:
        neg_speed, neg_inputs = read_speeds(f_neg)
    elif result_pos[1] < 300:
        pos_speed, pos_inputs = read_speeds(f_pos)
    elif result_pos[1] > 300:
        neg_speed, neg_inputs = read_speeds(f_neg)
    else:
        1/0

f_pos.close()
f_neg.close()
outfile.close()
