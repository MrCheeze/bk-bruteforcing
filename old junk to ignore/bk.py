from numpy import float32
import numpy as np

np.set_printoptions(floatmode='unique')

def toFloat(x):
    return np.frombuffer(np.array(x, dtype=np.uint32).tobytes(), dtype=float32)[0]

def fromFloat(f):
    return np.frombuffer(np.array(f, dtype=float32).tobytes(), dtype=np.uint32)[0]

motions = [
    (toFloat(0x40400001), -59),
    (toFloat(0x403B3334), -58),
    (toFloat(0x40366667), -57),
    (toFloat(0x40340001), -56),
    (toFloat(0x402F3334), -55),
    (toFloat(0x402CCCCD), -54),
    (toFloat(0x40280001), -53),
    (toFloat(0x4025999A), -52),
    (toFloat(0x4020CCCD), -51),
    (toFloat(0x401E6667), -50),
    (toFloat(0x4019999A), -49),
    (toFloat(0x40173334), -48),
    (toFloat(0x40126667), -47),
    (toFloat(0x40100000), -46),
    (toFloat(0x400B3334), -45),
    (toFloat(0x40066667), -44),
    (toFloat(0x40040000), -43),
    (toFloat(0x3FFE6668), -42),
    (toFloat(0x3FF9999C), -41),
    (toFloat(0x3FF00001), -40),
    (toFloat(0x3FEB3334), -39),
    (toFloat(0x3FE1999B), -38),
    (toFloat(0x3FDCCCCE), -37),
    (toFloat(0x3FD33334), -36),
    (toFloat(0x3FCE6668), -35),
    (toFloat(0x3FC4CCCD), -34),
    (toFloat(0x3FC00001), -33),
    (toFloat(0x3FB66667), -32),
    (toFloat(0x3FACCCCD), -31),
    (toFloat(0x3FA80001), -30),
    (toFloat(0x3F9E6667), -29),
    (toFloat(0x3F99999A), -28),
    (toFloat(0x3F900000), -27),
    (toFloat(0x3F8B3334), -26),
    (toFloat(0x3F81999A), -25),
    (toFloat(0x3F79999C), -24),
    (toFloat(0x3F666668), -23),
    (toFloat(0x3F5CCCCE), -22),
    (toFloat(0x3F49999B), -21),
    (toFloat(0x3F400001), -20),
    (toFloat(0x3F2CCCCD), -19),
    (toFloat(0x3F19999A), -18),
    (toFloat(0x3F100000), -17),
    (toFloat(0x3EF9999C), -16),
    (toFloat(0x3EE66668), -15),
    (toFloat(0x3EC00001), -14),
    (toFloat(0x3EACCCCD), -13),
    (toFloat(0x3E866667), -12),
    (toFloat(0x3E666668), -11),
    (toFloat(0x3E19999A), -10),
    (toFloat(0x3DE66668), -9),
    (toFloat(0x3D19999A), -8),
]

for x,_ in motions:
    print(x, 3/x)
1/0

for i in reversed(range(len(motions))):
    motions.append((-motions[i][0], -motions[i][1]))

motions.append((toFloat(0x00000000), 0))


def minDiff(start):
    minDiff = 1000
    motionList = None
    minResult = None
    for m1, input1 in motions:
        for m2, input2 in motions:
            for m3, input3 in motions:
                    result = ((start + m1) + m2) + m3
                    diff = abs(result - start)
                    if diff < minDiff and diff != 0:
                        minDiff = diff
                        motionList = (input1, input2, input3)
                        minResult = result
                        
    print(start, hex(fromFloat(start)), minResult, hex(fromFloat(minResult)), motionList)

for i in list(range(10)) + list(range(10, 360, 10)):
    minDiff(float32(i))
