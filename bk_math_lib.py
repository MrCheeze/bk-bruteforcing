import numpy as np
from numpy import float32, float64
import math

def sinf(x):

    x = float32(x)

    xpt = (np.frombuffer(x.tobytes(), dtype=np.int32)[0] >> 22) & 0x1FF

    if xpt < 230:
        return x
    
    dx = float64(x)

    n = 0
    if xpt >= 255:
        dn = dx * 0.3183098861837907

        if dn >= 0:
            n = int(dn + 0.5)
        else:
            n = int(dn - 0.5)
        dn = n

        dx -= dn * 3.1415926218032837
        dx -= dn * 3.178650954705639E-8
    
    xsq = dx * dx
    poly = (((((2.605780637968037E-6 * xsq) + -1.980960290193795E-4) * xsq) + 0.008333066246082155) * xsq) + -0.16666659550427756
    result = float32(((dx * xsq) * poly) + dx)

    return -result if n&1 else result

def cosf(x):

    x = float32(x)

    dx = float64(abs(x))

    dn = dx * 0.3183098861837907 + 0.5

    if dn >= 0:
        n = int(dn + 0.5)
    else:
        n = int(dn - 0.5)
    dn = n

    dx -= (dn - 0.5) * 3.1415926218032837
    dx -= (dn - 0.5) * 3.178650954705639E-8
    
    xsq = dx * dx
    poly = (((((2.605780637968037E-6 * xsq) + -1.980960290193795E-4) * xsq) + 0.008333066246082155) * xsq) + -0.16666659550427756
    result = float32(((dx * xsq) * poly) + dx)

    return -result if n&1 else result

asin_table = [int(sinf(i*math.pi/20000) * float32(65535)) for i in range(10001)]

def ml_abs_asin_f(x):

    x = float32(x)

    lower_index = 0
    upper_index = 10000
    index = 10000

    target = int(abs(x) * 65535.0)

    while target != asin_table[index]:
        index = (upper_index + lower_index) // 2

        if target < asin_table[index]:
            upper_index = index
        else:
            lower_index = index

        if upper_index - lower_index < 2:
            break

    return float32(index * 90.0 / 10000.0)

def ml_vec3f_yaw_towards(diff):
    diff_x = float32(diff[0])
    diff_z = float32(diff[1])

    h = np.sqrt(diff_x*diff_x + diff_z*diff_z)

    if h < 0.01:
        return 0

    yaw = ml_abs_asin_f(diff_x / h)

    if diff_z < 0:
        yaw = float32(180 - yaw)
    
    if diff_x < 0:
        yaw = float32(360 - yaw)

    return float32(yaw)

def ml_vec3f_yaw_between(src, target):
    
    diff_x = float32(target[0]) - float32(src[0])
    diff_z = float32(target[1]) - float32(src[1])

    return float32(ml_vec3f_yaw_towards((diff_x, diff_z)))

def ml_sin_deg(angle_deg):
    return sinf(float32(angle_deg) * 0.017453292522222223)

def ml_cos_deg(angle_deg):
    return cosf(float32(angle_deg) * 0.017453292522222223)

