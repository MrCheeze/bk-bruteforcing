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
    diff = np.array(diff, dtype=float32)

    h = np.sqrt(diff[1]*diff[1] + diff[0]*diff[0])

    if h < 0.01:
        return 0

    yaw = ml_abs_asin_f(diff[0] / h)

    if diff[1] < 0:
        yaw = float32(180 - yaw)
    
    if diff[0] < 0:
        yaw = float32(360 - yaw)

    return float32(yaw)

def ml_vec3f_yaw_between(src, target):
    src = np.array(src, dtype=float32)
    target = np.array(target, dtype=float32)
    
    diff = target - src

    return float32(ml_vec3f_yaw_towards(diff))

def ml_sin_deg(angle_deg):
    return sinf(float32(angle_deg) * 0.017453292522222223)

def ml_cos_deg(angle_deg):
    return cosf(float32(angle_deg) * 0.017453292522222223)

def translate_magnitude_to_speed(movement_type, h):

    h = float32(h)

    if movement_type in ['walk', 'jump', 'midair']:
        
        if h <= 0.12:
            target_speed = 0
        elif 0.12 < h <= 0.2:
            target_speed = float32(h - 0.12) / float32(0.2 - 0.12) * (80 - 30) + 30
        elif 0.2 < h <= 0.5:
            target_speed = float32(h - 0.2) / float32(0.5 - 0.2) * (150 - 80) + 80
        elif 0.5 < h <= 0.75:
            target_speed = float32(h - 0.5) / float32(0.75 - 0.5) * (225 - 150) + 150
        elif 0.75 < h <= 1.0:
            target_speed = float32(h - 0.75) / float32(1.0 - 0.75) * (500 - 225) + 225
    
    elif movement_type in ['trot', 'trotjump', 'trotmidair']:

        if h <= 0.03:
            target_speed = 0
        elif 0.03 < h <= 1.0:
            target_speed = float32(h - 0.03) / float32(1.0 - 0.03) * (700 - 30) + 30
            
    else:
        raise Exception('unsupported movement type')

    if movement_type in ['walk', 'trot']:
        style = 'can_turn'
        drag_factor = 0.29
    elif movement_type in ['jump', 'trotjump']:
        style = 'instant'
        drag_factor = None
    elif movement_type in ['midair', 'trotmidair']:
        style = 'no_turn'
        drag_factor = 0.07
    else:
        raise Exception('unsupported movement type')

    return style, float32(target_speed), float32(drag_factor)
        

def move_for_input(start_pos, movement_type, cam_angle, banjo_angle, x, y, speeds, moving):
    start_pos = np.array(start_pos, dtype=float32)
    cam_angle = float32(cam_angle)
    banjo_angle = float32(banjo_angle)
    speeds = np.array(speeds, dtype=float32)

    if x < 0:
        x = min(x+7, 0)
    if x > 0:
        x = max(x-7, 0)
    x = max(x, -52)
    x = min(x, 52)
    
    if y < 0:
        y = min(y+7, 0)
    if y > 0:
        y = max(y-7, 0)
    y = max(y, -54)
    y = min(y, 54)

    x_processed = int(x * 80 / 52)
    y_processed = int(y * 80 / 54)

    h = np.sqrt(float32((x_processed/80)**2 + (y_processed/80)**2))
    h = min(h, 1)

    style, target_speed, drag_factor = translate_magnitude_to_speed(movement_type, h)

    pal_factor = float32(1.2000011)

    input_angle = float32( (ml_vec3f_yaw_towards((y_processed, x_processed)) + cam_angle + float32(90)) % 360 )
    
    if style == 'can_turn':

        if moving:

            target_speed_x = ml_sin_deg(banjo_angle) * target_speed
            target_speed_z = ml_cos_deg(banjo_angle) * target_speed
        
            speeds[0] = ((target_speed_x * drag_factor) - (speeds[0] * drag_factor)) * pal_factor + speeds[0]
            speeds[1] = ((target_speed_z * drag_factor) - (speeds[1] * drag_factor)) * pal_factor + speeds[1]

        if x != 0 or y != 0:
            banjo_angle = input_angle

    elif style == 'no_turn':

        target_speed_x = ml_sin_deg(input_angle) * target_speed
        target_speed_z = ml_cos_deg(input_angle) * target_speed

        speeds[0] = ((target_speed_x * drag_factor) - (speeds[0] * drag_factor)) * pal_factor + speeds[0]
        speeds[1] = ((target_speed_z * drag_factor) - (speeds[1] * drag_factor)) * pal_factor + speeds[1]

    elif style == 'instant':
        
        banjo_angle = input_angle

        target_speed_x = ml_sin_deg(banjo_angle) * target_speed
        target_speed_z = ml_cos_deg(banjo_angle) * target_speed
        
        speeds[0] = target_speed_x
        speeds[1] = target_speed_z
        
    else:
        raise Exception('unsupported movement style')
    
    result_pos = start_pos + speeds / 25

    moving = target_speed != 0 or speeds[0] != 0 or speeds[1] != 0

    return result_pos, speeds, banjo_angle, moving

def update_camera(pos, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum):
    pos = np.array(pos, dtype=float32)
    cam_pos = np.array(cam_pos, dtype=float32)
    cam_momentum = np.array(cam_momentum, dtype=float32)
    cam_angle = float32(cam_angle)
    cam_angle_momentum = float32(cam_angle_momentum)

    assert camera_pivot_info['min_camera_distance'] == camera_pivot_info['max_camera_distance'] # TODO handle cases where min != max
    camera_distance = float32(camera_pivot_info['max_camera_distance'])
    
    pivot_distance_x = float32(camera_pivot_info['x_center']) - pos[0]
    pivot_distance_z = float32(camera_pivot_info['z_center']) - pos[1]
    h = np.sqrt(np.square(pivot_distance_x) + np.square(pivot_distance_z))
    target_camera_x = camera_distance / h * pivot_distance_x + pos[0]
    target_camera_z = camera_distance / h * pivot_distance_z + pos[1]

    cam_direction_x = np.sign(target_camera_x - cam_pos[0])
    cam_direction_z = np.sign(target_camera_z - cam_pos[1])

    for _ in range(12):
        cam_momentum[0] += float32( ((target_camera_x - cam_pos[0]) * 0.003333 * float32(camera_pivot_info['position_factor_a']) - cam_momentum[0]) * 0.003333 * float32(camera_pivot_info['position_factor_b']) )
        cam_momentum[1] += float32( ((target_camera_z - cam_pos[1]) * 0.003333 * float32(camera_pivot_info['position_factor_a']) - cam_momentum[1]) * 0.003333 * float32(camera_pivot_info['position_factor_b']) )
        cam_pos += cam_momentum

    # Prevent overshooting
    if cam_direction_x != np.sign(target_camera_x - cam_pos[0]):
        cam_pos[0] = target_camera_x
        cam_momentum[0] = float(0)
    if cam_direction_z != np.sign(target_camera_z - cam_pos[1]):
        cam_pos[1] = target_camera_z
        cam_momentum[1] = float(0)

    target_camera_angle = ml_vec3f_yaw_between(pos, cam_pos)
    camera_angle_diff = target_camera_angle - cam_angle
    if camera_angle_diff > 180:
        camera_angle_diff -= 360

    cam_angle_momentum += float32( (camera_angle_diff * 0.04 * float32(camera_pivot_info['angle_factor_a']) - cam_angle_momentum) * 0.0333 * float32(camera_pivot_info['angle_factor_b']) )
    cam_angle += cam_angle_momentum
    cam_angle %= 360


    return float32(cam_pos), float32(cam_momentum), float32(cam_angle), float32(cam_angle_momentum)

def update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, movement_type, x_input, y_input):
    pos, speed, banjo_angle, moving = move_for_input(pos, movement_type, cam_angle, banjo_angle, x_input, y_input, speed, moving)
    cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update_camera(pos, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    return pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum


def test_260_door_movement():

    camera_pivot_info = {
        'x_center': -615.718811035,
        'y_center': 1659.86364746,
        'z_center': 1671.5916748,
        'position_factor_a': 1.75,
        'position_factor_b': 3.75,
        'angle_factor_a': 2.75,
        'angle_factor_b': 12.0,
        'min_camera_distance': 1250.75,
        'max_camera_distance': 1250.75,
    }

    pos = np.array((3406, 351), dtype=float32)
    speed = np.array((0, 0), dtype=float32)
    banjo_angle = float32(288.171051025)
    moving = False

    cam_pos = np.array((2217.65478516, 741.204345703), dtype=float32)
    cam_momentum = np.array((0.000118193223898, 0), dtype=float32)
    cam_angle = float32(288.171051025)
    cam_angle_momentum = float32(-6.71386760587e-06)

    print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    for _ in range(5):
        pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'walk', -128, -128)
        print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    for _ in range(5):
        pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'walk', 0, 0)
        print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    for _ in range(5):
        pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'walk', -128, -128)
        print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'jump', 127, 127)
    print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    for _ in range(3):
        pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'midair', 127, -128)
        print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    for _ in range(3):
        pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'midair', 0, 0)
        print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

    for _ in range(3):
        pos, speed, banjo_angle, moving, cam_pos, cam_momentum, cam_angle, cam_angle_momentum = update(pos, speed, banjo_angle, moving, camera_pivot_info, cam_pos, cam_momentum, cam_angle, cam_angle_momentum, 'midair', 127, -128)
        print('banjo', pos, speed, banjo_angle, moving, 'cam', cam_pos, cam_momentum, cam_angle, cam_angle_momentum)
    print()

if __name__ == '__main__':
    test_260_door_movement()

