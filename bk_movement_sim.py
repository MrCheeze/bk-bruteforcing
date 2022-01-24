import numpy as np
from numpy import float32, float64
import math
import copy

from bk_math_lib import *

class Banjo:
    def __init__(self, pos, speed, angle, moving):
        self.pos = np.array(pos, dtype=float32)
        self.speed = np.array(speed, dtype=float32)
        self.angle = float32(angle)
        self.moving = bool(moving)

    def __repr__(self):
        return 'Banjo((%-11s, %-11s), (%-14s, %-14s), %-10s, %d)' % (self.pos[0], self.pos[1], self.speed[0], self.speed[1], self.angle, self.moving)

    def __eq__(self, other):
        return (np.array_equal(self.pos, other.pos) and
                np.array_equal(self.speed, other.speed) and
                self.angle == other.angle and
                self.moving == other.moving)

    def __copy__(self):
        return Banjo(copy.copy(self.pos), copy.copy(self.speed), self.angle, self.moving)

class Camera:
    def __init__(self, pos, speed, angle, angular_momentum):
        self.pos = np.array(pos, dtype=float32)
        self.speed = np.array(speed, dtype=float32)
        self.angle = float32(angle)
        self.angular_momentum = float32(angular_momentum)

    def __repr__(self):
        return 'Camera((%-11s, %-11s), (%-15s, %-15s), %-10s, %-15s)' % (self.pos[0], self.pos[1], self.speed[0], self.speed[1], self.angle, self.angular_momentum)

    def __eq__(self, other):
        return (np.array_equal(self.pos, other.pos) and
                np.array_equal(self.speed, other.speed) and
                self.angle == other.angle and
                self.angular_momentum == other.angular_momentum)

    def __copy__(self):
        return Camera(copy.copy(self.pos), copy.copy(self.speed), self.angle, self.angular_momentum)

class CameraPivot:
    def __init__(self, x, y, z,
                 pos_factor_a, pos_factor_b,
                 ang_factor_a, ang_factor_b,
                 min_dist, max_dist):
        self.x = float32(x)
        self.y = float32(y)
        self.z = float32(z)
        self.pos_factor_a = float32(pos_factor_a)
        self.pos_factor_b = float32(pos_factor_b)
        self.ang_factor_a = float32(ang_factor_a)
        self.ang_factor_b = float32(ang_factor_b)
        self.min_dist = float32(min_dist)
        self.max_dist = float32(max_dist)

    def __repr__(self):
        return 'CameraPivot(%s,%s,%s,%s,%s,%s,%s,%s,%s)' % (self.x,self.y,self.z,
                                                                    self.pos_factor_a,self.pos_factor_b,
                                                                    self.ang_factor_a,self.ang_factor_b,
                                                                    self.min_dist,self.max_dist)

    def __eq__(self, other):
        return vars(self) == vars(other)

class Game:
    def __init__(self, banjo, camera, camera_pivot):
        self.banjo = banjo
        self.camera = camera
        self.camera_pivot = camera_pivot

    def __repr__(self):
        return 'Game(%s, %s)' % (self.banjo, self.camera)

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __copy__(self):
        return Game(copy.copy(self.banjo), copy.copy(self.camera), self.camera_pivot)

    def _get_style_and_drag_factor(self, movement_type):

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

        return style, float32(drag_factor)
        
    def _translate_input_magnitude_to_speed(self, movement_type, h):

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

        return float32(target_speed)
        
    def _reverse_translate_speed_to_input_magnitude(self, movement_type, target_speed):
        
        assert movement_type in ['walk', 'jump', 'midair']
            
        if target_speed == 0:
            h = 0
        elif 0 < target_speed <= 15:
            h = 0
        elif 15 < target_speed <= 30:
            h = 0.12
        elif 30 < target_speed <= 80:
            h = (target_speed - 30) / (80 - 30) * (0.2 - 0.12) + 0.12
        elif 80 < target_speed <= 150:
            h = (target_speed - 80) / (150 - 80) * (0.5 - 0.2) + 0.2
        elif 150 < target_speed <= 225:
            h = (target_speed - 150) / (225 - 150) * (0.75 - 0.5) + 0.5
        elif 225 < target_speed <= 500:
            h = (target_speed - 225) / (500 - 225) * (1.0 - 0.75) + 0.75
        elif 500 < target_speed:
            h = 1

        return h

    def _process_joystick(self, x, y):

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

        return x_processed, y_processed, h

    def _update_banjo(self, movement_type, x, y):

        x_processed, y_processed, h = self._process_joystick(x, y)

        style, drag_factor = self._get_style_and_drag_factor(movement_type)
        target_speed = self._translate_input_magnitude_to_speed(movement_type, h)

        pal_factor = float32(1.2000011)

        input_angle = float32( (ml_vec3f_yaw_towards((y_processed, x_processed)) + self.camera.angle + float32(90)) % 360 )
        
        if style == 'can_turn':

            if self.banjo.moving:

                target_speed_x = ml_sin_deg(self.banjo.angle) * target_speed
                target_speed_z = ml_cos_deg(self.banjo.angle) * target_speed
            
                self.banjo.speed[0] += ((target_speed_x * drag_factor) - (self.banjo.speed[0] * drag_factor)) * pal_factor
                self.banjo.speed[1] += ((target_speed_z * drag_factor) - (self.banjo.speed[1] * drag_factor)) * pal_factor

            if x_processed != 0 or y_processed != 0:
                self.banjo.angle = input_angle

        elif style == 'no_turn':

            target_speed_x = ml_sin_deg(input_angle) * target_speed
            target_speed_z = ml_cos_deg(input_angle) * target_speed

            self.banjo.speed[0] += ((target_speed_x * drag_factor) - (self.banjo.speed[0] * drag_factor)) * pal_factor
            self.banjo.speed[1] += ((target_speed_z * drag_factor) - (self.banjo.speed[1] * drag_factor)) * pal_factor

        elif style == 'instant':
            
            if x_processed != 0 or y_processed != 0:
                self.banjo.angle = input_angle
            
            self.banjo.speed[0] = ml_sin_deg(input_angle) * target_speed
            self.banjo.speed[1] = ml_cos_deg(input_angle) * target_speed
            
        else:
            raise Exception('unsupported movement style')

        if abs(self.banjo.speed[0]) < 0.0001:
            self.banjo.speed[0] = float32(0)
        if abs(self.banjo.speed[1]) < 0.0001:
            self.banjo.speed[1] = float32(0)
        
        self.banjo.pos[0] += self.banjo.speed[0] / float32(25)
        self.banjo.pos[1] += self.banjo.speed[1] / float32(25)

        self.banjo.moving = (target_speed != 0 or self.banjo.speed[0] != 0 or self.banjo.speed[1] != 0)


    def _get_target_camera_pos(self, pos):

        assert self.camera_pivot.min_dist == self.camera_pivot.max_dist # TODO handle cases where min != max
        camera_distance = self.camera_pivot.max_dist
        
        pivot_distance_x = self.camera_pivot.x - pos[0]
        pivot_distance_z = self.camera_pivot.z - pos[1]
        h = np.sqrt(np.square(pivot_distance_x) + np.square(pivot_distance_z))
        target_camera_x = float32(camera_distance / h * pivot_distance_x + pos[0])
        target_camera_z = float32(camera_distance / h * pivot_distance_z + pos[1])

        return target_camera_x, target_camera_z

    def _update_camera(self):

        target_camera_x, target_camera_z = self._get_target_camera_pos(self.banjo.pos)

        cam_direction_x = np.sign(target_camera_x - self.camera.pos[0])
        cam_direction_z = np.sign(target_camera_z - self.camera.pos[1])

        unk_factor = float32(0.003333)

        for _ in range(12):
            self.camera.speed[0] += ((target_camera_x - self.camera.pos[0]) * unk_factor * self.camera_pivot.pos_factor_a - self.camera.speed[0]) * unk_factor * self.camera_pivot.pos_factor_b
            self.camera.speed[1] += ((target_camera_z - self.camera.pos[1]) * unk_factor * self.camera_pivot.pos_factor_a - self.camera.speed[1]) * unk_factor * self.camera_pivot.pos_factor_b
            self.camera.pos[0] += self.camera.speed[0]
            self.camera.pos[1] += self.camera.speed[1]

        # Prevent overshooting
        if cam_direction_x != np.sign(target_camera_x - self.camera.pos[0]):
            self.camera.pos[0] = target_camera_x
            self.camera.speed[0] = float32(0)
        if cam_direction_z != np.sign(target_camera_z - self.camera.pos[1]):
            self.camera.pos[1] = target_camera_z
            self.camera.speed[1] = float32(0)

        target_camera_angle = ml_vec3f_yaw_between(self.banjo.pos, self.camera.pos)
        
        camera_angle_diff = target_camera_angle - self.camera.angle
        if camera_angle_diff > 180:
            camera_angle_diff -= float32(360)

        unk_factor2 = float32(0.04)
        unk_factor3 = float32(0.0333)

        self.camera.angular_momentum += (camera_angle_diff * unk_factor2 * self.camera_pivot.ang_factor_a - self.camera.angular_momentum) * unk_factor3 * self.camera_pivot.ang_factor_b
        self.camera.angle += self.camera.angular_momentum
        self.camera.angle %= float32(360)


    def update(self, movement_type, x_input, y_input):
        self._update_banjo(movement_type, x_input, y_input)
        self._update_camera()
        return self

    def stand_until_convergence(self):
        num_frames_waited = 0
        while True:
            old_banjo = copy.copy(self.banjo)
            self.update('walk', 0, 0)
            if np.array_equal(self.banjo.pos, old_banjo.pos) and self.banjo.angle == old_banjo.angle:
                self.banjo.speed[0] = float32(0)
                self.banjo.speed[1] = float32(0)
                self.banjo.moving = False
                break
            num_frames_waited += 1
        while True:
            old_camera = copy.copy(self.camera)
            self._update_camera()
            if np.array_equal(self.camera.pos, old_camera.pos) and self.camera.angle == old_camera.angle:
                break
            num_frames_waited += 1
        return num_frames_waited

    def calculate_input_for_desired_speed(self, movement_type, desired_speed):

        _, drag_factor = self._get_style_and_drag_factor(movement_type)
        pal_factor = 1.2000011

        target_speed_xz = ((desired_speed - self.banjo.speed) / pal_factor + (self.banjo.speed * drag_factor)) / drag_factor

        target_speed_mag = math.sqrt(sum(target_speed_xz**2))

        h = self._reverse_translate_speed_to_input_magnitude(movement_type, target_speed_mag)

        yaw = math.atan2(target_speed_xz[0], target_speed_xz[1]) - math.pi/2 - math.radians(self.camera.angle)
        x_processed, y_processed = math.cos(yaw) * h * 80, math.sin(yaw) * h * 80

        x = x_processed / 80
        y = y_processed / 80

        if abs(x) > 1:
            y /= abs(x)
            x /= abs(x)

        if abs(y) > 1:
            x /= abs(y)
            y /= abs(y)

        x *= 52
        y *= 54

        x += 7*np.sign(x)
        y += 7*np.sign(y)

        return x, y

    def calculate_input_for_desired_pos(self, movement_type, desired_pos):

        return self.calculate_input_for_desired_speed(movement_type, (desired_pos-self.banjo.pos)*25)

    def calculate_input_for_desired_pos_no_overshoot(self, movement_type, desired_pos):

        assert movement_type == 'walk'

        return self.calculate_input_for_desired_speed(movement_type, (desired_pos-self.banjo.pos)/0.11494243)



distinct_x_inputs = [-59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
distinct_y_inputs = [-61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

useless_inputs = [(-59, -61), (-59, -58), (-59, -56), (-59, -48), (-59, -42), (-59, -41), (-59, -40), (-59, -34), (-59, 34), (-59, 40), (-59, 41), (-59, 42), (-59, 48), (-59, 56), (-59, 58), (-59, 61), (-58, -60), (-58, -51), (-58, 51), (-58, 60), (-57, -60), (-57, -58), (-57, -46), (-57, 46), (-57, 58), (-57, 60), (-56, -58), (-56, -48), (-56, -41), (-56, -38), (-56, -33), (-56, 33), (-56, 38), (-56, 41), (-56, 48), (-56, 58), (-55, -56), (-55, -51), (-55, 51), (-55, 56), (-54, -60), (-54, -59), (-54, -56), (-54, -50), (-54, -44), (-54, 44), (-54, 50), (-54, 56), (-54, 59), (-54, 60), (-53, -61), (-53, -59), (-53, -58), (-53, -54), (-53, -50), (-53, -48), (-53, -39), (-53, 39), (-53, 48), (-53, 50), (-53, 54), (-53, 58), (-53, 59), (-53, 61), (-52, -57), (-52, -54), (-52, 54), (-52, 57), (-50, -60), (-50, -56), (-50, -52), (-50, 52), (-50, 56), (-50, 60), (-49, -61), (-49, -56), (-49, -50), (-49, 50), (-49, 56), (-49, 61), (-48, -50), (-48, 50), (-46, -60), (-46, -59), (-46, -58), (-46, -56), (-46, 56), (-46, 58), (-46, 59), (-46, 60), (-41, -60), (-41, 60), (-40, -58), (-40, 58), (-37, -59), (-37, 59), (-33, -61), (-33, 61), (-28, -61), (-28, 61), (-13, -9), (-13, -8), (-13, 0), (-13, 8), (-13, 9), (-12, -11), (-12, -10), (-12, -9), (-12, -8), (-12, 0), (-12, 8), (-12, 9), (-12, 10), (-12, 11), (-11, -12), (-11, -11), (-11, -10), (-11, -9), (-11, -8), (-11, 0), (-11, 8), (-11, 9), (-11, 10), (-11, 11), (-11, 12), (-10, -13), (-10, -12), (-10, -11), (-10, -10), (-10, -9), (-10, -8), (-10, 0), (-10, 8), (-10, 9), (-10, 10), (-10, 11), (-10, 12), (-10, 13), (-9, -13), (-9, -12), (-9, -11), (-9, -10), (-9, -9), (-9, -8), (-9, 0), (-9, 8), (-9, 9), (-9, 10), (-9, 11), (-9, 12), (-9, 13), (-8, -13), (-8, -12), (-8, -11), (-8, -10), (-8, -9), (-8, -8), (-8, 0), (-8, 8), (-8, 9), (-8, 10), (-8, 11), (-8, 12), (-8, 13), (0, -13), (0, -12), (0, -11), (0, -10), (0, -9), (0, -8), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (8, -13), (8, -12), (8, -11), (8, -10), (8, -9), (8, -8), (8, 0), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, -13), (9, -12), (9, -11), (9, -10), (9, -9), (9, -8), (9, 0), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (10, -13), (10, -12), (10, -11), (10, -10), (10, -9), (10, -8), (10, 0), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (11, -12), (11, -11), (11, -10), (11, -9), (11, -8), (11, 0), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (12, -11), (12, -10), (12, -9), (12, -8), (12, 0), (12, 8), (12, 9), (12, 10), (12, 11), (13, -9), (13, -8), (13, 0), (13, 8), (13, 9), (28, -61), (28, 61), (33, -61), (33, 61), (37, -59), (37, 59), (40, -58), (40, 58), (41, -60), (41, 60), (46, -60), (46, -59), (46, -58), (46, -56), (46, 56), (46, 58), (46, 59), (46, 60), (48, -50), (48, 50), (49, -61), (49, -56), (49, -50), (49, 50), (49, 56), (49, 61), (50, -60), (50, -56), (50, -52), (50, 52), (50, 56), (50, 60), (52, -57), (52, -54), (52, 54), (52, 57), (53, -61), (53, -59), (53, -58), (53, -54), (53, -50), (53, -48), (53, -39), (53, 39), (53, 48), (53, 50), (53, 54), (53, 58), (53, 59), (53, 61), (54, -60), (54, -59), (54, -56), (54, -50), (54, -44), (54, 44), (54, 50), (54, 56), (54, 59), (54, 60), (55, -56), (55, -51), (55, 51), (55, 56), (56, -58), (56, -48), (56, -41), (56, -38), (56, -33), (56, 33), (56, 38), (56, 41), (56, 48), (56, 58), (57, -60), (57, -58), (57, -46), (57, 46), (57, 58), (57, 60), (58, -60), (58, -51), (58, 51), (58, 60), (59, -61), (59, -58), (59, -56), (59, -48), (59, -42), (59, -41), (59, -40), (59, -34), (59, 34), (59, 40), (59, 41), (59, 42), (59, 48), (59, 56), (59, 58), (59, 61)]

distinct_inputs = []
for x in distinct_x_inputs:
    for y in distinct_y_inputs:
        if (x, y) not in useless_inputs:
            distinct_inputs.append((x, y))

distinct_x_inputs = [x for x in distinct_x_inputs if abs(x) > 13 or x == 0]
distinct_y_inputs = [y for y in distinct_x_inputs if abs(y) > 13 or y == 0]

low_magnitude_inputs = [(-18, 0), (-17, -11), (-17, -10), (-17, -9), (-17, -8), (-17, 0), (-17, 8), (-17, 9), (-17, 10), (-17, 11), (-16, -13), (-16, -12), (-16, -11), (-16, -10), (-16, -9), (-16, -8), (-16, 0), (-16, 8), (-16, 9), (-16, 10), (-16, 11), (-16, 12), (-16, 13), (-15, -14), (-15, -13), (-15, -12), (-15, -11), (-15, -10), (-15, -9), (-15, -8), (-15, 0), (-15, 8), (-15, 9), (-15, 10), (-15, 11), (-15, 12), (-15, 13), (-15, 14), (-14, -15), (-14, -14), (-14, -13), (-14, -12), (-14, -11), (-14, -10), (-14, -9), (-14, -8), (-14, 0), (-14, 8), (-14, 9), (-14, 10), (-14, 11), (-14, 12), (-14, 13), (-14, 14), (-14, 15), (-13, -16), (-13, -15), (-13, -14), (-13, -13), (-13, -12), (-13, -11), (-13, -10), (-13, 10), (-13, 11), (-13, 12), (-13, 13), (-13, 14), (-13, 15), (-13, 16), (-12, -17), (-12, -16), (-12, -15), (-12, -14), (-12, -13), (-12, -12), (-12, 12), (-12, 13), (-12, 14), (-12, 15), (-12, 16), (-12, 17), (-11, -17), (-11, -16), (-11, -15), (-11, -14), (-11, -13), (-11, 13), (-11, 14), (-11, 15), (-11, 16), (-11, 17), (-10, -17), (-10, -16), (-10, -15), (-10, -14), (-10, 14), (-10, 15), (-10, 16), (-10, 17), (-9, -17), (-9, -16), (-9, -15), (-9, -14), (-9, 14), (-9, 15), (-9, 16), (-9, 17), (-8, -17), (-8, -16), (-8, -15), (-8, -14), (-8, 14), (-8, 15), (-8, 16), (-8, 17), (0, -18), (0, -17), (0, -16), (0, -15), (0, -14), (0, 0), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (8, -17), (8, -16), (8, -15), (8, -14), (8, 14), (8, 15), (8, 16), (8, 17), (9, -17), (9, -16), (9, -15), (9, -14), (9, 14), (9, 15), (9, 16), (9, 17), (10, -17), (10, -16), (10, -15), (10, -14), (10, 14), (10, 15), (10, 16), (10, 17), (11, -17), (11, -16), (11, -15), (11, -14), (11, -13), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (12, -17), (12, -16), (12, -15), (12, -14), (12, -13), (12, -12), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (13, -16), (13, -15), (13, -14), (13, -13), (13, -12), (13, -11), (13, -10), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (14, -15), (14, -14), (14, -13), (14, -12), (14, -11), (14, -10), (14, -9), (14, -8), (14, 0), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (15, -14), (15, -13), (15, -12), (15, -11), (15, -10), (15, -9), (15, -8), (15, 0), (15, 8), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (16, -13), (16, -12), (16, -11), (16, -10), (16, -9), (16, -8), (16, 0), (16, 8), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (17, -11), (17, -10), (17, -9), (17, -8), (17, 0), (17, 8), (17, 9), (17, 10), (17, 11), (18, 0)]



def test_260_door_movement():

    banjo = Banjo(pos=(3406, 351), speed=(0, 0), angle=288.171051025, moving=False)
    camera = Camera(pos=(2217.65478516, 741.204345703), speed=(0.000118193223898, 0), angle=288.171051025, angular_momentum=-6.71386760587e-06)
    camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    game = Game(banjo, camera, camera_pivot)

    print(game)
    print()

    for _ in range(5):
        game.update('walk', -128, -128)
        print(game)
    print()

    for _ in range(5):
        game.update('walk', 0, 0)
        print(game)
    print()

    for _ in range(5):
        game.update('walk', -128, -128)
        print(game)
    print()

    '''
    game.update('jump', 127, 127)
    print(game)
    print()

    for _ in range(3):
        game.update('midair', 127, -128)
        print(game)
    print()

    for _ in range(3):
        game.update('midair', 0, 0)
        print(game)
    print()

    for _ in range(3):
        game.update('midair', 127, -128)
        print(game)
    print()
    '''
    game.update('jump', 0, 0)
    print(game)
    game.stand_until_convergence()
    print(game)

def test_260_door_search():

    banjo = Banjo(pos=(3406, 351), speed=(0, 0), angle=288.171051025, moving=False)
    camera = Camera(pos=(2217.65478516, 741.204345703), speed=(0.000118193223898, 0), angle=288.171051025, angular_momentum=-6.71386760587e-06)
    camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    game = Game(banjo, camera, camera_pivot)

    #banjo = Banjo(pos=(3250.00878906, 359.46295166), speed=(0, 0), angle=22.2581481934, moving=False)
    #camera = Camera(pos=(2065.62573242, 761.473388672), speed=(0, 0), angle=288.746948242, angular_momentum=6.71386760587e-06)
    #camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    #game = Game(banjo, camera, camera_pivot)

    #banjo = Banjo(pos=(0, 300), speed=(0, 0), angle=90, moving=False)
    #camera = Camera(pos=(1000, 300), speed=(0, 0), angle=90, angular_momentum=0)
    #camera_pivot = CameraPivot(1000, 0, 300, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    #game = Game(banjo, camera, camera_pivot)

    print(game)

    goal_pos = (float32(3631.65234375), float32(277.50274658203125))

    '''
    min_distance = 999999999

    candidate_inputs = [(-128, -128), (-128, 0), (-128, 127), (0, -128), (0, 127), (127, -128), (127, 0), (127, 127)]

    while True:
    
        found = False
        for x1, y1 in candidate_inputs:
                
            game_copy = copy.copy(game)
            game_copy.update('midair', x1, y1)

            if game_copy.camera.angle == game.camera.angle:
            
                for x2, y2 in candidate_inputs:
                    
                            game_copy2 = copy.copy(game_copy)
                            game_copy2.update('midair', x2, y2)
                            
                            h, leftright, updown = dist(game_copy2, goal_pos)
                            if game_copy2.camera.angle == game.camera.angle and leftright < min_distance:
                                min_distance = leftright
                                best_x1, best_y1 = x1, y1
                                best_x2, best_y2 = x2, y2
                                found = True

        if found:

            game.update('midair', best_x1, best_y1)
            print(game, 'midair', (best_x1, best_y1), dist(game, goal_pos))
            game.update('midair', best_x2, best_y2)
            print(game, 'midair', (best_x2, best_y2), dist(game, goal_pos))
        else:
            break
            
    print("done")
    '''

    '''
    ideal_angle = ml_vec3f_yaw_between(game.banjo.pos, goal_pos)

    possible_angles = {}
    for x in distinct_x_inputs:
        for y in distinct_y_inputs:
            game_copy = copy.copy(game)
            game_copy.update('walk', x, y)
            possible_angles[game_copy.banjo.angle] = (x, y)
    print(len(possible_angles))
    print(len(distinct_x_inputs) * len(distinct_y_inputs))

    while True:

        desired_angle = ml_vec3f_yaw_between(game.banjo.pos, goal_pos)

        closest_angle_diff = 99999
        for angle in possible_angles:
            diff = abs(desired_angle - angle)
            if diff < closest_angle_diff:
                closest_angle_diff = diff
                best_angle = angle
                
        best_x1, best_y1 = possible_angles[best_angle]
        
        game.update('walk', best_x1, best_y1)
        print(game, 'walk', (best_x1, best_y1), dist(game, goal_pos))
    '''

    '''
    for x in [-14,14]:
        if abs(x) < 14:# and x != 0:
            continue
        game_copy = copy.copy(game)
        game_copy.update('midair', x, 0)
        if game_copy.camera.angle == game.camera.angle:
            print(x)
            for x2 in distinct_x_inputs:
                if abs(x2) < 14 and x2 != 0:
                    continue
                game_copy2 = copy.copy(game_copy)
                game_copy2.update('midair', x2, 0)
                if game_copy2.camera.angle == game.camera.angle:
                    print(x, x2)
                    for x3 in distinct_x_inputs:
                        if abs(x3) < 14 and x3 != 0:
                            continue
                        game_copy3 = copy.copy(game_copy2)
                        game_copy3.update('midair', x3, 0)
                        if game_copy3.camera.angle == game.camera.angle:
                            for x4 in distinct_x_inputs:
                                if abs(x4) < 14 and x4 != 0:
                                    continue
                                game_copy4 = copy.copy(game_copy3)
                                game_copy4.update('midair', x4, 0)
                                if game_copy4.camera.angle == game.camera.angle:
                                    for x5 in distinct_x_inputs:
                                        if abs(x5) < 14 and x5 != 0:
                                            continue
                                        game_copy5 = copy.copy(game_copy4)
                                        game_copy5.update('midair', x5, 0)
                                        if game_copy5.camera.angle == game.camera.angle:
                                            for x6 in distinct_x_inputs:
                                                if abs(x6) < 14 and x6 != 0:
                                                    continue
                                                game_copy6 = copy.copy(game_copy5)
                                                game_copy6.update('midair', x6, 0)
                                                if game_copy6.camera.angle != game.camera.angle:
                                                    continue
                                                game_copy6.update('midair', 0, 0)
                                                if game_copy6.camera.angle != game.camera.angle:
                                                    continue
                                                game_copy6.update('midair', 0, 0)
                                                if game_copy6.camera.angle != game.camera.angle:
                                                    continue
                                                game_copy6.update('midair', 0, 0)
                                                if game_copy6.camera.angle != game.camera.angle:
                                                    continue
                                                game_copy6.update('midair', 0, 0)
                                                if game_copy6.camera.angle != game.camera.angle:
                                                    continue
                                                game_copy6.update('midair', 0, 0)
                                                if game_copy6.camera.angle != game.camera.angle:
                                                    continue
                                                prev_pos = copy.copy(game_copy6.banjo.pos)
                                                game_copy6.update('midair', 0, 0)
                                                if game_copy6.camera.angle == game.camera.angle and np.array_equal(prev_pos, game_copy6.banjo.pos):
                                                    print(game_copy6, x, x2, x3, x4, x5, x6, game_copy6.banjo.pos-prev_pos)
    '''

    '''
    y_inputs = [y for y in distinct_y_inputs if y==0 or abs(y) > 13]

    asdf = np.array((300,300), dtype=float32)

    for i in range(1):

        for y1 in y_inputs:
            game_copy1 = copy.copy(game)
            game_copy1.update('midair', 0, y1)
            if game_copy1.camera.angle != game.camera.angle or (i == 0 and max(abs(game_copy1.banjo.speed)) == 0):
                continue
            for y2 in y_inputs:
                game_copy2 = copy.copy(game_copy1)
                game_copy2.update('midair', 0, y2)
                if game_copy2.camera.angle != game.camera.angle:
                    continue
                for y3 in y_inputs:
                    game_copy3 = copy.copy(game_copy2)
                    game_copy3.update('midair', 0, y3)
                    if game_copy3.camera.angle != game.camera.angle:
                        continue
                    if np.array_equal(asdf + game_copy3.banjo.speed / float32(25), asdf):
                        best_y1 = y1
                        best_y2 = y2
                        best_y3 = y3
                        print()
                        print(game_copy1, y1)
                        print(game_copy2, y2)
                        print(game_copy3, y3)'''

    
    for x, y in distinct_inputs:
        game_copy = copy.copy(game)
        game_copy.update('walk', x, y)
        game_copy.update('walk', x, y)
        game_copy.update('jump', 0, 0)
        a = game_copy.camera.angle
        predicted_camera_angle, q = game_copy.predict_converged_camera_angle()
        game_copy.stand_until_convergence()
        if predicted_camera_angle != game_copy.camera.angle:
            print(x, y, a, ml_vec3f_yaw_between(game_copy.banjo.pos, game_copy.camera.pos), predicted_camera_angle, game_copy.camera.angle, abs(predicted_camera_angle-a))


def test_line_calculation_thing():
    import re

    banjo = Banjo(pos=(3406, 351), speed=(0, 0), angle=288.171051025, moving=False)
    camera = Camera(pos=(2217.65478516, 741.204345703), speed=(0.000118193223898, 0), angle=288.171051025, angular_momentum=-6.71386760587e-06)
    camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    game = Game(banjo, camera, camera_pivot)
    
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

    #y1, y2, y3, y4, displacement
    zero_speed_inputs = [
        [-59,-35,26,55],
        [-59,-26,51,40],
        [-59,39,51,-17],
        [-58,20,23,37],
        [-58,54,-26,33],
        [-57,52,-20,33],
        [-57,54,-44,46],
        [-56,17,-36,56],
        [-55,19,28,24],
        [-55,49,-42,49],
        [-54,-18,47,30],
        [-53,53,-45,38],
        [-52,-35,14,53],
        [-52,-16,20,44],
        [-52,46,-27,37],
        [-52,50,28,-23],
        [-51,19,18,19],
        [-51,33,-39,50],
        [-51,42,-19,32],
        [-50,39,41,-26],
        [-49,52,-39,22],
        [-48,-28,27,38],
        [-48,16,-18,43],
        [-48,35,-24,32],
        [-48,44,36,-32],
        [-48,50,18,-28],
        [-47,31,-24,35],
        [-46,-14,37,15],
        [-46,34,-51,52],
        [-45,40,-20,21],
        [-44,14,14,22],
        [-44,38,-30,31],
        [-42,-40,36,35],
        [-42,44,-35,27],
        [-42,56,-24,-19],
        [-41,-19,57,-38],
        [-40,-19,-35,54],
        [-39,23,-20,30],
        [-37,-38,44,19],
        [-37,24,47,-37],
        [-37,51,14,-34],
        [-37,59,-42,-19],
        [-36,55,-55,34],
        [-36,56,-57,39],
        [-36,57,-40,-17],
        [-35,-43,57,-22],
        [-35,18,23,-15],
        [-35,22,-37,41],
        [-35,31,46,-42],
        [-34,-44,39,28],
        [-34,32,-23,20],
        [-33,27,-32,32],
        [-33,43,31,-40],
        [-31,-48,56,-17],
        [-30,-21,15,32],
        [-29,22,-54,53],
        [-29,33,-45,38],
        [-28,-28,29,19],
        [-28,27,26,-25],
        [-28,30,42,-42],
        [-27,-43,37,24],
        [-26,-14,57,-50],
        [-26,38,-54,50],
        [-25,-26,-28,51],
        [-25,15,43,-30],
        [-25,21,-27,26],
        [-25,25,-39,35],
        [-24,-51,32,43],
        [-24,-14,54,-45],
        [-24,27,-39,33],
        [-24,54,31,-53],
        [-23,-29,-31,52],
        [-23,-21,14,28],
        [-23,14,32,-18],
        [-22,-50,35,36],
        [-22,-18,42,-14],
        [-21,47,-41,16],
        [-21,50,43,-53],
        [-20,-52,43,33],
        [-20,-42,57,-35],
        [-20,-26,19,20],
        [-20,14,35,-23],
        [-20,28,-25,17],
        [-19,-41,32,21],
        [-18,-42,54,-22],
        [-18,15,-18,23],
        [-18,40,-46,24],
        [-17,-57,-17,59],
        [-17,-48,57,-33],
        [-17,-20,40,-15],
        [-17,23,-39,29],
        [-17,36,24,-40],
        [-16,19,-48,40],
        [-16,36,-57,51],
        [-16,59,-39,-40],
        [-15,-49,38,20],
        [15,49,-38,-20],
        [16,-59,39,40],
        [16,-36,57,-51],
        [16,-19,48,-40],
        [17,-36,-24,40],
        [17,-23,39,-29],
        [17,20,-40,15],
        [17,48,-57,33],
        [17,57,17,-59],
        [18,-40,46,-24],
        [18,-15,18,-23],
        [18,42,-54,22],
        [19,41,-32,-21],
        [20,-28,25,-17],
        [20,-14,-35,23],
        [20,26,-19,-20],
        [20,42,-57,35],
        [20,52,-43,-33],
        [21,-50,-43,53],
        [21,-47,41,-16],
        [22,18,-42,14],
        [22,50,-35,-36],
        [23,-14,-32,18],
        [23,21,-14,-28],
        [23,29,31,-52],
        [24,-54,-31,53],
        [24,-27,39,-33],
        [24,14,-54,45],
        [24,51,-32,-43],
        [25,-25,39,-35],
        [25,-21,27,-26],
        [25,-15,-43,30],
        [25,26,28,-51],
        [26,-38,54,-50],
        [26,14,-57,50],
        [27,43,-37,-24],
        [28,-30,-42,42],
        [28,-27,-26,25],
        [28,28,-29,-19],
        [29,-33,45,-38],
        [29,-22,54,-53],
        [30,21,-15,-32],
        [31,48,-56,17],
        [33,-43,-31,40],
        [33,-27,32,-32],
        [34,-32,23,-20],
        [34,44,-39,-28],
        [35,-31,-46,42],
        [35,-22,37,-41],
        [35,-18,-23,15],
        [35,43,-57,22],
        [36,-57,40,17],
        [36,-56,57,-39],
        [36,-55,55,-34],
        [37,-59,42,19],
        [37,-51,-14,34],
        [37,-24,-47,37],
        [37,38,-44,-19],
        [39,-23,20,-30],
        [40,19,35,-54],
        [41,19,-57,38],
        [42,-56,24,19],
        [42,-44,35,-27],
        [42,40,-36,-35],
        [44,-38,30,-31],
        [44,-14,-14,-22],
        [45,-40,20,-21],
        [46,-34,51,-52],
        [46,14,-37,-15],
        [47,-31,24,-35],
        [48,-50,-18,28],
        [48,-44,-36,32],
        [48,-35,24,-32],
        [48,-16,18,-43],
        [48,28,-27,-38],
        [49,-52,39,-22],
        [50,-39,-41,26],
        [51,-42,19,-32],
        [51,-33,39,-50],
        [51,-19,-18,-19],
        [52,-50,-28,23],
        [52,-46,27,-37],
        [52,16,-20,-44],
        [52,35,-14,-53],
        [53,-53,45,-38],
        [54,18,-47,-30],
        [55,-49,42,-49],
        [55,-19,-28,-24],
        [56,-17,36,-56],
        [57,-54,44,-46],
        [57,-52,20,-33],
        [58,-54,26,-33],
        [58,-20,-23,-37],
        [59,-39,-51,17],
        [59,26,-51,-40],
        [59,35,-26,-55],
    ]

    for i in range(len(zero_speed_inputs)):
        y1, y2, y3, y4 = zero_speed_inputs[i]
        game = copy.copy(game_orig)
        game._update_banjo('jump', 0, 0)
        game._update_banjo('midair', 0, y1)
        game._update_banjo('midair', 0, y2)
        game._update_banjo('midair', 0, y3)
        game._update_banjo('midair', 0, y4)
        pos = copy.copy(game.banjo.pos)
        game.update('midair', 0, 0)
        assert np.array_equal(game.banjo.pos, pos)
        zero_speed_inputs[i].append(pos-game_orig.banjo.pos)

    target_cam_angle = ml_vec3f_yaw_between(game.banjo.pos, game._get_target_camera_pos(game.banjo.pos))
    target_gaps = gaps[target_cam_angle]
    print(len(target_gaps))

    average_gap_coord = np.mean(target_gaps, axis=0)

    goal_coords = {}

    print('Computing goal coords...')
    for gap_coords in target_gaps:
        for y1, y2, y3, y4, displacement1 in zero_speed_inputs:
            for y5, y6, y7, y8, displacement2 in zero_speed_inputs:
                goal_coord = gap_coords - displacement2 - displacement1
                goal_coords[tuple(goal_coord)] = y1, y2, y3, y4, y5, y6, y7, y8
    print(len(goal_coords))

    game = copy.copy(game_orig)
    print(game)

    dist = max(abs(average_gap_coord - game.banjo.pos))
    print('Walking towards', average_gap_coord, '...')

    walkforward_inputs = []

    while dist >= 10:
        game_copy = copy.copy(game)
        x_input, y_input = game.calculate_input_for_desired_pos_no_overshoot('walk', average_gap_coord)
        x_input = round(x_input)
        if abs(x_input) < 8:
            x_input = 0
        y_input = round(y_input)
        if abs(y_input) < 8:
            y_input = 0
        if (x_input, y_input) == (0, 0):
            break
        game_copy.update('walk', x_input, y_input)
        dist_new = max(abs(average_gap_coord - game_copy.banjo.pos))
        if dist_new > dist or game_copy.camera.angle != game_orig.camera.angle:
            break
        dist = dist_new
        game = game_copy
        print(game, x_input, y_input)
        walkforward_inputs.append((x_input, y_input))

    if game.camera.angle != game_orig.camera.angle:
        print('Camera desync, giving up')
        return

    print('Searching for landing on a goal coord...')
    found = None
    
    for x1, y1 in distinct_inputs:
        game_copy1 = copy.copy(game)
        game_copy1.update('walk', x1, y1)
        if game_copy1.camera.angle != game_orig.camera.angle:
            continue
        for x2, y2 in [(13,10)]:
            game_copy2 = copy.copy(game_copy1)
            game_copy2.update('walk', x2, y2)
            if game_copy2.camera.angle != game_orig.camera.angle:
                continue
            if tuple(game_copy2.banjo.pos) in goal_coords:
                print('!!!! FOUND !!!!')
                print(game_copy1, x1, y1)
                print(game_copy2, x2, y2)
                found = tuple(game_copy2.banjo.pos)
                game = game_copy2
                break
        if found:
            break

    if not found:
        print('Found no results')
        return

    y_inputs = goal_coords[found]

    game.update('jump', 0, 0)
    print(game, 'jump')
    for i in range(8):
        game.update('midair', 0, y_inputs[i])
        print(game, 0, y_inputs[i])

    if tuple(game.banjo.pos) in target_gaps:
        print('Success!')
    else:
        print('Failed for some totally unknown reason!')
    return

    prev_x = None
    for x,y in distinct_inputs:
        break
        if x == 0:
            continue
        if x != prev_x:
            print(x)
        prev_x = x
        game = copy.copy(game_orig)
        game.update('walk', x, y)
        game.update('walk', 13, 10)
        game.update('jump', 0, 0)

        if game.camera.angle != game_orig.camera.angle or np.array_equal(game.banjo.pos, game_orig.banjo.pos):
            continue

        banjo_coords_orig = copy.copy(game.banjo.pos)
        actual_cam_angle = game.camera.angle

        input_angle = float32( (float32(90) + actual_cam_angle + float32(90)) % 360 )
        sin_input_angle = ml_sin_deg(input_angle)
        cos_input_angle = ml_cos_deg(input_angle)

        game_beforeforwards = game

        for goal_coord in goal_coords:

            game = copy.copy(game_beforeforwards)

            dist = max(abs(goal_coord - game.banjo.pos))

            walkforward_inputs = []

            while True:
                game_copy = copy.copy(game)
                x_input, y_input = game.calculate_input_for_desired_pos('walk', goal_coord)
                x_input = round(x_input)
                y_input = round(y_input)
                game_copy.update('walk', x_input, y_input)
                if tuple(game_copy.banjo.pos) in goal_coords:
                    print('!!!!!!!!!')
                    print(game_copy)
                    1/0
                dist_new = max(abs(goal_coord - game_copy.banjo.pos))
                if dist_new > dist or game_copy.camera.angle != game_orig.camera.angle:
                    break
                dist = dist_new
                game = game_copy
                walkforward_inputs.append((x_input, y_input))

            if game.camera.angle != game_orig.camera.angle:
                continue

            if dist < 0.01:
                print(dist)
            if dist == 0:
                break

        for gap_coords in gaps[target_cam_angle]:
            break ####

            game = copy.copy(game_beforeforwards)

            dist = max(abs(gap_coords - game.banjo.pos))

            walkforward_inputs = []

            while True:
                game_copy = copy.copy(game)
                x_input, y_input = game.calculate_input_for_desired_pos('walk', gap_coords)
                x_input = round(x_input)
                y_input = round(y_input)
                game_copy.update('walk', x_input, y_input)
                dist_new = max(abs(gap_coords - game_copy.banjo.pos))
                if dist_new > dist or game_copy.camera.angle != game_orig.camera.angle:
                    break
                dist = dist_new
                game = game_copy
                walkforward_inputs.append((x_input, y_input))

            if game.camera.angle != game_orig.camera.angle:
                continue

            target_speed = np.hypot(gap_coords[0]-game.banjo.pos[0], gap_coords[1]-game.banjo.pos[1])

            if target_speed > 10:
                continue
            
            speeds = np.array((sin_input_angle * target_speed, cos_input_angle * target_speed), dtype=float32)

            reached_pos = game.banjo.pos+speeds

            if np.array_equal(gap_coords, reached_pos):
                    
                print('candidate...', x, y, walkforward_inputs, banjo_coords_orig, actual_cam_angle, game.banjo.pos, gap_coords, target_speed)

                game.update('jump', 0, 0)
                print(game)
                1/0

                '''
                for y1 in y_inputs:
                    print(x, y, y1)
                    game_copy1 = copy.copy(game)
                    game_copy1.update('midair', 0, y1)
                    for y2 in y_inputs:
                        game_copy2 = copy.copy(game_copy1)
                        game_copy2.update('midair', 0, y2)
                        for y2b in y_inputs:
                            game_copy2b = copy.copy(game_copy2)
                            game_copy2b.update('midair', 0, y2b)
                            float_x3, float_y3 = game_copy2.calculate_input_for_desired_pos('midair', gap_coords)
                            x3 = round(float_x3)
                            for y3 in [round(float_y3)]:
                                game_copy3 = copy.copy(game_copy2b)
                                game_copy3.update('midair', x3, y3)
                                if np.array_equal(game_copy3.banjo.pos, gap_coords):
                                    float_x4, float_y4 = game_copy3.calculate_input_for_desired_speed('midair', (0, 0))
                                    x4 = round(float_x4)
                                    for y4 in [round(float_y4)]:
                                        game_copy4 = copy.copy(game_copy3)
                                        game_copy4.update('midair', x4, y4)
                                        if np.array_equal(game_copy4.banjo.pos, gap_coords):
                                            print('found!!!!', x, y, walkforward_inputs, y1, y2, y2b, (x3, y3), (x4, y4), banjo_coords_orig, actual_cam_angle, game.banjo.pos, gap_coords)
                                            '''

def test_line_calculation_thing2():
    banjo = Banjo((3630.4082  , 277.49838  ), (0.0           , 0.0           ), 108.17105 , 0)
    camera = Camera((2269.8596  , 724.14606  ), (0.72773755     , -0.23853453    ), 288.17105 , -6.7138676e-06 )
    camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    game = Game(banjo, camera, camera_pivot)

    goal = np.array((3630.5452, 277.45343), dtype=float32)

    import random

    #game.banjo.pos = np.array([0, 0], dtype=float32)
    #game.camera.angle = float32(random.random()*360)
    #game.camera.angle = float32(0)
    
    banjo = Banjo((0, 0), (0, 0), 0, 0)
    camera = Camera((0, 0), (0, 0), 0, 0)
    game = Game(banjo, camera, None)

    game.banjo.pos = np.array([300, 300], dtype=float32)
    game.camera.angle = float32(random.random()*360)

    for y1 in distinct_y_inputs:
     game_copy1 = copy.copy(game)
     game_copy1._update_banjo('midair', 0, y1)
     for y2 in distinct_y_inputs:
      game_copy2 = copy.copy(game_copy1)
      game_copy2._update_banjo('midair', 0, y2)
      for y3 in distinct_y_inputs:
       game_copy3 = copy.copy(game_copy2)
       game_copy3._update_banjo('midair', 0, y3)
       float_x4, float_y4 = game_copy3.calculate_input_for_desired_speed('midair', (0, 0))
       for y4 in [math.floor(float_y4), math.ceil(float_y4)]:
        game_copy4 = copy.copy(game_copy3)
        game_copy4._update_banjo('midair', 0, y4)
        prev_pos = copy.copy(game_copy4.banjo.pos)
        game_copy4._update_banjo('midair', 0, 0)
        if np.array_equal(prev_pos, game_copy4.banjo.pos):
            print(game_copy4, y1, y2, y3, y4, prev_pos-game.banjo.pos)

    '''
    for y1 in distinct_y_inputs:
     game_copy1 = copy.copy(game)
     game_copy1._update_banjo('midair', 0, y1)
     for y2 in distinct_y_inputs:
      game_copy2 = copy.copy(game_copy1)
      game_copy2._update_banjo('midair', 0, y2)
      for y3 in distinct_y_inputs:
       game_copy3 = copy.copy(game_copy2)
       game_copy3._update_banjo('midair', 0, y3)
       float_x4, float_y4 = game_copy3.calculate_input_for_desired_speed('midair', (0, 0))
       for y4 in [math.floor(float_y4), math.ceil(float_y4)]:
        game_copy4 = copy.copy(game_copy3)
        game_copy4._update_banjo('midair', 0, y4)
        #for i in range(12):
        #    prev_pos = copy.copy(game_copy4.banjo.pos)
        #    game_copy4._update_banjo('midair', 0, 0)
        #    if np.array_equal(prev_pos, game_copy4.banjo.pos):
        #        diff = game_copy4.banjo.pos-game.banjo.pos
        #        if max(abs(diff)) < 1:
        #            print(diff, [y1, y2, y3, y4] + [0]*i)
        #        break
    '''
         
    '''
    diffs = [np.array(x, dtype=float32) for x in set([
    ])]

    min_err = 0.1

    for i1, d1 in enumerate(diffs):
     for i2, d2 in enumerate(diffs, i1):
      for i3, d3 in enumerate(diffs, i2):
       #for i4, d4 in enumerate(diffs, i3):
        err = max(abs(game.banjo.pos+d1+d2+d3 - goal))
        if err < min_err:
            min_err = err
            print(game.banjo.pos+d1+d2+d3, err, d1, d2, d3)
        if err == 0:
            print('found')
            1/0
    '''

    '''
    game_orig = game

    #outfile = open('distances_speeds_for_angle_288.17105.bin', 'wb')

    for y1 in distinct_y_inputs:
     print(y1)
     game_copy1 = copy.copy(game_orig)
     game_copy1._update_banjo('midair', 0, y1)
     for y2 in distinct_y_inputs:
      print(y1, y2)
      game_copy2 = copy.copy(game_copy1)
      game_copy2._update_banjo('midair', 0, y2)
      for y3 in distinct_y_inputs:
       game_copy3 = copy.copy(game_copy2)
       game_copy3._update_banjo('midair', 0, y3)
       for y4 in distinct_y_inputs:
        game_copy4 = copy.copy(game_copy3)
        game_copy4._update_banjo('midair', 0, y4)
        outfile.write(bytes(reversed(game_copy4.banjo.speed[0].tobytes())))
        outfile.write(bytes(reversed(game_copy4.banjo.speed[1].tobytes())))
        outfile.write(y1.to_bytes(1, 'little', signed=True))
        outfile.write(y2.to_bytes(1, 'little', signed=True))
        outfile.write(y3.to_bytes(1, 'little', signed=True))
        outfile.write(y4.to_bytes(1, 'little', signed=True))
    outfile.close()
    '''

    """
    max_dist = 0.01
    min_speed = 999999
    
    for y1 in distinct_y_inputs:
        if abs(y1) < 14 and y1 != 0:
            continue
        if y1 == 0:
            continue
        game_copy1 = copy.copy(game)
        game_copy1.update('midair', 0, y1)
        for y2 in distinct_y_inputs:
            if abs(y2) < 14 and y2 != 0:
                continue
            game_copy2 = copy.copy(game_copy1)
            game_copy2.update('midair', 0, y2)
            for y3 in distinct_y_inputs:
                if abs(y3) < 14 and y3 != 0:
                    continue
                game_copy3 = copy.copy(game_copy2)
                game_copy3.update('midair', 0, y3)
                #float_x4, float_y4 = game_copy3.calculate_input_for_desired_pos('midair', goal)
                float_x4, float_y4 = game_copy3.calculate_input_for_desired_speed('midair', (0, 0))
                for y4 in [math.floor(float_y4), math.ceil(float_y4)]:
                    #if abs(y4) < 14 and y4 != 0:
                    #    continue
                    game_copy4 = copy.copy(game_copy3)
                    game_copy4.update('midair', 0, y4)
                    dist = max(abs(game_copy4.banjo.pos - goal))
                    if dist < max_dist:
                        max_dist = dist
                        #print(game_copy4, y1, y2, y3, y4, dist)
                    #if goal[0] == game_copy4.banjo.pos[0] and goal[1] == game_copy4.banjo.pos[1]:
                    #    print(game_copy4, y2, y3, y4)
                    if game_copy4.banjo.speed[0] == game_copy4.banjo.speed[1] == 0:
                        print(game_copy4.banjo.pos-game.banjo.pos, game_copy4.banjo.speed, y1, y2, y3, y4)
                        '''
                        float_x5, float_y5 = game_copy4.calculate_input_for_desired_speed('midair', (0,0))
                        for y5 in distinct_y_inputs:
                            if abs(y5) < 14 and y5 != 0:
                                continue
                            game_copy5 = copy.copy(game_copy4)
                            game_copy5.update('midair', 0, y5)
                            for y6 in distinct_y_inputs:
                                if abs(y6) < 14 and y6 != 0:
                                    continue
                                game_copy6 = copy.copy(game_copy5)
                                game_copy6.update('midair', 0, y6)
                                float_x7, float_y7 = game_copy6.calculate_input_for_desired_pos('midair', goal)
                                for y7 in distinct_y_inputs:#[math.floor(float_y7), math.ceil(float_y7)]:
                                    if abs(y7) < 14 and y7 != 0:
                                        continue
                                    game_copy7 = copy.copy(game_copy6)
                                    game_copy7.update('midair', 0, y7)
                                    if True:#if goal[0] == game_copy7.banjo.pos[0] and goal[1] == game_copy7.banjo.pos[1]:
                                        #print(game_copy7, y1, y2, y3, y4, y5, y6, y7)
                                        float_x8, float_y8 = game_copy7.calculate_input_for_desired_pos('midair', goal)
                                        for y8 in [math.floor(float_y8), math.ceil(float_y8)]:
                                            game_copy8 = copy.copy(game_copy7)
                                            game_copy8.update('midair', 0, y8)
                                            if goal[0] == game_copy8.banjo.pos[0] and goal[1] == game_copy8.banjo.pos[1]:
                                                print(game_copy8, y1, y2, y3, y4, y5, y6, y7, y8)
                                        
                                            #speed = max(abs(game_copy5.banjo.speed))
                                            #if speed < min_speed:
                                            #    min_speed = speed
                                            #    print(game_copy5, y1, y2, y3, y4, y5, speed)
                                            #if goal[0] == game_copy5.banjo.pos[0] and goal[1] == game_copy5.banjo.pos[1]:
                                            #    print('!!!!!!!!!!!!!!!! Found !!!!!!!!!!!!!!!!')
                                            #    print(game_copy5, y1, y2, y3, y4, y5)
                                            #    print('!!!!!!!!!!!!!!!! Found !!!!!!!!!!!!!!!!')'''"""

def test_input_guesser():

    banjo = Banjo(pos=(3406, 351), speed=(0, 0), angle=288.171051025, moving=False)
    camera = Camera(pos=(2217.65478516, 741.204345703), speed=(0.000118193223898, 0), angle=288.171051025, angular_momentum=-6.71386760587e-06)
    camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    game = Game(banjo, camera, camera_pivot)

    game.update('midair', 33, -8)
    
    desired_speed = np.array((0, 10), dtype=float32)

    x, y = game.calculate_input_for_desired_speed('midair', desired_speed)

    print(x, y)
    print(round(x), round(y))
    
    min_h_error = 999999

    for x,y in distinct_inputs:
        s = copy.copy(game).update('midair', x, y).banjo.speed
        h_error = np.hypot(s[1]-desired_speed[1], s[0]-desired_speed[0])

        if h_error < min_h_error:
            min_h_error = h_error
            best_input_h = (x, y)

    print(best_input_h)

    desired_pos = np.array((3406, 351), dtype=float32)

    x, y = game.calculate_input_for_desired_pos('midair', desired_pos)

    print(x, y)
    print(round(x), round(y))
    
    min_h_error = 999999

    for x,y in distinct_inputs:
        p = copy.copy(game).update('midair', x, y).banjo.pos
        h_error = np.hypot(p[1]-desired_pos[1], p[0]-desired_pos[0])

        if h_error < min_h_error:
            min_h_error = h_error
            best_input_h = (x, y)

    print(best_input_h)

def test_some_jump():

    banjo = Banjo(pos=(3634.0786,  277.05), speed=(0, 0), angle=288.171051025, moving=False)
    camera = Camera(pos=(2217.65478516, 741.204345703), speed=(0.000118193223898, 0), angle=288.171051025, angular_momentum=-6.71386760587e-06)
    camera_pivot = CameraPivot(-615.7188, 1659.8636, 1671.5917, 1.75, 3.75, 2.75, 12.0, 1250.75, 1250.75)
    game = Game(banjo, camera, camera_pivot)

    game.stand_until_convergence()

    print(game)
    game.update('jump',0,0)
    game.update('midair',0, -52)
    print(game)
    game.update('midair',0, 27)
    print(game)
    game.update('midair',0, 34)
    print(game)
    
if __name__ == '__main__':
    #test_260_door_movement()
    #test_260_door_search()
    test_line_calculation_thing()
    #test_line_calculation_thing2()
    #test_input_guesser()
    #test_some_jump()

