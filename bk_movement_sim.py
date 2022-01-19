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

        
    def _translate_magnitude_to_speed(self, movement_type, h):

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

        style, target_speed, drag_factor = self._translate_magnitude_to_speed(movement_type, h)

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

    def _update_camera(self):

        assert self.camera_pivot.min_dist == self.camera_pivot.max_dist # TODO handle cases where min != max
        camera_distance = self.camera_pivot.max_dist
        
        pivot_distance_x = self.camera_pivot.x - self.banjo.pos[0]
        pivot_distance_z = self.camera_pivot.z - self.banjo.pos[1]
        h = np.sqrt(np.square(pivot_distance_x) + np.square(pivot_distance_z))
        target_camera_x = float32(camera_distance / h * pivot_distance_x + self.banjo.pos[0])
        target_camera_z = float32(camera_distance / h * pivot_distance_z + self.banjo.pos[1])

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


distinct_x_inputs = [-59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
distinct_y_inputs = [-61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

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

    print(game)
    print()

    for x1 in distinct_x_inputs:
        for y1 in distinct_y_inputs:
            _, _, h = game._process_joystick(x1, y1)
            if h <= 0.12 and (x1,y1) != (0,0):
                continue
            
            game_copy1 = copy.copy(game)
            game_copy1.update('jump', x1, y1)
            if game_copy1.camera.angle == game.camera.angle:
                print('jump(%3d,%3d) -- %s' % (x1, y1, game_copy1))

                '''
                for x2 in distinct_x_inputs:
                    for y2 in distinct_y_inputs:
                        _, _, h = game._process_joystick(x2, y2)
                        if h <= 0.12 and (x2,y2) != (0,0):
                            continue

                        game_copy2 = copy.copy(game_copy1)
                        game_copy2.update('midair', x2, y2)
                        if game_copy2.camera.angle == game.camera.angle:
                            print('jump(%3d,%3d) midair(%3d,%3d) -- %s' % (x1, y1, x2, y2, game_copy2))
                '''
            
    

if __name__ == '__main__':
    #test_260_door_movement()
    test_260_door_search()

