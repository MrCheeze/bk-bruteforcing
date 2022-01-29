import numpy as np
from numpy import float32, float64
import math
import copy

from bk_math_lib import *

class Banjo:
    def __init__(self, pos, speed, angle, moving=None):
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
    def __init__(self, banjo, camera, camera_pivot, is_pal):
        self.banjo = banjo
        self.camera = camera
        self.camera_pivot = camera_pivot
        self.is_pal = bool(is_pal)

    def __repr__(self):
        return 'Game(%s, %s)' % (self.banjo, self.camera)

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __copy__(self):
        return Game(copy.copy(self.banjo), copy.copy(self.camera), self.camera_pivot, self.is_pal)

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

        input_angle = float32( (ml_vec3f_yaw_towards((y_processed, x_processed)) + self.camera.angle + float32(90)) % 360 )

        if self.is_pal:
            pal_factor = float32(1.2000011)
        else:
            pal_factor = float32(1.0000011)
        
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

        if self.is_pal:
            self.banjo.pos[0] += self.banjo.speed[0] / float32(25)
            self.banjo.pos[1] += self.banjo.speed[1] / float32(25)
        else:
            self.banjo.pos[0] += self.banjo.speed[0] / float32(30)
            self.banjo.pos[1] += self.banjo.speed[1] / float32(30)

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

        if self.is_pal:
            iteration_count = 12
        else:
            iteration_count = 10

        for _ in range(iteration_count):
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

        if self.is_pal:
            frame_length = float32(1/25)
        else:
            frame_length = float32(1/30)
            
        unk_factor3 = float32(0.0333)

        self.camera.angular_momentum += (camera_angle_diff * frame_length * self.camera_pivot.ang_factor_a - self.camera.angular_momentum) * unk_factor3 * self.camera_pivot.ang_factor_b
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

        if self.is_pal:
            pal_factor = float32(1.2000011)
        else:
            pal_factor = float32(1.0000011)

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

        if self.is_pal:
            return self.calculate_input_for_desired_speed(movement_type, (desired_pos-self.banjo.pos)*25)
        else:
            return self.calculate_input_for_desired_speed(movement_type, (desired_pos-self.banjo.pos)*30)

    def calculate_input_for_desired_pos_no_overshoot(self, movement_type, desired_pos):

        assert movement_type == 'walk'

        return self.calculate_input_for_desired_speed(movement_type, (desired_pos-self.banjo.pos)*8.7)


'''
distinct_y_inputs = [-61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
distinct_y_inputs = [y for y in distinct_y_inputs if abs(y) > 13 or y == 0]


if __name__ == '__main__':
    
    game = Game(Banjo((0  , 0  ), (0.0           , 0.0           ), 0 , 0), Camera((0  , 0 ), (0            , 0   ), 0 , 0   ), CameraPivot(2474.62084960938,792.052490234375,-1855.71276855469,1.75,3.75,2.75,12,1250.5,1250.5), False)
    print(game.is_pal)
    for y1 in distinct_y_inputs:
        c1 = copy.copy(game)
        c1._update_banjo('midair', 0, y1)
        for y2 in distinct_y_inputs:
         c2 = copy.copy(c1)
         c2._update_banjo('midair', 0, y2)
         for y3 in distinct_y_inputs:
          c3 = copy.copy(c2)
          c3._update_banjo('midair', 0, y3)
          _, float_y4 = c3.calculate_input_for_desired_speed('midair', (0, 0))
          for y4 in [math.floor(float_y4), math.ceil(float_y4)]:
              game_copy = copy.copy(c3)
              game_copy._update_banjo('midair', 0, y4)
              pos = copy.copy(game_copy.banjo.pos)
              game_copy._update_banjo('midair', 0, 0)
              if np.array_equal(pos, game_copy.banjo.pos):
                  print((y1, y2, y3, y4), game_copy.banjo.pos-game.banjo.pos)'''

