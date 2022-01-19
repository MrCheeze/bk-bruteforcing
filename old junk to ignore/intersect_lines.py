import math

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

A = 3335.069, 26.46553
B = 3113.513, 118.7273
C = 3216.31, 406.5995
D = 3064.376, 456.7586

print(line_intersection((A, B), (C, D)))

C = 3489.985, 682.8096
D = 3276.096, 734.3018

print(line_intersection((A, B), (C, D)))
