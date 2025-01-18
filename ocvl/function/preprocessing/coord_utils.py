from math import sqrt

import numpy as np


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def distance(self, other_coord):
        # If it's another point, perform the calc and return it.
        if isinstance(other_coord, Point):
            return sqrt((self.x - other_coord.x)*(self.x - other_coord.x) +
                        (self.y - other_coord.y)*(self.y - other_coord.y))
        elif isinstance(other_coord, (np.ndarray, np.generic)): #If its a numpy array, treat it that way.
            return sqrt((self.x - other_coord[0]) * (self.x - other_coord[0]) +
                        (self.y - other_coord[1]) * (self.y - other_coord[1]))
        else:
            return None


