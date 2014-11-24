from src.utils.helper import *
import numpy as np

# toy sample vector fields:

# (3,3,3)
v1 = np.array([
              [
                [1, 4, 1],
                [4, 16, 4],
                [1, 4, 1]
              ],
              [
                [16, 26, 16],
                [26, 41, 26],
                [16, 26, 16]
              ],
              [
                [4, 7, 4,],
                [7, 16, 7],
                [4, 7, 4]
              ]
              ])

# (3,4,4)
v2 = np.array([
              [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
              ],
              [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
              ],
              [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
              ]
              ])

# 3d vector field on a grid of dimension (5,5,5)
v3 = np.array([
               np.random.normal(0, 1, (5,5,5)),
               np.random.normal(0, 1, (5,5,5)),
               np.random.normal(0, 1, (5,5,5))
              ])