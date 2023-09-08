# python utils/make_glider.py -n 256

import argparse
import ctypes
import pathlib
import skimage.transform
import numpy as np
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='glider.bin', type=pathlib.Path)
    parser.add_argument('-n', '--size', required=True, type=int)
    parser.add_argument('-c', '--scale', default=1., type=float)
    parser.add_argument('-g', '--cnt', default=5, type=int)
    parser.add_argument('-s', '--seed', default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    glider = np.array([
      [0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
      [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
      [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
      [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
      [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
      [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
      [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
      [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
      [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
      [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
      [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
      [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
      [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
      [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
      [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
      [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
      [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
      [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
      [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
      [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])

    glider_size = len(glider)
    field_ctype = ctypes.c_float * args.size * args.size
    field = field_ctype()
    offset = (args.size - glider_size) // 2
    for it in range(args.cnt):
      rotated = skimage.transform.rotate(glider, random.randrange(360))
      pos_i = random.randrange(args.size - glider_size)
      pos_j = random.randrange(args.size - glider_size)
      for i in range(glider_size):
          for j in range(glider_size):
              field[i + pos_i][j + pos_j] = rotated[i][j]
    
    with args.output.open('wb') as f:
        f.write(ctypes.c_size_t(args.size))
        f.write(ctypes.c_float(0.1))  # dT
        f.write(ctypes.c_size_t(13))  # R
        f.write(ctypes.c_float(0.15))  # GROWSH_MEAN
        f.write(ctypes.c_float(0.015))  # GROWSH_STD
        f.write(field)

