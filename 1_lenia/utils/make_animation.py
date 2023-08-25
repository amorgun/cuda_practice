import argparse
import ctypes
import pathlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path)
    parser.add_argument('-o', '--output', default='rendered.mkv', type=pathlib.Path)
    parser.add_argument('-d', '--interval', default=30, type=int)
    parser.add_argument('--interpolation', default='bicubic', type=str)
    args = parser.parse_args()
    with args.input.open('rb') as f:
        size = ctypes.c_size_t()
        f.readinto(size)
        nsteps = ctypes.c_size_t()
        f.readinto(nsteps)
        frame_ctype = ctypes.c_float * size.value * size.value * nsteps.value
        frames = frame_ctype()
        f.readinto(frames)

    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=1000. / args.interval)
    fig, ax = plt.subplots(dpi=100, frameon=False)
    fig.set_size_inches(8, 8, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    im = ax.imshow(frames[0], interpolation=args.interpolation, aspect='equal', cmap='magma', vmin=0, vmax=1)
    with writer.saving(fig, args.output, 100):
        for idx in range(nsteps.value):
            im.set_data(frames[idx])
            writer.grab_frame()
