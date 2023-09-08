import argparse
import ctypes
import pathlib

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path)
    parser.add_argument('-o', '--output', default='rendered.mkv', type=pathlib.Path)
    parser.add_argument('-d', '--interval', default=30, type=int)
    parser.add_argument('--interpolation', default='bicubic', type=str)
    parser.add_argument('--dpi', default=100, type=int)
    parser.add_argument('-s', '--size', default=8, type=int)
    args = parser.parse_args()

    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=1000. / args.interval)
    fig, ax = plt.subplots(dpi=args.dpi, frameon=False)
    fig.set_size_inches(args.size, args.size, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    im = None
    with args.input.open('rb') as f:
       with writer.saving(fig, args.output, dpi=args.dpi):
            size = ctypes.c_size_t()
            f.readinto(size)
            nsteps = ctypes.c_size_t()
            f.readinto(nsteps)
            frame_ctype = ctypes.c_float * size.value * size.value
            frame = frame_ctype()
            for idx in tqdm(range(nsteps.value)):
                f.readinto(frame)
                if idx == 0:
                    im = ax.imshow(frame, interpolation=args.interpolation, aspect='equal', cmap='magma', vmin=0, vmax=1)
                else:
                    im.set_data(frame)
                writer.grab_frame()
