# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import argparse
import os
import pickle
import sys
from io import BytesIO
from multiprocessing import cpu_count
from pathlib import Path
from queue import Queue
from threading import Thread

import PIL.Image
import numpy as np

import config
import dnnlib
import dnnlib.tflib as tflib

image_queue = Queue(maxsize=100)
io_queue = Queue(maxsize=100)


# Thread function: convert image data into image buffer
def T_parse_image():
    while True:
        item = image_queue.get()
        if item is None:
            io_queue.put(None)
            break
        else:
            im, path = item
            im = PIL.Image.fromarray(im, 'RGB')
            io = BytesIO()
            im.save(io, format='png')
            io_queue.put((io, path))


# Thread function: save image buffer into file
# It's better to do IO works in one thread especially when it's on HDD
def T_save_image(num_threads):
    none_cnt = 0
    while True:
        item = io_queue.get()
        if item is None:
            none_cnt += 1
            if none_cnt == num_threads:
                break
        else:
            io, path = item
            print(path)
            with open(path, 'wb') as f:
                f.write(io.getvalue())
            io.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result_dir', type=str, help='Directory path in which result images will be saved')
    p.add_argument('num_images', type=int, help='Number of images to generate')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size. A batch usually takes 2gb of GPU memory (default: %(default)s)')
    p.add_argument('--num-threads', type=int, default=cpu_count(), help='Number of threads (default: %(default)s)')
    args = p.parse_args(sys.argv[1:])

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    num_images = args.num_images
    batch_size = args.batch_size
    num_threads = args.num_threads

    # Initialize TensorFlow.
    tflib.init_tf()

    # Try to load pre-trained network
    if os.path.exists('models/karras2019stylegan-ffhq-1024x1024.pkl'):
        with open('models/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
            _G, _D, Gs = pickle.load(f)
    else:
        try:
            url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
            with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
                _G, _D, Gs = pickle.load(f)
                # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
                # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
                # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        except:
            print('Cannot download pretrained network.')
            print('Download the network from https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
                  ' and move the file "karras2019stylegan-ffhq-1024x1024.pkl" into "./models"')

    # Print network details.
    Gs.print_layers()

    # Create threads
    print('Create', num_threads, 'threads ...')
    image_threads = []
    for i in range(num_threads):
        t = Thread(target=T_parse_image, name=f'ThreadImageParser_{i}', daemon=True)
        image_threads.append(t)
        t.start()
    io_thread = Thread(target=T_save_image, name='ThreadImageSaver', daemon=True, args=(num_threads,))
    io_thread.start()

    # Pick latent vector.
    i = 0
    while i < num_images:
        t = num_images if i + batch_size > num_images else i + batch_size
        latents = []
        paths = []
        for j in range(i, t):
            rnd = np.random.RandomState(j)
            latent = rnd.randn(1, Gs.input_shape[1])
            latents.append(latent)
            paths.append(result_dir / f'seed{j:06d}.png')
        i = t
        latents = np.concatenate(latents)

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=1.0, randomize_noise=True, output_transform=fmt)

        # Save image.
        for j in range(len(paths)):
            image_queue.put((images[j], paths[j]))

    # Close threads
    for i in range(num_threads):
        image_queue.put(None)
    for t in image_threads:
        t.join()
    io_thread.join()
    print('done')


if __name__ == "__main__":
    main()
