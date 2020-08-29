# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib

class trainer():
    def __init__(self):
        self.result_dir = 'results'
        self.cache_dir = 'cache'

        tflib.init_tf()

        # Load pre-trained network.
        url = 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl' # karras2019stylegan-ffhq-1024x1024.pkl
        with dnnlib.util.open_url(url, cache_dir=self.cache_dir) as f:
            _G, _D, self.Gs = pickle.load(f)
            # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
            # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
            # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        # Print network details.
        self.Gs.print_layers()

        # Pick latent vector.
        rnd = np.random.RandomState(5)
        latents = rnd.randn(1, self.Gs.input_shape[1])
        self.make(latents)
        latents[-1] = 0
        self.make(latents)


    def make(self, latents):
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = self.Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(self.result_dir, exist_ok=True)
        png_filename = os.path.join(self.result_dir, 'example'+str(latents[-1])+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

p = trainer()
p