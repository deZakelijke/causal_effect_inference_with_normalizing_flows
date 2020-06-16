""" Gym environment for space objects task in 2D.

Does it make sense to do this in a Gym env? We don't really have an episode
or multiple states. Most likely we only have an initial state and one action.
So we can't even do mutliple steps. If we were to do this in a Gym env we would
have to end each episode after the first action. Doe it then make sense to do
it like this?
Since the data generation process consists of 'playing' the env with random actions,
we could also just cut out the middel man. We can sample the prior, sample the 
intervention, generate the proxy and then calculate the outcome. This could be
parallelised massively to generate a lot of data quickly.

"""

import matplotlib.pyplot as plt
import numpy as np
import skimage

from itertools import product
from numpy import random
from PIL import Image
from scipy.special import softmax


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width//2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


class SpaceShapesGenerator():
    """Gym environment for space objects task."""

    def __init__(self, width=5, height=5, num_objects=5, prior_dims=64,
                 seed=None):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.prior_dims = prior_dims

        self.colors = get_colors(num_colors=max(9, self.num_objects))

        # rng = default_rng()
        self.intervention_map = random.standard_normal((prior_dims + num_objects, 2))
        self.position_map = random.standard_normal((prior_dims + num_objects,
                                                width * height))
        self.object_gravity = random.standard_normal((1, num_objects))

    def generate_data(self, nr_samples):
        prior_data = random.standard_normal((nr_samples, self.prior_dims))
        object_gravity = np.tile(self.object_gravity, nr_samples)
        prior_data = np.concatenate((prior_data, object_gravity), axis=1)
        position_probs = softmax(prior_data @ self.position_map)
        position_probs = np.reshape(position_probs, (self.height, self.width))
        object_positions = self.set_positions(position_probs)

        steering = prior_data @ self.intervention_map

        rendering = self.render(object_positions)
        plt.imshow(rendering)
        plt.show()
        print()

        move_result = self.move_spaceship(object_positions, object_gravity, steering)
        rendering = self.render(move_result[0])
        plt.imshow(rendering)
        plt.show()

    def move_spaceship(self, object_positions, object_gravity, steering):
        """
        """
        distances = object_positions[1:] - object_positions[0]
        weighted_distances = distances * object_gravity.T
        direction = (steering + np.sum(weighted_distances, axis=0)) / 2
        new_pos = object_positions[0] + direction
        new_dist = np.linalg.norm(object_positions[:1] - new_pos, axis=-1)
        while self.valid_new_pos(object_positions, new_pos, new_dist):
            direction *= 0.5
            new_pos = object_positions[0] + direction
            new_dist = np.linalg.norm(object_positions[1:] - new_pos, axis=-1)
        object_positions[0] = new_pos 
        return object_positions, direction

    def valid_new_pos(self, object_positions, new_pos, new_dist, threshold=0.9):
        if np.any(new_dist < 1):
            return True
        if np.any(new_pos < 0.0):
            return True
        if np.any(new_pos > 5.0):
            return True
        return False


    def render(self, object_positions):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=float)
        for idx, pos in enumerate(object_positions):
            print(idx, pos)
            if idx % 3 == 0:
                rr, cc = skimage.draw.circle(
                    pos[0] * 10 + 5, pos[1] * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]
            elif idx % 3 == 1:
                rr, cc = triangle(pos[0] * 10, pos[1] * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]
            else:
                rr, cc = square(pos[0] * 10, pos[1] * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]
        # return im.transpose([2, 0, 1])
        return im

    def set_positions(self, position_probs):
        """ Generate positions of the objects in the grid."""
        positions = product(range(self.height), range(self.width))
        positions = np.array([i for i in positions])
        position_idx = random.choice(len(positions), self.num_objects + 1,
                                     replace=False, p=position_probs.flatten())
        object_positions = positions[position_idx]
        return object_positions

if __name__ == "__main__":
    generator = SpaceShapesGenerator()
    generator.generate_data(1)
