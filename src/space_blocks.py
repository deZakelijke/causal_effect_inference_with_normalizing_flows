import h5py
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
    """Class to generate dataset for space objects task.

    The class generates three files, the original state, visualised as images
    with some noise, the intervenion variable, which is a 2D 'steering'
    vector, and the outcome variable. For the outcome variable it is possible
    to represent it as an image as well but for now we pick the distance to
    the middle right of the image, inverted and normalised. This means that
    the score is 1 when the 'space ship' is in the middle right position
    after the action and has a score of 0 when it is widht * height away from
    that position. Of course that position is outside of the frame so no
    instace will actually have a score of zero.
    """

    def __init__(self, width=6, height=6, num_objects=5, prior_dims=64,
                 no_gravity=False, seed=None):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.prior_dims = prior_dims
        self.colors = get_colors(num_colors=max(9, self.num_objects))

        self.intervention_map = random.standard_normal((prior_dims +
                                                        num_objects, 2)) * 0.5
        self.position_map = random.standard_normal((prior_dims + num_objects,
                                                    width * height))
        self.scale = 10.
        if no_gravity:
            self.object_gravity = np.zeros((1, num_objects))
            self.save_path = "datasets/SPACE_NO_GRAV/"
        else:
            self.object_gravity = random.standard_normal((1, num_objects))
            self.save_path = "datasets/SPACE/"
        self.goal = np.array([[(height - 1) / 2, (width - 1)]])
        print(f"Gravities: {self.object_gravity[0]}")

    def generate_data(self, nr_samples, render=False, save=False):
        prior_data = random.standard_normal((nr_samples, self.prior_dims))
        object_gravity = np.tile(self.object_gravity, (nr_samples, 1))
        prior_data = np.concatenate((prior_data, object_gravity), axis=1)

        position_probs = softmax(prior_data @ self.position_map, axis=1)
        position_probs = np.reshape(position_probs, (nr_samples, self.height,
                                                     self.width))
        object_positions = self.set_positions(position_probs)
        rendering = np.array([self.render(objects) for objects in
                              object_positions])

        if save:
            with h5py.File(f"{self.save_path}space_data_x.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_x", data=rendering)

        steering = prior_data @ self.intervention_map
        if save:
            with h5py.File(f"{self.save_path}space_data_t.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_t", data=steering)

        move_result = self.move_spaceship(object_positions.copy(),
                                          object_gravity, steering)

        dist_to_goal = np.linalg.norm(move_result[0][:, 0] - self.goal, axis=1)
        score = self.calc_score(dist_to_goal)

        if np.any(score < 0) or np.any(score > self.scale):
            print(score[np.where(score < 0)])
            print(score[np.where(score > self.scale)])
            raise ValueError("Some scores are out of bounds")

        if render:
            plt.subplot(121)
            plt.imshow(rendering[0])
            plt.title(f"Steering: {np.around(steering[0], 2)}\nScore: {score[0]:.2f}")
        print()

        if save:
            with h5py.File(f"{self.save_path}space_data_y.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_y", data=score)

        if render:
            rendering = np.array([self.render(objects) for objects in
                                  move_result[0]])
            plt.subplot(122)
            plt.imshow(rendering[0])
            plt.title(f"Movement: {move_result[1][0]}")
            plt.show()

        # Okay so here we need to sample two set of vectors for t
        # One is a standard normal and the other one is going to the right
        # We then calculate the y values and we save them.
        steering_0 = random.standard_normal(steering.shape)
        steering_1 = random.normal(loc=(0, 2), size=steering.shape)
        print(steering_1)

        move_result_0 = self.move_spaceship(object_positions.copy(),
                                            object_gravity, steering_0)
        move_result_1 = self.move_spaceship(object_positions.copy(),
                                            object_gravity, steering_1)
        dist_to_goal_0 = np.linalg.norm(move_result_0[0][:, 0] - self.goal,
                                        axis=1)
        dist_to_goal_1 = np.linalg.norm(move_result_1[0][:, 0] - self.goal,
                                        axis=1)
        score_0 = self.calc_score(dist_to_goal_0)
        score_1 = self.calc_score(dist_to_goal_1)

        steering_0 = np.reshape(steering_0, (len(steering_0), 1, 2))
        steering_1 = np.reshape(steering_1, (len(steering_1), 1, 2))
        steering_predict = np.concatenate((steering_0, steering_1), axis=1)
        score_0 = np.reshape(score_0, (len(score_0), 1))
        score_1 = np.reshape(score_1, (len(score_1), 1))
        print(np.mean(score_0), np.mean(score_1))
        print(np.mean(score_0) - np.mean(score_1))
        score_predict = np.concatenate((score_0, score_1), axis=1)

        if save:
            with h5py.File(f"{self.save_path}space_data_t_predict.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_t_predict",
                                        data=steering_predict)
            with h5py.File(f"{self.save_path}space_data_y_predict.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_y_predict",
                                        data=score_predict)

    def move_spaceship(self, object_positions, object_gravity, steering):
        """
        """
        distances = object_positions[:, 1:] -\
            np.expand_dims(object_positions[:, 0], 1)
        weighted_distances = distances * np.expand_dims(object_gravity, -1)
        direction = (steering + np.sum(weighted_distances, axis=1)) / 2
        new_pos = np.around(object_positions[:, 0] + direction)
        new_dist = np.linalg.norm(object_positions[:, 1:] -
                                  np.expand_dims(new_pos, 1), axis=-1)
        # Round this to integers
        valid = self.valid_new_pos(object_positions, new_pos, new_dist)
        while np.any(valid):
            direction *= np.expand_dims(0.5 * valid + 1.0 * ~valid, 1)
            new_pos = np.around(object_positions[:, 0] + direction)
            new_dist = np.linalg.norm(object_positions[:, 1:] -
                                      np.expand_dims(new_pos, 1), axis=-1)
            valid = self.valid_new_pos(object_positions, new_pos, new_dist)
        object_positions[:, 0] = new_pos
        direction = np.around(direction)
        return object_positions, direction

    def valid_new_pos(self, object_positions, new_pos, new_dist):
        a = np.any(new_dist < 1, axis=1)
        b = np.any(new_pos < 0.0, axis=1)
        c = np.any(new_pos > 5.0, axis=1)
        return a | b | c

    def render(self, object_positions):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=float)
        for idx, pos in enumerate(object_positions):
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
        position_idx = []
        for prob in position_probs:
            position_idx.append(random.choice(len(positions),
                                              self.num_objects + 1,
                                              replace=False,
                                              p=prob.flatten()))
        position_idx = np.array(position_idx)
        object_positions = positions[position_idx]
        return object_positions

    def calc_score(self, dist_to_goal):
        score = self.scale * (1 - dist_to_goal /
                np.sqrt((self.height - 1) ** 2 + (self.width - 1) ** 2))
        return score


if __name__ == "__main__":
    generator = SpaceShapesGenerator(width=6, height=6, num_objects=5,
                                     no_gravity=False)
    generator.generate_data(50000, render=False, save=True)
