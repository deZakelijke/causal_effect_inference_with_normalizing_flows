import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage

from itertools import product, cycle
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
        colors.append((cm(0.64 * i / num_colors)))

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

    def __init__(
        self,
        width=6,
        height=6,
        n_objects=5,
        prior_dims=64,
        seed=None,
        **_
    ):
        self.width = width
        self.height = height
        self.n_objects = n_objects
        self.prior_dims = prior_dims
        self.num_colors = min(4, self.n_objects)
        self.colors = get_colors(num_colors=self.num_colors)
        self.save_path = "datasets/SPACE/"

    def load_priors(self, path="datasets/SPACE/", file_prefix=''):
        self.save_path = path
        path = f"{self.save_path}{file_prefix}space_data"
        with h5py.File(f"{path}_priors.hdf5", "r") as f:
            self.intervention_map = np.array(f['intervention_map'])
            self.position_map = np.array(f['position_map'])
            self.object_gravity = np.array(f['object_gravity'])
            remaining_data = np.array(f['remaining_data'])

        self.width = remaining_data[0]
        self.height = remaining_data[1]
        self.n_objects = remaining_data[2]
        self.prior_dims = remaining_data[3]
        self.num_colors = remaining_data[4]
        self.colors = get_colors(num_colors=self.num_colors)

        self.scale = 10.
        self.goal = np.array([[(self.height - 1) / 2, (self.width - 1)]])
        print(f"Gravities: {self.object_gravity[0]}")

    def set_priors(self, color_scale_factor=2, shape_scale_factor=2,
                   save=False, file_prefix=''):
        """
        It has to be more clearly defined if n_objects does or does not
        include the space ship itself.

        As the hyperparameter it does seem to include it?
        """
        path = f"{self.save_path}{file_prefix}space_data"
        self.intervention_map = random.standard_normal((self.prior_dims +
                                                        self.n_objects, 2)) * .5
        self.position_map = random.standard_normal((self.prior_dims +
                                                    self.n_objects,
                                                    self.width * self.height))
        self.scale = 10.

        color_weights = cycle(np.linspace(1., color_scale_factor, 8))
        shape_weights = cycle(np.linspace(1., shape_scale_factor, 3))
        obj_factor = []
        for _, c, s in zip(range(self.n_objects), color_weights, shape_weights):
            obj_factor.append(c * s)

        self.object_gravity = random.standard_normal((1, self.n_objects)) *\
            np.array(obj_factor)

        if save:
            remaining_data = np.array([self.width, self.height, self.n_objects,
                                       self.prior_dims, self.num_colors])
            with h5py.File(f"{path}_priors.hdf5", "w") as f:
                f.create_dataset("intervention_map", data=self.intervention_map)
                f.create_dataset("position_map", data=self.position_map)
                f.create_dataset("object_gravity", data=self.object_gravity)
                f.create_dataset("remaining_data", data=remaining_data)

        self.goal = np.array([[(self.height - 1) / 2, (self.width - 1)]])
        print(f"Gravities: {self.object_gravity[0]}")

    def generate_data(self, n_obj_train=5, n_samples=100, render=False,
                      save=False, file_prefix='', size_noise_flag=False):
        path = f"{self.save_path}{file_prefix}space_data"

        obj_indices = np.concatenate(([0], np.random.choice(
            range(1, self.n_objects), n_obj_train - 1,
            replace=False)))

        prior_data = random.standard_normal((n_samples, self.prior_dims))
        object_gravity = np.tile(self.object_gravity[0, obj_indices],
                                 (n_samples, 1))
        prior_data = np.concatenate((prior_data, object_gravity), axis=1)

        map_indices = np.concatenate((range(self.prior_dims),
                                      [i + self.prior_dims for i in obj_indices]))
        position_map = self.position_map[map_indices]

        position_probs = softmax(prior_data @ position_map, axis=1)
        position_probs = np.reshape(position_probs, (n_samples, self.height,
                                                     self.width))
        object_positions = self.set_positions(position_probs, obj_indices)
        # Iterate over number of samples
        rendering = np.array([self.render(objects, obj_indices, size_noise_flag) for objects in
                              object_positions])

        if save:
            with h5py.File(f"{path}_x.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_x", data=rendering)

        steering = prior_data @ self.intervention_map[map_indices]
        if save:
            with h5py.File(f"{path}_t.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_t", data=steering)

        move_result = self.move_spaceship(object_positions.copy(),
                                          object_gravity, steering,
                                          obj_indices)
        tmp = move_result[1] == 0
        dist_to_goal = np.linalg.norm(move_result[0][:, 0] - self.goal, axis=1)
        score = self.calc_score(dist_to_goal)

        if np.any(score < 0) or np.any(score > self.scale):
            print(score[np.where(score < 0)])
            print(score[np.where(score > self.scale)])
            raise ValueError("Some scores are out of bounds")

        if render:
            plt.subplot(121)
            plt.imshow(rendering[0])
            plt.title(f"Steering: {np.around(steering[0], 2)}\n"
                      f"Score: {score[0]:.2f}")
        print()

        if save:
            with h5py.File(f"{path}_y.hdf5", "w") as f:
                dset = f.create_dataset("Space_dataset_y", data=score)

        if render:
            rendering = np.array([self.render(objects, obj_indices, size_noise_flag) for objects
                                  in move_result[0]])
            plt.subplot(122)
            plt.imshow(rendering[0])
            plt.title(f"Movement: {move_result[1][0]}")
            plt.show()

        # Okay so here we need to sample two set of vectors for t
        # One is a standard normal and the other one is going to the right
        # We then calculate the y values and we save them.
        steering_0 = random.standard_normal(steering.shape)
        steering_1 = random.normal(loc=(0, 2), size=steering.shape)

        move_result_0 = self.move_spaceship(object_positions.copy(),
                                            object_gravity, steering_0,
                                            obj_indices)
        move_result_1 = self.move_spaceship(object_positions.copy(),
                                            object_gravity, steering_1,
                                            obj_indices)
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
        score_predict = np.concatenate((score_0, score_1), axis=1)

        if save:
            with h5py.File(f"{path}_t_predict.hdf5", "w")\
                    as f:
                dset = f.create_dataset("Space_dataset_t_predict",
                                        data=steering_predict)
            with h5py.File(f"{path}_y_predict.hdf5", "w")\
                    as f:
                dset = f.create_dataset("Space_dataset_y_predict",
                                        data=score_predict)

    def move_spaceship(self, object_positions, object_gravity, steering,
                       obj_indices):
        """
        """
        distances = object_positions[:, 1:] -\
            np.expand_dims(object_positions[:, 0], 1)
        weighted_distances = distances * np.expand_dims(object_gravity[:, 1:], -1)
        weighted_distances = weighted_distances
        direction = steering + np.sum(weighted_distances, axis=1) / 3
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

    def render(self, object_positions, object_indices, size_noise_flag=False):
        """ Render scence for particular state

        Create an image as a numpy array and paint the specific shapes at
        specific locations. Only the objects that are indicated are
        draw. The resolution is ten by ten pixels for each position in the
        2D grid.

        Parameters
        ----------
        self
            The generator object

        object_positions : numpy array
            The positions of all objects as 2D coordinates

        indices_to_render : numpy array
            Array of indices of objects that are present in the scene.

        Returns
        -------
        im : numpy array
            Float array of (10*widh, 10*height, 3) of pixel values.
        """
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=float)
        if size_noise_flag:
            size_noise = np.round(np.random.normal(scale=1.5, size=len(object_positions)))
            size_noise[size_noise < -2.] = -2.
        else:
            size_noise = np.zeros((len(object_positions)))

        for idx, pos in enumerate(object_positions):
            if idx not in object_indices:
                continue
            # print(idx)
            if idx % 3 == 0:
                rr, cc = skimage.draw.circle(
                    pos[0] * 10 + 5, pos[1] * 10 + 5,
                    5 + size_noise[idx], im.shape)
                im[rr, cc, :] = self.colors[idx % self.num_colors][:3]
            elif idx % 3 == 1:
                rr, cc = triangle(pos[0] * 10, pos[1] * 10,
                                  10 + size_noise[idx], im.shape)
                im[rr, cc, :] = self.colors[idx % self.num_colors][:3]
            else:
                rr, cc = square(pos[0] * 10, pos[1] * 10,
                                10 + size_noise[idx], im.shape)
                im[rr, cc, :] = self.colors[idx % self.num_colors][:3]
        return im

    def set_positions(self, position_probs, obj_indices):
        """ Generate positions of the objects in the grid.
        
        The position of an object is indepentent of its colour and
        shape, given the priors and position probabilities.

        """
        positions = product(range(self.height), range(self.width))
        positions = np.array([i for i in positions])
        position_idx = []
        for prob in position_probs:
            position_idx.append(random.choice(len(positions),
                                              len(obj_indices),
                                              replace=False,
                                              p=prob.flatten()))
        position_idx = np.array(position_idx)
        object_positions = positions[position_idx]
        return object_positions

    def calc_score(self, dist_to_goal):
        score = self.scale * (1 - dist_to_goal /
                              np.sqrt((self.height - 1) ** 2 +
                                      (self.width - 1) ** 2))
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space shapes generator")
    parser.add_argument("--height", type=int, default=6,
                        help="Height of the grid")
    parser.add_argument("--n_objects", type=int, default=8,
                        help="Total number of objects in the grid in both"
                        " training samples and test samples")
    parser.add_argument("--n_obj_train", type=int, default=6,
                        help="Number of objects in training samples")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Render and display the first sample")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save generated data to file")
    parser.add_argument("--width", type=int, default=6,
                        help="Width of the grid")
    args = parser.parse_args()

    generator = SpaceShapesGenerator(width=args.width, height=args.height,
                                     n_objects=args.n_objects)
    generator.load_priors()

    # generator.set_priors(save=args.save)

    generator.generate_data(n_obj_train=args.n_obj_train,
                            render=args.render,
                            n_samples=args.n_samples,
                            save=args.save,
                            file_prefix="fixed_gravity_")
