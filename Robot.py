import numpy as np
from matplotlib import pyplot as plt
from World import World
from scipy.stats import norm
from Ploter import config_plot


class Robot:
    """
    the robot class, we will use this to describe a robot
    """

    def __init__(self, world=World(), forward_noise=6, turn_noise=0.1, sense_noise_range=5, sense_noise_bearing=0.3):
        """
        constractor of the Robot class
        :param world: the world in which the robot is
        :param forward_noise: the noise for moving forward
        :param turn_noise: the noise in the turn of the robot
        :param sense_noise_range: the noise in range measurement
        :param sense_noise_bearing: the noise in bearing measurement
        """
        self.world = world
        self._world_size = world.get_world_size()
        # pose declaration
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise_range = sense_noise_range
        self.sense_noise_bearing = sense_noise_bearing
        self.x = self._world_size/2
        self.y = self._world_size/2
        self.theta = 0

    def set(self, new_x, new_y, new_orientation):
        """
        setting the configuration of the robot
        :param new_x: the new location coordinate
        :param new_y: the new y coordinate
        :param new_orientation: the new orientation
        """
        if new_x < 0 or new_x >= self._world_size:
            raise Exception('X coordinate out of bound')

        if new_y < 0 or new_y >= self._world_size:
            raise Exception('Y coordinate out of bound')

        if new_orientation < 0.0 or new_orientation >= 2 * np.pi:
            Exception('Orientation must be in [0,2pi]')

        self.x = new_x
        self.y = new_y
        self.theta = new_orientation

    def set_noise(self, forward_noise=6, turn_noise=0.1, sense_noise_range=5,
                  sense_noise_bearing=0.3):  ##add param exp.
        """
        setting the noise if pose of the robot
        :param forward_noise: the noise for moving forward
        :param turn_noise: the noise in the turn of the robot
        :param sense_noise_range: the noise in range measurement
        :param sense_noise_bearing: the noise in bearing measurement
        """
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise_range = sense_noise_range
        self.sense_noise_bearing = sense_noise_bearing

    def print(self):
        """"
        printing the pose
        """
        print('[location= {} y={} heading={}]'.format(self.x, self.y, self.theta))

    def print_noise(self):
        """
        printing the noise parameters
        """
        print("forward_noise = " + str(self.forward_noise))
        print("turn_noise = " + str(self.turn_noise))
        print("sense_noise_range = " + str(self.sense_noise_range))
        print("sense_noise_bearing = " + str(self.sense_noise_bearing))

    def plot(self, mycolor="b", style="robot", show=True, markersize=1):
        """
        plotting the pose of the robot in the world
        :param mycolor: the color of the robot
        :param style: the style to plot with
        :param show: if to show or not show - used to create a new figure or not
        """
        if style == "robot":
            phi = np.linspace(0, 2 * np.pi, 101)
            r = 1
            # plot robot body
            plt.plot(self.x + r * np.cos(phi), self.y + r * np.sin(phi), color=mycolor)
            # plot heading direction
            plt.plot([self.x, self.x + r * np.cos(self.theta)], [self.y, self.y + r * np.sin(self.theta)],
                     color=mycolor)

        elif style == "particle":
            plt.plot(self.x, self.y, '.', color=mycolor, markersize=markersize)
        else:
            print("unknown style")

        if show:
            plt.show()

    def get_pose(self):
        """
        returning the pose vector
        :return: (location, y, theta) the pose vector
        """
        return self.x, self.y, self.theta

    def move(self, u1, u2, noise=True):
        """
        move according to motor commend, with or without noise

        :param u1: turn commend (radians)
        :param u2: distance commend
        :param noise: with or without noise (0/1)
        :return:
                with noise: (position), sense vector
                without noice: position
        """
        if noise:
            u1 = u1 + np.random.normal(0, self.turn_noise)  # add noise
            u2 = u2 + np.random.normal(0, self.forward_noise)  # add noise
        self.x = (self.x + u2 * np.cos(self.theta + u1)) % 100  # move, the world is cyclic
        self.y = (self.y + u2 * np.sin(self.theta + u1)) % 100
        self.theta = self.theta + u1
        if noise:
            return (self.x, self.y, self.theta), self.sense()  # add to position history
        return self.x, self.y, self.theta  # return position

    def get_location(self, sense_vector):
        """
        extracting approximated location from a sense vector
        :param sense_vector: output from sense() function
        :return: approximated position
        """
        landmark_array = self.world.get_landmarks()
        n = len(landmark_array)
        if len(sense_vector) != n:
            raise Exception('length of sense vector != number of landmarks')

        x_values = []
        y_values = []
        theta_values = []
        for index in range(n):
            landmark = landmark_array[index]
            measurement = sense_vector[index]
            x_values.append(landmark[0] - measurement[1] * np.cos(measurement[0] + self.theta))
            y_values.append(landmark[1] - measurement[1] * np.sin(measurement[0] + self.theta))



        return x_values, y_values

    def sense(self, noise=True):
        """
        sensing the landmarks around
        :param noise: with or without noise (for debugging purposes)
        :return: vector of 6 motor commends to get to each landmark
        """
        res = []
        noise_distance = 0
        noise_rot = 0
        for LM in self.world.get_landmarks():
            if(noise):
                noise_distance = np.random.normal(0, self.sense_noise_range)
                noise_rot = np.random.normal(0,self.sense_noise_bearing)
            dx = LM[0] - self.x  # calc. distance
            dy = LM[1] - self.y
            distance = np.sqrt(dx ** 2 + dy ** 2) + noise_distance
            rot = (np.arctan2(dy, dx) - self.theta + 2 * np.pi + noise_rot) % (2 * np.pi)   # calc. rot
            res.append((rot, distance))
        return res

    def clone(self):
        """
        cloning the robot
        :return: cloned robot
        """
        new_robot = Robot(self.world, self.forward_noise, self.turn_noise, self.sense_noise_range,
                          self.sense_noise_bearing)
        new_robot.set(self.x % 100, self.y % 100, self.theta)  # clone
        return new_robot
