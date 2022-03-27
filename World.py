"""
in this module we are declaring the world class
"""
from matplotlib import pyplot as plt
from Ploter import config_plot


class World:
    """
    a class used to describe the world, actually the map of the world
    """
    def __init__(self, world_size=100, landmarks=[(20.0, 20.0),(50.0, 20.0), (80.0, 20.0), (80.0, 80.0), (50.0, 80.0),(20.0, 80.0)]):
        """
        constructor for the map
        :param world_size: the size of the world in pixels
        """
        self.world_size = world_size
        # the locations of the 4 landmarks in our problem
        self.landmarks = landmarks

    def plot(self, show=False):
        """
        plotting the map with sizes
        :param show: indicator - to show the map or not
        """
        fig = plt.figure()
        plt.rcParams.update({'font.size': 16})
        plt.plot([x[0] for x in self.landmarks], [x[1] for x in self.landmarks], "ko")
        config_plot(plt, self.world_size)
        ax = fig.gca()
        ax.set_aspect("equal")

        if show:
             plt.show()

    def get_world_size(self):
        """
        returns the world size
        :return: the world size
        """
        return self.world_size

    def get_landmarks(self):
        """
        return the landmarks of the world
        :return:  the landmarks - a list of tuples
        """
        return self.landmarks
