import matplotlib.pyplot as plt
import numpy as np
from World import World
from Robot import Robot
from tqdm import tqdm
from Ploter import config_plot
from scipy.optimize import minimize
import timeit

"""
this python file runs a new filter I came up with.
the aim of this filter is robot's localization using as little computing power as possible.
it should be fast and easy to compute.

the algorithm tries to find the maximum likelihood estimator for the measurement using COBYLA optimization method
with the initial guess being the location of the closest landmark 
"""
show = False  # plotting or not (0/1)
forward_noise = 0
turn_noise = 0
sense_noise_range = 0
sense_noise_bearing = 0
init_pose = (0, 0, 0)

"""
        forward_noise: the noise for moving forward
        turn_noise: the noise in the turn of the robot
        sense_noise_range: the noise in range measurement
        sense_noise_bearing: the noise in bearing measurement
        init_pose: the initial position of the robot

        please note that all the parameters are about to change
"""

# data arrays
pose_history = [init_pose]  # position history
move_history = []  # move history
sense_history = []  # sense history
pose_expected = [init_pose]  # expected pose history
estimation_history = [init_pose]  # estimated positions history
errors_dist = []  # list of error estimators (sum of absolute deviation)

# make a world
myWorld = None
landmarks = []

# value of pi
pi = np.pi

# how many times to run the algorithm
number_of_iterations = 400
method = 'COBYLA'  # optimization method


def run(noise_array, start_position, motor_commands, world=World(), printing=False):
    """
    run the whole algorithm
    :param noise_array: given noise parameters
    :param start_position: initial position
    :param motor_commands: given motor commands
    :param world: the world in which the robot is
    :param printing: with or without plotting? (0/1)
    :return: MSE (mean squared error)
    """
    global myWorld, landmarks
    myWorld = world
    landmarks = world.get_landmarks()
    initialize_parameters(noise_array, start_position, motor_commands, printing)  # init everything
    generate_path()  # generate noisy path and expected path
    new_filter(motor_commands)  # here is the magic

    errors = calc_error()  # calculate the error estimator for the run
    mad_dist = sum(errors)/len(move_history)
    errors_dist.append(mad_dist)

    if printing:
        plot_pose()  # plot both paths
        plt.show()


def find_approximated_location(sense_vector):
    """
    create an initial guess for the robot's location.
    this guess is not a good one, it's just the closest landmark according to the measurement
    :param sense_vector: the measurement
    :return: the location f the closest landmark (initial guess)
    """
    distances = [item[1] for item in sense_vector]  # take only the distance measurement
    min_value = min(distances)
    min_index = distances.index(min_value)  # index of the closest landmark
    return landmarks[min_index]  # return the landmark


def new_filter(controls):
    """
    run the algorithm
    :param controls: motor commends
    """
    for index in range(len(controls)):  # for each commend
        loc = evaluate_location(index)  # run it
        estimation_history.append(loc)  # keep it


def evaluate_location(index):
    """
    find the location using COBYLA optimization method
    :param index:
    :return:
    """
    measurement = sense_history[index]
    temp_location = find_approximated_location(measurement)
    estimation = minimize(location_likelihood_function, temp_location, method=method,
                          options={'disp': show}, args=measurement,
                          tol=0.2)
    return estimation.x


def location_likelihood_function(location, measurement):
    """
    the likelihood function of a measurement given the true position
    this is the function we are trying to maximize
    :param location: true location
    :param measurement: output of the robot's sense() function
    :return: density function times a big negative number
    """
    density = -10000  # we want to maximize it, so we are using a negative number
    sigma = sense_noise_range  # the noise

    for index in range(len(landmarks)):  #for each landmark
        LM = landmarks[index]
        distance = np.sqrt((location[0] - LM[0]) ** 2 + (location[1] - LM[1]) ** 2)  # calculate distance
        miu = measurement[index][1]
        density *= 1000
        density *= np.exp(-1 * ((miu - distance) ** 2) / (sigma ** 2) / 2) / np.sqrt(
            2 * pi * (sigma ** 2))  # normal density * a big number, for sensitivity purposes
    return density



def calc_error():
    """
    calculate the error estimator (sum of absolute deviation)
    :return: list of error estimators, one for each run
    """
    distance = []
    for i in range(len(estimation_history)):
        dx = pose_history[i][0] - estimation_history[i][0]
        dy = pose_history[i][1] - estimation_history[i][1]
        distance.append(np.sqrt(dx ** 2 + dy ** 2))

    return distance


def create_generic_robot():
    """
    create a robot in the initial position
    :return: the robot created
    """
    generic_robot = Robot(myWorld, forward_noise, turn_noise, sense_noise_range, sense_noise_bearing)
    generic_robot.set(init_pose[0], init_pose[1], init_pose[2])  # setup
    return generic_robot


def generate_path():
    """
    generate noisy path and expected path
    add both to the data vectors
    """
    generic_robot = create_generic_robot()
    for step in move_history:
        pose_expected.append(generic_robot.move(step[0], step[1], noise=False))  # move with no noise
    back_to_starting_position(generic_robot)

    for step in move_history:
        temp_pose, temp_sense = generic_robot.move(step[0], step[1], noise=True)  # move with noise
        pose_history.append(temp_pose)
        sense_history.append(temp_sense)


def initialize_parameters(noise_array, start_position, motor_commands, printing):
    """
    initialize the global parameters as given
    :param noise_array: given noise parameters
    :param start_position: initial position
    :param motor_commands: given motor commands
    :param printing: with or without plotting? (0/1)
    """
    global forward_noise, turn_noise, sense_noise_range, sense_noise_bearing, init_pose, init_pose, move_history, show
    forward_noise, turn_noise, sense_noise_range, sense_noise_bearing = noise_array

    if printing:
        show = True

    init_pose = start_position
    global pose_history, sense_history, pose_expected, estimation_history, move_history
    pose_history = [init_pose]  # position history
    move_history = motor_commands
    sense_history = []  # sense history
    pose_expected = [init_pose]  # expected pose history
    estimation_history = [init_pose]  # estimated positions history


def plot_pose(ax=None):
    """
    plot real path and path as expected
    :param ax: axis given
    :return: figure

    plot:
    pose_history: real path (will be plotted as a solid red line)
    pose_expected: path as expected (will be plotted as a dotted black line)
    estimation_history: estimation of the robot's location (will be plotted as magenta dots)
    """
    if not show:
        return

    if ax is None:
        fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 16})  # plot world
    plt.plot([x[0] for x in landmarks], [x[1] for x in landmarks], "ko")
    config_plot(plt, 100)
    ax = fig.gca()
    ax.set_aspect("equal")
    for i in range(len(pose_history) - 1):  # plot real path
        plt.plot([pose_history[i][0], pose_history[i + 1][0]], [pose_history[i][1], pose_history[i + 1][1]], c='red')
    for i in range(len(pose_expected) - 1):  # plot expected path
        plt.plot([pose_expected[i][0], pose_expected[i + 1][0]], [pose_expected[i][1], pose_expected[i + 1][1]],
                 ls='dotted', c='black')
    for est in estimation_history:  # plot estimated paths
        plt.scatter(est[0], est[1], zorder=3, color='m')  # plot est. pose
    return fig  # return plot


def back_to_starting_position(robot):
    """
    return robot back into the initial position
    """
    robot.set(init_pose[0], init_pose[1], init_pose[2])  # setup


def dist(point1, point2):
    """
    finds the distance between two points
    :param point1, point2: points in the form of (location,y)
    :return: distance (float)
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


if __name__ == '__main__':
    start = timeit.default_timer()
    commends = [(0, 60), (pi / 3, 30), (pi / 4, 30), (pi / 4, 20), (pi / 4, 40)]
    for i in tqdm(range(number_of_iterations)):
        run([6, 0.1, 5, 0.3], [10, 15, 0], commends, printing=False)

    end = timeit.default_timer()
    print(f"time took: {end - start}, time for each iteration: {(end - start)/number_of_iterations}")
    print(f"distance error:\naverage: {np.average(errors_dist)}, median: {np.median(errors_dist)}")
    plt.hist(errors_dist)
    plt.show()
