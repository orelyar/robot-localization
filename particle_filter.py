from Robot import Robot
from World import World
import numpy as np
from matplotlib import pyplot as plt
from random import random
from Ploter import config_plot
from tqdm import tqdm
import timeit

"""
this python file runs particle filter
"""

CONST_N = 1000  # number of particles in each step

show = False  # plotting or not (0/1)
forward_noise = 0
turn_noise = 0
sense_noise_range = 0
sense_noise_bearing = 0
init_pose = (0, 0, 0)

"""
        :param forward_noise: the noise for moving forward
        :param turn_noise: the noise in the turn of the robot
        :param sense_noise_range: the noise in range measurement
        :param sense_noise_bearing: the noise in bearing measurement
        :param init_pose: the initial position of the robot

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
number_of_iterations = 40


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
    particle_filter()  # here is the magic

    if printing:
        plt.show()

    errors = calc_error()  # calculate the error estimator for the run
    mad_dist = sum(errors)/len(move_history)
    errors_dist.append(mad_dist)


def initialize_parameters(noise_array, start_position, motor_commands, printing):
    """
    initialize the global parameters as given
    :param noise_array: given noise parameters
    :param start_position: initial position
    :param motor_commands: given motor commands
    :param printing: with or without plotting? (0/1)
    """
    global forward_noise, turn_noise, sense_noise_range, sense_noise_bearing, init_pose, move_history, show
    forward_noise, turn_noise, sense_noise_range, sense_noise_bearing = noise_array

    if printing:
        show = True

    init_pose = start_position
    move_history = motor_commands

    global pose_history, sense_history, pose_expected, estimation_history
    pose_history = [init_pose]  # position history
    sense_history = []  # sense history
    pose_expected = [init_pose]  # expected pose history
    estimation_history = [init_pose]  # estimated positions history


def create_generic_robot():
    """
    create a robot in the initial position
    :return: the robot created
    """
    generic_robot = Robot(myWorld, forward_noise, turn_noise, sense_noise_range, sense_noise_bearing)
    generic_robot.set(init_pose[0], init_pose[1], init_pose[2])  # setup
    return generic_robot


def back_to_starting_position(robot):
    """
    return robot back into the initial position
    """
    robot.set(init_pose[0], init_pose[1], init_pose[2])  # setup


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


def particle_filter():
    """
    run particle filter algorithm
    """
    particles = sample_init()  # set up

    for i in range(len(move_history)):  # for each command
        particles = step_particle_filter(particles, move_history[i], sense_history[i])  # do particle filter


def normal_probability(miu, sigma, x):
    """
    return prob. according to normal dist.
    it's basically a math function
    """
    return np.exp(-1 * ((miu - x) ** 2) / (sigma ** 2) / 2) / np.sqrt(2 * pi * (sigma ** 2))


def measurement_probability(self, measurement):  #
    """
    added method for the Robot class
    calculate the probability for the robot to be in its position
    :param measurement: measurement from the sense() function
    :return: probability
    """
    p = 1  # setup
    for index in range(len(landmarks)):  # for each landmark
        dx = landmarks[index][0] - self.x
        dy = landmarks[index][1] - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)  # calc. distance
        rot = (np.arctan2(dy, dx) - self.theta + 2 * pi + np.random.normal(0, 0.3)) % (2 * pi)  # calc rot.
        p = p * normal_probability(distance, sense_noise_range, measurement[index][1])  # prob. distance
        p = p * normal_probability(rot, sense_noise_bearing, measurement[index][0])  # prob. rot

    return p


Robot.measurement_probability = measurement_probability  # add to the Robot class


def plot_pose(ax=None):
    """
    plot real path and path as expected
    :param ax: axis given
    :return: figure

    plot:
    pose_history: real path (will be plotted as a solid red line)
    pose_expected: path as expected (will be plotted as a dotted black line)
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

    return fig  # return plot


def sample_init():
    """
    sample CONST_N particles at the init. pose
    :return: vector of robots, all in the same position
    """
    particles_vector = []
    for i in range(CONST_N):
        robot = Robot(myWorld, forward_noise, turn_noise, sense_noise_range, sense_noise_bearing)  # make it
        robot.set(init_pose[0], init_pose[1], init_pose[2])  # set it
        particles_vector.append(robot)  # save it
    return particles_vector


def sample_move(particles_vector, step_given):
    """
    move the sample according to motor command (with noise)
    :param particles_vector: vector of robots
    :param step_given: motor commend
    """
    for robot in particles_vector:  # for each
        robot.move(step_given[0], step_given[1])


def weights(particles_vector, measurement):
    """
    calculate weights
    :param particles_vector: vector of robots
    :param measurement: measurement from the sense() function
    :return: weights array
    """
    w_temp = []
    w_final = []

    for robot in particles_vector:  # for each
        w_temp.append(robot.measurement_probability(measurement))  # calc prob.
    w_sum = sum(w_temp)  # sum it up
    for w in w_temp:
        temp = w / w_sum  # normalize
        w_final.append(temp)
    return w_final


def resample(particles_vector, w_array):
    """
    resample the robots according to probability
    :param particles_vector: the particles
    :param w_array: wights vector
    :return: array of the resampled robots
    """
    new_particles = []
    for i in range(CONST_N):  # for each
        r = random()
        sum_w = 0
        for i in range(CONST_N):
            sum_w += w_array[i]
            if r < sum_w:  # choose in random
                new_particles.append(particles_vector[i].clone())
                break
    return new_particles


def estimate(particles_re):
    """
    estimate pose according to resampled robots
    :param particles_re: resampled robots
    :return: approximated position
    """
    x_val = []
    y_val = []
    theta_val = []

    for robot in particles_re:
        x_val.append(robot.x)
        y_val.append(robot.y)
        theta_val.append(robot.theta)

    return np.average(x_val), np.average(y_val), np.average(theta_val)  # return average


def print_sample(particles_vector, particles_re):
    """
    print both sampled and resampled robots
    :param particles_vector: sampled robots
    :param particles_re: resampled robots
    :return:
    """
    global estimation_history
    est = estimate(particles_re)
    estimation_history.append(est)
    if show:
        plot_pose()  # plot both paths
        plt.scatter(est[0], est[1], zorder=3, color='m')  # plot est. pose
        for robot in particles_vector:
            robot.plot(mycolor='black', style="particle", show=0)  # plot sampled
        for robot in particles_re:
            robot.plot(mycolor='cyan', style="particle", show=0)  # plot resampled


def step_particle_filter(particles_vector, move_step, sense_step):
    """
    do one step of the particle filter
    :param particles_vector: vector of robots
    :param move_step: motor command
    :param sense_step: output from the sense() function
    :return: vector of the resampled robots
    """
    sample_move(particles_vector, move_step)  # move
    w_array = weights(particles_vector, sense_step)  # calc. weights
    particles_resampled = resample(particles_vector, w_array)  # resample
    print_sample(particles_vector, particles_resampled)  # plot
    return particles_resampled  # return


def calc_error():
    """
    calculate the error estimator (euclidean distance)
    :return: list of error estimators in this run, one for each measure
    """
    distance = []
    for i in range(len(estimation_history)):
        dx = pose_history[i][0] - estimation_history[i][0]
        dy = pose_history[i][1] - estimation_history[i][1]
        distance.append(np.sqrt(dx ** 2 + dy ** 2))

    return distance


if __name__ == '__main__':
    start = timeit.default_timer()
    commends = [(0, 60), (pi / 3, 30), (pi / 4, 30), (pi / 4, 20), (pi / 4, 40)]
    for i in tqdm(range(number_of_iterations)):
        run([6, 0.1, 5, 0.3], [10, 15, 0], commends, printing=0)

    end = timeit.default_timer()
    print(f"time took: {end - start}, time for each iteration: {(end - start)/number_of_iterations}")
    print(f"distance error:\naverage: {np.average(errors_dist)}, median: {np.median(errors_dist)}")
    plt.hist(errors_dist)
    plt.show()
