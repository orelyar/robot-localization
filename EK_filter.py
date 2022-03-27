import math
import matplotlib.pyplot as plt
import numpy as np
from World import World
from Robot import Robot
from tqdm import tqdm
from Ploter import config_plot
import timeit

"""
this python file runs extended kalman filter
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
    miu_init = init_pose
    sig_init = np.zeros((3, 3))
    sig_init[0, 0], sig_init[1, 1], sig_init[2, 2] = 1, 1, 1
    EKF(motor_commands, miu_init, sig_init)  # here is the magic

    if printing:
        plot_pose()  # plot both paths
        plt.show()

    errors = calc_error()  # calculate the error estimator for the run
    mad_dist = sum(errors)/len(move_history)
    errors_dist.append(mad_dist)


def z_func(iteration, noise=True):
    """
    this function calculates and returns the z matrix, i.e. the robot's sense() function gaven true position
    shape:12x1
    :param iteration: number of iteration
    :param noise: with or without noise (for debugging purposes)
    :return: z matrix
    """
    generic_robot = Robot()
    location = pose_history[iteration + 1]
    generic_robot.set(location[0], location[1], location[2])
    measurement = np.matrix(generic_robot.sense(noise))
    measurement[:, [0, 1]] = measurement[:, [1, 0]]  # switch columns
    measurement = np.reshape(measurement, (12, 1), order='C')  # reshape to 12x1
    return measurement


def g_func(u, miu):
    """
     this function calculates and returns the g matrix, i.e. approximate pose given the last approximate position and
     the motor command. this matrix serves as the "belief"
     shape:3x1
    :param u: motor command
    :param miu: miu matrix
    :return: g matrix
    """
    g1 = miu[0] + math.cos(miu[2] + u[0]) * u[1]  # location
    g2 = miu[1] + math.sin(miu[2] + u[0]) * u[1]  # y
    g3 = miu[2] + u[0]  # theta
    return np.transpose(np.matrix([float(g1), float(g2), float(g3)]))


def h_func(miu):
    """
    this function calculates and returns the h matrix, i.e. the not noisy z matrix I would have gotten if the
    miu position in correct
    shape:12x1
    :param miu: approximate position
    :return: h matrix
    """
    x = miu[0]
    y = miu[1]
    h_matrix = []
    for LM in landmarks:
        dx = LM[0] - x
        dy = LM[1] - y
        h_matrix.append(float(np.sqrt(dx ** 2 + dy ** 2)))
        h_matrix.append(float((np.arctan2(dy, dx) - miu[2] + 2 * pi) % (2 * pi)))
    h_matrix = np.transpose(np.matrix(h_matrix))
    return h_matrix


def H_func(miu):  # H function
    """
    this function calculates and returns the H matrix, i.e. d(h)/d(X[t])
    shape:12x3
    :param miu: approximate position
    :return: H matrix
    """
    x = miu[0]
    y = miu[1]
    h1 = []
    h2 = []
    h3 = []
    for LM in landmarks:
        h1.append(float((x - LM[0]) / (math.sqrt((LM[0] - x) ** 2 + (LM[1] - y) ** 2))))
        h2.append(float((y - LM[1]) / (math.sqrt((LM[0] - x) ** 2 + (LM[1] - y) ** 2))))
        h3.append(0)
        h1.append(float((LM[1] - y) / ((LM[0] - x) ** 2 + (LM[1] - y) ** 2)))
        h2.append(float((x - LM[0]) / ((LM[0] - x) ** 2 + (LM[1] - y) ** 2)))
        h3.append(-1)
    return np.column_stack((h1, h2, h3))


def G_func(u, miu):
    """
    this function calculates and returns the G matrix, i.e. d(g)/d(X[t])
    shape:3x3
    :param u: motor command
    :param miu: approximate position
    :return: G function
    """
    th = miu[2]  # theta
    odt = u[0]  # omega
    v = u[1]
    g13 = 0 - v * math.sin(th + odt)
    g23 = v * math.cos(th + odt)
    return np.matrix([[1, 0, float(g13)], [0, 1, float(g23)], [0, 0, 1]])


def Q_func():  # return matrix Q
    """
    this function calculates and returns the Q matrix, i.e. the covariance matrix of the robot's sense() function
    shape:12x12
    :return: Q matrix
    """
    q_matrix = np.zeros((12, 12))
    for index in range(12):
        if index % 2 == 1:
            q_matrix[index, index] = sense_noise_bearing ** 2
        else:
            q_matrix[index, index] = sense_noise_range ** 2
    return q_matrix


def V_func(u, miu):
    """
    this function calculates and returns the V matrix, i.e. d(F)/d(u), where F is the dynamics
    used for obtaining the R function
    shape:3x2
    :param u: motor command
    :param miu: approximate position
    :return: V matrix
    """
    v = u[1]
    thdt = miu[2] + u[0]  # theta + omega
    v12 = math.cos(thdt)
    v22 = math.sin(thdt)
    v11 = math.sin(thdt) * (-1) * v
    v21 = math.cos(thdt) * v
    return np.matrix([[float(v11), float(v12)], [float(v21), float(v22)], [1, 0]])


def M_func():
    """
    this function calculates and returns the M matrix, i.e. the covariance matrix of the controls (meaning the
    motor command's noise)
    used for obtaining the R function
    shape:2x2
    :return:
    """
    m11 = sense_noise_range ** 2
    m22 = sense_noise_bearing ** 2
    return np.matrix([[m11, 0], [0, m22]])


def R_func(u, miu):  # return matrix R
    """
    this function calculates and returns the R matrix, i.e. linear approximation of the dynamics at the miu point
    obtained from the noise in control space by a linear approximation of the form Rt = VMV^T, where V = d(F)/d(u),
    F is the dynamics and M is the covariance matrix of the controls.
    shape:3x3
    :param u: motor command
    :param miu: approximate position
    :return: R matrix
    """
    M = M_func()
    V = V_func(u, miu)
    return np.dot(V, np.dot(M, np.transpose(V)))


def miu_bar_func(u, miu):
    """
    this function calculates and returns the miu_bar matrix, i.e. the "belief" matrix.
    shape:3x1
    :param u: motor command
    :param miu: approximate position
    :return: miu_bar matrix
    """
    return g_func(u, miu)


def sig_bar_func(u, miu, sig):  # return sigma bar
    """
    this function calculates and returns the sigma_bar matrix, i.e. the covariance matrix of the "belief" (miu_bar)
    shape:3x3
    :param u: motor command
    :param miu: approximate position
    :param sig: the covariance matrix of the last position
    :return: sigma_bar matrix
    """
    G = G_func(u, miu)
    R = R_func(u, miu)
    temp = np.dot(G, np.dot(sig, np.transpose(G)))
    return np.add(temp, R)


def K_func(sig_bar, H):  # return the kalman gain
    """
    this function calculates and returns the K matrix, i.e. the kalman gain
    shape:3x12
    :param sig_bar: the covariance matrix of the "belief"
    :param H: H matrix
    :return: K matrix
    """
    Q = Q_func()
    temp = np.dot(H, np.dot(sig_bar, np.transpose(H)))
    temp = np.add(temp, Q)
    temp = np.linalg.inv(temp)
    return np.dot(sig_bar, np.dot(np.transpose(H), temp))


def miu_func(z, miu_bar, K):
    """
    this function calculates and returns the miu matrix, i.e. the approximate position given the measurement and the belief
    shape:3x1
    :param z: z matrix (the measurement)
    :param miu_bar: miu_bar matrix (the belief)
    :param K: K matrix (kalman gain)
    :return: miu matrix
    """
    temp = np.subtract(z, h_func(miu_bar))
    temp = np.dot(K, temp)
    return np.add(miu_bar, temp)


def sig_func(K, H, sig_bar):
    """
    this function calculates and returns the sigma matrix, i.e. the covariance matrix of the approximate position
    shape:3x3
    :param K: K matrix (kalman gain)
    :param H: H matrix
    :param sig_bar: covariance matrix of the "belief"
    :return: sigma matrix
    """
    I = np.identity(3)
    temp = np.dot(K, H)
    temp = np.subtract(I, temp)
    return np.dot(temp, sig_bar)


def EKF_calc(u, miu, z, sig):
    """
    calculation of single a single EKF step
    :param u: motor commend
    :param miu: last approximate position
    :param z: z matrix (measurement)
    :param sig: the covariance matrix of the last position
    :return: new approximate position and its covariance matrix
    """
    H = H_func(miu)
    miu_bar = miu_bar_func(u, miu)  # form belief
    estimation_history.append((miu_bar[0, 0], miu_bar[1, 0], miu_bar[2, 0]))  # keep belief in history
    sig_bar = sig_bar_func(u, miu, sig)  # calculate the belief covariance matrix
    K = K_func(sig_bar, H)  # calculate near optimal kalman gain
    miu = miu_func(z, miu_bar, K)  # approximate position
    sig = sig_func(K, H, sig_bar)  # approximate its covariance matrix
    return [miu, sig]


def EKF_step(controls, index, miu, sig):
    """
    get measurements and controls and call EKF_calc
    :param controls: motor commend
    :param index: iteration number
    :param miu: last position
    :param sig: last covariance matrix
    :return: new approximate position and its covariance matrix
    """
    u = controls[index]  # extract motor command
    z = z_func(index)  # extract measurement
    return EKF_calc(u, miu, z, sig)  # do a single step


def EKF(controls, miu_init, sig_init):
    """
    run EKF algorithm
    :param controls: motor commends list
    :param miu_init: initial position
    :param sig_init: initial covariance matrix
    :return: list of the approximate positions and their covariance matrices
    """
    # init
    miu = miu_init
    sig = sig_init
    res_miu = [miu]
    res_sig = [sig]
    n = len(controls)  # number of iterations
    for index in range(n):
        res = EKF_step(controls, index, miu, sig)
        miu = res[0]
        sig = res[1]
        res_miu.append(miu)
        res_sig.append(sig)

    return [res_miu, res_sig]  # return results


"""
from here it's almost a straight copy-paste of functions from particle_filter.py. little changes have been made.
list of functions:
1) back_to_starting_position
2) create_generic_robot
3) initialize_parameters
4) generate_path
5) initialize_parameters
6) plot_pose
"""


def back_to_starting_position(robot):
    """
    return robot back into the initial position
    """
    robot.set(init_pose[0], init_pose[1], init_pose[2])  # setup


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
    move_history = motor_commands

    global pose_history, sense_history, pose_expected, estimation_history
    pose_history = [init_pose]  # position history
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


if __name__ == '__main__':
    start = timeit.default_timer()
    commends = [(0, 60), (pi / 3, 30), (pi / 4, 30), (pi / 4, 20), (pi / 4, 40)]
    for i in tqdm(range(number_of_iterations)):
        run([6, 0.1, 5, 0.3], [10, 15, 0], commends, printing=False)

    end = timeit.default_timer()
    print(f"time took: {end - start}, time for each iteration: {(end - start)/number_of_iterations}")
    print(f"error:\naverage: {np.average(errors_dist)}, median: {np.median(errors_dist)}")
    plt.hist(errors_dist)
    plt.show()
