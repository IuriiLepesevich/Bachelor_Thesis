import torch
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()

target_point = np.array([-2.0, 0])
anchor_point = np.array([0, 0])

is_running = True
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app

fig, _ = plt.subplots()
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(0)
theta_2 = np.deg2rad(10)
theta_3 = np.deg2rad(0)

test_vect1 = anchor_point + np.array([0, 1]) * length_joint

der = 0
loss = 0

def rotation(theta):
    R = np.identity(2)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, -s],
        [s, c]
    ])
    return R

def d_rotation1(theta):
    h = 0.0001
    d_R = (rotation(theta + h) - rotation(theta)) / h
    return d_R

def d_rotation2(theta):
    R = np.identity(2)
    c = np.cos(theta)
    s = np.sin(theta)
    d_R = np.array([
        [-s, -c],
        [c, -s]
    ])
    return d_R

while is_running:
    plt.clf()
    plt.title(f'loss: {loss} theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))} theta_3: {round(np.rad2deg(theta_3))}')

    joints = []
    joints.append(anchor_point)

    R1 = rotation(theta_1)
    R2 = rotation(theta_2)
    R3 = rotation(theta_3)

    R_test = R1 + R2

    vec = np.dot(R1, test_vect1)
    joints.append(vec)

    vec = np.dot(R1, test_vect1 + np.dot(R2, test_vect1))
    joints.append(vec)

    vec = np.dot(R1, test_vect1 + np.dot(R2, test_vect1 + np.dot(R3, test_vect1)))
    joints.append(vec)

    loss = np.sum((target_point - vec) ** 2)

    der1 = np.sum(-2 * (target_point - vec) * np.dot(d_rotation1(theta_1), test_vect1))

    der2 = np.sum(-2 * (target_point - vec) * np.dot(R1, np.dot(d_rotation1(theta_2), test_vect1)))

    der3 = np.sum(-2 * (target_point - vec) * np.dot(R2, np.dot(R1, np.dot(d_rotation1(theta_3), test_vect1))))

    alpha = 0.01
    theta_1 -= alpha * der1
    theta_2 -= alpha * der2
    theta_3 -= alpha * der3


    np_joints = np.array(joints)


    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
    #break
#input('end')