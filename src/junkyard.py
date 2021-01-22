"""
Created on Mon Jan 18 2021

@author: Alex Korthouwer
@author: Yu Anlan
"""
from __future__ import unicode_literals, print_function, absolute_import, division, generators, nested_scopes
import sys, os
import numpy as np
from deap import base, creator, tools, algorithms
import pandas as pd
import random
import robobo
import cv2
import sys
import signal
import prey
import time
import array
from controller import Controller
import math
from blob_detection import detect
import calendar
from datetime import datetime

parameters = []
with open('D:\Subjects\Learning Machines\weights_111_1611195794.csv', 'r') as f:
    for w in f.readlines():
        parameters.append(float(w))

print(parameters)
c = Controller(parameters, 7, 2, 2)


with open('IP.txt', 'r') as f:
    ip = f.readlines()[0]


rob = robobo.SimulationRobobo(number='').connect(address=ip, port=19997)
while (rob.is_simulation_running()):
    pass
rob.play_simulation()
rob.set_phone_tilt(0.8, 50)
start = rob.get_sim_time()
while (rob.get_sim_time() - start < 360 * 1000):
    sensors = rob.read_irs()[3:]  # front sensors, length 5
    sensors = [sensors[i] * 5 if sensors[i] is not False else 1 for i in range(len(sensors))]
    detection_x, detection_y = detect(rob.get_image_front())

    if detection_x is False:
        detection_x, detection_y = -1, -1
    else:
        detection_x, detection_y = detection_x / 128, detection_y / 128
    inputs = sensors + [detection_x, detection_y]
    x, y = c.act(inputs)
    rob.move(float(x), float(y), 500)
# fitness_score += rob.collected_food()

rob.stop_world()