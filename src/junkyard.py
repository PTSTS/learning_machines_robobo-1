#!/usr/bin/env python2
from __future__ import print_function

import time

import robobo
import cv2
import sys
import signal
import prey
# import deap
import math
# from deap import base, creator, tools, algorithms

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
#     import pickle
#     fit = pickle.load(open('fittest_222_1611757408.pkl', 'rb'))
#     print(fit)
#     pop = pickle.load(open('population_222_1611757408.pkl', 'rb'))
#     print(pop)
#     exit()

    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo(number='').connect(address='127.0.0.1', port=19997)
    rob.play_simulation()

    rob.set_phone_tilt(0.8, 50)
    image = rob.get_image_front()

    # mask = cv2.inRange(image, (0, 140, 0), (20, 255, 20))
    mask = cv2.inRange(image, (0, 0, 80), (60, 60, 255))
    # mask = cv2.bitwise_not(mask, mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('9', mask)
    cv2.imshow('1', image)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    
    cv2.imwrite('masked.png', masked)
    cv2.imwrite('test.png', image)
    exit()
    # prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.71', port=19989)
    # prey_controller = prey.Prey(robot=prey_robot, level=2, hardware=True)
    prey_robot = robobo.SimulationRoboboPrey('#0').connect(address='127.0.0.1', port=19989)
    prey_controller = prey.Prey(robot=prey_robot, level=2)

    prey_controller.start()
    for i in range(5000):
            print("robobo is at {}".format(rob.position()))
            rob.move(90, 90, millis=200)
    prey_controller.stop()
    prey_controller.join()
    prey_robot.disconnect()

    rob.stop_world()
    # move and talk
    # rob.set_emotion('sad')

    # initialise class prey
    # needs to receive the robot to move
    # there are 5 levels of difficulties. From 0 (super easy) to 4 (hard).
    # you can select the one you want,using a parameter in the following constructor, default is 2
    prey_controller = prey.Prey(robot=prey_robot, level=2)
    # start the thread prey
    # makes the prey move
    prey_controller.start()
    # print("start")

    for i in range(10):
            print("robobo is at {}".format(rob.position()))
            rob.move(5, 5, 2000)
    #
    # print("robobo is at {}".format(rob.position()))
    # rob.sleep(1)

    # # Following code moves the phone stand
    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    # time.sleep(1)
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    # rob.talk('Hi, my name is Robobo')
    # rob.sleep(1)
    # rob.set_emotion('happy')

    # Following code gets an image from the camera
    # image = rob.get_image_front()
    # cv2.imwrite("test_pictures.png",image)

    # time.sleep(0.1)

    # # IR reading
    # for i in range(1000000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    # stop the prey
    # if you want to stop the prey you have to use the two following commands
    prey_controller.stop()
    prey_controller.join()
    prey_robot.disconnect()


    # pause the simulation and read the collected food
    # rob.pause_simulation()
    # print("Robobo collected {} food".format(rob.collected_food()))

    # Stopping the simualtion resets the environment
    rob.stop_world()

    time.sleep(10)
    # rob.kill_connections()
    # rob = robobo.SimulationRobobo().connect(address='192.168.1.6', port=19997)
    rob.play_simulation()

    rob.set_phone_tilt(0.8, 50)
    # prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.71', port=19989)
    # prey_controller = prey.Prey(robot=prey_robot, level=2, hardware=True)
    prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.6', port=19989)
    prey_controller = prey.Prey(robot=prey_robot, level=2)

    prey_controller.start()
    for i in range(10):
            print("robobo is at {}".format(rob.position()))
            rob.move(90, 90, millis=200)
    prey_controller.stop()
    prey_controller.join()

    rob.stop_world()



if __name__ == "__main__":
    main()
