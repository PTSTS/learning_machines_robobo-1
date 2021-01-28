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
import time
import array
from controller import Controller
import math
from blob_detection import detect
import calendar
from datetime import datetime
from operator import attrgetter
start1 = 0
start2 = 0
seed = 222
def seed_everything(seed):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
# seed_everything(seed)
timestamp = calendar.timegm(time.gmtime())

NUMBER_OF_GENERATIONS = 30
NUMBER_OF_INPUTS = 7
NUMBER_OF_INPUTS2 = 8
NUMBER_OF_OUTPUTS = 2
NUMBER_OF_HIDDEN_Neurons_1 = 2
NUMBER_OF_HIDDEN_Neurons_2  = 2
NUMBER_OF_WEIGHTS = (NUMBER_OF_INPUTS + 1) * NUMBER_OF_HIDDEN_Neurons_1+ (NUMBER_OF_HIDDEN_Neurons_1 + 1) \
                    * NUMBER_OF_HIDDEN_Neurons_2 +  (NUMBER_OF_HIDDEN_Neurons_2 + 1) * NUMBER_OF_OUTPUTS
NUMBER_OF_SIGMAS = NUMBER_OF_WEIGHTS

MU = 10  # how many parents per generation
LAMBDA = 30 # how many children per generation
LOWER = NUMBER_OF_WEIGHTS*[-1]
UPPER = NUMBER_OF_WEIGHTS*[1]

TIME_OUT = 45

with open('IP.txt', 'r') as f:
    ip = f.readlines()[0]


rob = robobo.SimulationRobobo(number='').connect(address=ip, port=19997)
rob2 = robobo.SimulationRobobo(number='#0').connect(address=ip, port=19998, prey = True)

def fitness(c, weights, prey =False):
    rob.set_phone_tilt(0.8, 50)
    if not prey:
        controller = c(weights, 7, 2, 2)
        prey_controller = c(best_prey, 8, 2, 2)
    else:
        controller =  c(best_pred, 7, 2, 2)
        prey_controller = c(weights, 8, 2, 2)
    while (rob.is_simulation_running()):
        pass
    rob.play_simulation()

    start = rob.get_sim_time()
    current_time = start
    

    while (current_time - start < TIME_OUT * 1000):
        sensors = rob.read_irs()[3:] # front sensors, length 5
        sensors = [sensors[i] * 5 if sensors[i] is not False else 1 for i in range(len(sensors))]
        detection_x, detection_y = detect(rob.get_image_front())
        
        sensors2 = rob2.read_irs() # front sensors, length 5
        sensors2 = [sensors2[i] * 5 if sensors2[i] is not False else 1 for i in range(len(sensors2))]
       

        if detection_x is False:
            detection_x, detection_y = 0.5, 0
        else:
            detection_x, detection_y = detection_x / 128, 0.5 + detection_y / 256
        inputs = sensors + [detection_x, detection_y]
        inputs2 = sensors2
        x, y = controller.act(inputs)
        x2,y2 = prey_controller.act(inputs2)
        rob.move(float(x), float(y),  millis=200)
        rob2.move(float(x2), float(y2),  millis=200)
        try:
            pos1 = rob.position()
            pos2 = rob2.position()
            distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        except:
            distance = 10
        current_time = rob.get_sim_time()
        if distance <= 0.23: # maximum "catching" distance
            break
    if not (prey and (current_time - start >= TIME_OUT * 1000)):
        fitness_score = (current_time - start) / 1000 + distance*10
    else:
        fitness_score = (current_time - start) / 1000


    # fitness_score += rob.collected_food()
    rob.stop_world()
    return [fitness_score]


# initialize fitness and set fitness weight to positive value (we want to maximize)
creator.create("FitnessMax", base.Fitness, weights=[-1.0])
creator.create("FitnessMin", base.Fitness, weights=[1.0])
# the goal ('fitness') function to be maximized

creator.create("Individual", array.array, typecode="d",
               fitness=creator.FitnessMax, strategy=None)
creator.create("Individual2", array.array, typecode="d",
               fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")
record = 0


def generateWeights(icls, scls, size, imin, imax, smin, smax, prey =False):
    if prey:
        ind =icls(np.loadtxt('fittestseed111.csv')) #initilize with solution of week 1
    else:
        ind = icls(np.loadtxt('weights_111_1611161296.csv'))
    ind.strategy = scls(random.gauss(0, 1) for _ in range(len(ind)))

    return ind


toolbox = base.Toolbox()
toolbox2 = base.Toolbox()

# generation functions
MIN_VALUE, MAX_VALUE = -1., 1.
MIN_STRAT, MAX_STRAT = -1., 1.
toolbox.register("individual", generateWeights, creator.Individual,  # defines an ES individual
                 creator.Strategy, NUMBER_OF_WEIGHTS, MIN_VALUE, MAX_VALUE, MIN_STRAT,
                 MAX_STRAT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox2.register("individual", generateWeights, creator.Individual2,  # defines an ES individual
                 creator.Strategy, NUMBER_OF_WEIGHTS, MIN_VALUE, MAX_VALUE, MIN_STRAT,
                 MAX_STRAT, True)
toolbox2.register("population", tools.initRepeat, list, toolbox2.individual)


MIN_STRATEGY = 0.1


def checkStrategy(minstrategy):  # modifies mutation strats so it doesn't get too small
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


toolbox.register("evaluate", fitness, Controller)
toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox2.register("evaluate", fitness, Controller, prey = True)
toolbox2.register("mate", tools.cxBlend, alpha=0.1)
toolbox2.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.3)
toolbox2.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
fittest = tools.HallOfFame(10)

stats2 = tools.Statistics(lambda ind: ind.fitness.values)
stats2.register("avg", np.mean)
stats2.register("std", np.std)
stats2.register("min", np.min)
stats2.register("max", np.max)
fittest2 = tools.HallOfFame(10)

population = toolbox.population(n=MU)
population2 = toolbox2.population(n=MU)
best_pred = min(population, key=attrgetter("fitness"))
best_prey = max(population2, key=attrgetter("fitness"))
globalLogbook = pd.DataFrame(columns=["gen", "nevals", "avg", "std", "min", "max"])
globalLogbook2 = pd.DataFrame(columns=["gen", "nevals", "avg", "std", "min", "max"])

for i in range(NUMBER_OF_GENERATIONS):
    population, logbook = algorithms.eaMuCommaLambda(population, toolbox, mu=MU, lambda_=LAMBDA,
                                                 cxpb=0.2, mutpb=0.7, ngen=1, stats=stats,
                                                 halloffame=fittest, verbose=True, start = start1)
    population2, logbook2 = algorithms.eaMuCommaLambda(population2, toolbox2, mu=MU, lambda_=LAMBDA,
                                                 cxpb=0.2, mutpb=0.7, ngen=1, stats=stats2,
                                                 halloffame=fittest2, verbose=True, start = start2)
    
    globalLogbook = pd.concat([globalLogbook, pd.DataFrame(logbook)], sort =False)
    globalLogbook2 = pd.concat([globalLogbook2, pd.DataFrame(logbook2)], sort =False)
    best_pred = max(population, key=attrgetter("fitness"))
    best_prey = min(population2, key=attrgetter("fitness"))
    start1 =1
    start2 =1
    
    
    
pd.DataFrame(globalLogbook2).to_csv(
    "./generations_fitness_Best_Prey{}_{}.csv".format(seed, timestamp), index=False
)
pd.DataFrame(np.array(best_prey)).to_csv(
    "./weights_Best_Prey{}_{}.csv".format(seed, timestamp), header=False, index=False
)

pd.DataFrame(globalLogbook).to_csv(
    "./generations_fitness_Best_Pred{}_{}.csv".format(seed, timestamp), index=False
)
pd.DataFrame(np.array(best_prey)).to_csv(
    "./weights_Best_Pred{}_{}.csv".format(seed, timestamp), header=False, index=False
)
