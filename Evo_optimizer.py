from sklearn.model_selection import train_test_split

from api import State
import pickle
import time
import sys
import numpy as np
# This package contains various machine learning algorithms
from sklearn.neural_network import MLPClassifier  ###########

from bots.ml_evo import ml_evo

import operator
import random

import numpy
import math

from deap import base
from deap import creator
from deap import tools

from bots.ml.ml import features

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)  # FitnessMin


# creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
#     smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


# Optimize the Bot's brain (such that the model.pkl is better at calculating probability)
def optimizing(theta):  # data = data, target = target)
    with open('/Users/xxx/Projects/schnapsen/rdeep_dataset.pkl', 'rb') as output:  # Change to your own path to the data set
        data, target = pickle.load(output)

    X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target, random_state=1)

    learner = MLPClassifier(hidden_layer_sizes=int(np.ceil(theta[0])), learning_rate_init=theta[1], alpha=theta[2],
                            verbose=True, early_stopping=True, n_iter_no_change=6)

    model = learner.fit(X_train, y_train)
    model.predict_proba(X_test)

    model.predict(X_test)
    print(model.score(X_test, y_test))
    # print(type(model.score(X_test,y_test)))
    return model.score(X_test,
                       y_test),  # return the best error/residual (best error is cloes to zero) The error of a model is the difference between your predicted outcome and the real observed outcome and therefore 0 is desired


# Optimize the Bot's strategy (based on the parameters in ml_evo.py)
def play_game(x):
    phase = 2
    games = 2000
    player = ml_evo.Bot(model_file='/Users/xxx/Desktop/schnapsen/bots/ml_evo/rdeep_model.pkl', theta=x) # Fill in your own path to the model
    data = []
    target = []

    # For progress bar
    bar_length = 30
    start = time.time()
    one = 0
    two = 0

    for g in range(games - 1):

        # For progress bar
        if g % 10 == 0:
            percent = 100.0 * g / games
            sys.stdout.write('\r')
            sys.stdout.write(
                "Playing games: [{:{}}] {:>3}%".format('=' * int(percent / (100.0 / bar_length)), bar_length,
                                                       int(percent)))
            sys.stdout.flush()

        # Randomly generate a state object starting in specified phase.
        state = State.generate(phase=phase)

        state_vectors = []

        while not state.finished():
            # Give the state a signature if in phase 1, obscuring information that a player shouldn't see.
            given_state = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state

            # Add the features representation of a state to the state_vectors array
            state_vectors.append(features(given_state))

            # Advance to the next state
            move = player.get_move(given_state)
            state = state.next(move)

        winner, score = state.winner()

        for state_vector in state_vectors:
            data.append(state_vector)

            if winner == 1:
                result = 'won'
                one += 1

            elif winner == 2:
                result = 'lost'
                two += 1

            target.append(result)
            # print(len(target))

    # with open(path, 'wb') as output:
    #     pickle.dump((data, target), output, pickle.HIGHEST_PROTOCOL)

    # For printing newline after progress bar
    print("\nDone. Time to generate dataset: {:.2f} seconds".format(time.time() - start))
    print('Totaal ' + str(one + two))
    print('Player1: ' + str(one))
    print(('Player2: ' + str(two)))

    # return data, target
    # print(score)
    # print(one/(one+two))
    return one / (one + two),  # Please check if the correct win and losses are used?


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=0.0000001, pmax=1, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", play_game)  # this is a global though so that's why they are not defined


def main():
    pop = toolbox.population(n=5)  # Number of pop
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 7  #it was 1000 from the beginning
    best = None

    for g in range(GEN):
        for part in pop:
            print(part)
            part.fitness.values = toolbox.evaluate(part)  #####
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    print("Best particle: ", best)

    return pop, logbook, best


if __name__ == "__main__":
    main()
