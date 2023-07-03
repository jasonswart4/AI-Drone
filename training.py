import AIDroneLib as lib
import tensorflow.keras as tf
import matplotlib.pyplot as plt
import pygame
import numpy as np
import copy
# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
[sprites, win] = lib.game_init(character_size, screen_size)
pygame.font.init()

# Inputs needed for updating the game screen
update_inputs = [character_size, screen_size, sprites, win]

# initial parameters

net = lib.createNet([6,8,8,2], "tanh")
#net = tf.models.load_model('n')
initial_conditions = [[0,0,0] , [0,0,0] , [0,0,0]]
target = [[100,40], [100,100], [50, 40], [30,30], [-30,100]]
max_counter = 100000
dt = 0.05
scores = [0]

mutation_rates = [0.2,0.02]
#mutation_rates = [0.01,0.001]
prob = [1/len(mutation_rates) for i in range(len(mutation_rates))]
r = 1 #mutation_rates[0]
no_progress = -1 # Counts how many iterations had no progress
mustBreak = True

runs = 0
while True:
    mutant = lib.mutateNet(net, r)
    [new_score, L, R, stop] = lib.run_test(mutant, initial_conditions, target, max_counter, dt, update_inputs, mustBreak, scores)
    if new_score > scores[-1]:
        mutant.save('m')
        net = copy.copy(mutant)
        scores.append(new_score)
        no_progress = -1
    else:
        scores.append(scores[-1])

    if no_progress > 3 :
        r = np.random.choice(mutation_rates,1,p=prob)[0]

    runs+=1
    print(r, "  current score is: ", new_score, "   runs: ", runs)
    no_progress += 1

    if stop == True:
        [mutation_rates, mustBreak] = lib.pause(scores, mutation_rates, mustBreak)
        prob = [1/len(mutation_rates) for i in range(len(mutation_rates))]
        
pygame.quit()
plt.figure(1)
plt.plot(scores)
plt.show()
plt.figure(2)
print(net.get_weights())