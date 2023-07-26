import AIDroneLib as lib
import tensorflow.keras as tf
import copy
import numpy as np
# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
game = lib.Game(screen_size, character_size)

targets = [[0,0], [0,20], [100,30], [-10,20], [50,10], [0,10]]
drones = [lib.Drone(i,[[0,0,0],[0,0,0],[0,0,0]],targets) for i in range(100)]
scores = [0]
max_score = -1e6
#r = [0.2,0.01]
#p = [1/len(r) for i in range(len(r))]

game.break_angle = 30
game.r = 0.01
game.r_auto = False
game.max_counter = 1500
game.max_dist = 200

# For model trained in another run
#the_chosen_one = tf.models.load_model('n')
#for i in range(len(drones)): drones[i].brain  = copy.copy(the_chosen_one)
#for j in range(len(drones)):
#   drones[j].mutate(game.r)

tests = 0
while True:
    tests += 1
    [max_index, max_score] = lib.run_test(game, drones, max_score)
    best = drones[max_index].brain.get_weights()
    #r = min(1, 0.001*(max_score**(-1)-1))
    theta = np.random.rand()*2*3.14
    for i in range(len(drones)):
        drones[i].alive = True
        drones[i].iTarget = 0
        #drones[i].set_state([[0.1*np.random.randn(),0.1*np.random.randn(),0.1*np.random.randn()],[0,0,0],[0,0,0]])
        drones[i].set_state([[0,0,0],[0,0,0],[0,0,0]])
        drones[i].brain.set_weights(best)

    if max_score > max(game.scores):
        drones[0].brain.save('m')

    for j in range(len(drones)):
        drones[j].mutate(game.r)

    game.scores.append(max_score)

    print(tests,'r = ', game.r, ' max score = ', max_score)
    print("break angle = ", game.break_angle)

    '''
    if tests%10 == 0:
        targets[0][0] = 100*np.cos(tests/10*np.pi * tests / 180)
        targets[0][1] = 100*np.sin(tests/10*np.pi * tests/ 180)
    '''