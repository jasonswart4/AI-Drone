from functions import *
import tensorflow.keras as tf
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
game = Game(screen_size, character_size)

#targets = [[0,20], [100,30], [-10,20], [50,10], [0,10]]
targets = [[uniform(-100,100), uniform(-100,100)] for _ in range(200)]
drones = [Drone(0,[[0,0,0],[0,0,0],[0,0,0]],targets)]
drones[0].brain = tf.models.load_model('net.h5')
scores = [0]
max_score = 1e6
game.break_angle = 1e6
game.max_counter = 1e6
game.max_dist = 1000
drones[0].dt = 0.02

run_test(game, drones, max_score)
plt.plot(drones[0].thrust[0])
plt.plot(drones[0].thrust[1])
plt.show()