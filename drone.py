import AIDroneLib as lib
import tensorflow.keras as tf
import matplotlib.pyplot as plt
# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
game = lib.Game(screen_size, character_size)

targets = [[100,0],[0,60]]
drones = [lib.Drone(0,[[0,0,0],[0,0,0],[0,0,0]],targets)]
drones[0].brain.set_weights(tf.models.load_model('m').get_weights())
scores = [0]
max_score = 1e6
game.break_angle = 1e6
drones[0].dt = 0.05

lib.run_test(game, drones, max_score)
plt.plot(drones[0].thrust[0])
plt.plot(drones[0].thrust[1])
plt.show()