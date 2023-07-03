import AIDroneLib as lib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pygame

# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
[sprites, win] = lib.game_init(character_size, screen_size)

# Inputs needed for updating the game screen
update_inputs = [character_size, screen_size, sprites, win]

# initial parameters

initial_conditions = [[0,0,0] , [0,0,0] , [0,0,0]]
target = [[1e-6,100], [100,100]]
max_counter = 100000
dt = 0.05

net = load_model('m')
[score, L, R, stop] = lib.run_test(net, initial_conditions, target, max_counter, dt, update_inputs, False, [0])


plt.plot(L)
plt.plot(R)
plt.legend(["Left Thrust", "Right Thrust"])
plt.show()

print(score)
print(net.get_weights())