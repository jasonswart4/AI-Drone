import AIDroneLib as lib
import matplotlib.pyplot as plt
# Initialising the game
character_size = [64, 64]
screen_size = [1500, 790]
game = lib.Game(screen_size, character_size)

targets = [[1,10],[0,60]]
drones = [lib.Drone(i,[[0,0,0],[0,0,0],[0,0,0]],targets) for i in range(10)]
scores = [0]
max_score = 0
#r = [0.2,0.01]
#p = [1/len(r) for i in range(len(r))]

tests = 0
while True:
    tests += 1
    [max_index, max_score] = lib.run_test(game, drones, max_score)
    best = drones[max_index].brain.get_weights()
    #r = min(1, 0.001*(max_score**(-1)-1))
    for i in range(len(drones)):
        drones[i].alive = True
        drones[i].set_state([[0,0,0],[0,0,0],[0,0,0]])
        drones[i].brain.set_weights(best)

    if max_score > max(scores):
        drones[0].brain.save('m')

    for j in range(len(drones)):
        drones[j].mutate(game.r)

    game.scores.append(max_score)

    print(tests,'r = ', game.r, ' max score = ', max_score)