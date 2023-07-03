import tensorflow.keras as tf
import random
import numpy as np
import copy
import pygame
import matplotlib.pyplot as plt

def createNet(structure, activF):
    name = tf.Sequential()
    name.add(tf.layers.Input(shape=(structure[0])))
    for i in range(len(structure)-1):
        name.add(tf.layers.Dense(structure[i+1], activation = activF))
    return name

def mutateNet(net,r):
    w = net.get_weights()
    w_new = w
    for i in np.arange(0,len(w)-1,2):
        for j in range(len(w[i][:,0])):
            for k in range(len(w[i][0,:])):
                #w_new[i][j,k] = np.random.choice([w[i][j,k],w[i][j,k]+random.uniform(-.5,.5)],1,p=[1-r,r])
                #w_new[i][j,k] = np.random.choice([w[i][j,k], np.random.randn(), w[i][j,k]+random.uniform(-.5,.5)], 1, p=[1-r,r*1/10,r*9/10])
                w_new[i][j,k] = np.random.choice([w[i][j,k], w[i][j,k]+random.uniform(-.5,.5)], 1, p=[1-r,r])

    mutant = tf.models.clone_model(net)
    mutant.set_weights(w_new)
    return mutant

def mutateBias(net,r):
    w = net.get_weights()
    w_new = w
    for i in np.arange(1,len(w),2):
        for j in range(len(w[i])):
           
            #w_new[i][j] = np.random.choice([w[i][j,k],w[i][j,k]+random.uniform(-.5,.5)],1,p=[1-r,r])
            w_new[i][j] = np.random.choice([w[i][j], w[i][j]+random.uniform(-.5,.5)],1,p=[1-r,r])

    mutant = tf.models.clone_model(net)
    mutant.set_weights(w_new)
    return mutant

def set0(net):
    w = net.get_weights()
    w_new = w
    for i in np.arange(0,len(w)-1,2):
        for j in range(len(w[i][:,0])):
            for k in range(len(w[i][0,:])):
                w_new[i][j,k] = 0

    model = tf.models.clone_model(net)
    model.set_weights(w_new)
    return model

def suvat(s0, v0, a0, dt):
    s = s0 + v0*dt + 0.5*a0*dt**2
    v = v0 + a0*dt
    return [s, v]

def game_init(character_size, screen_size):
    pygame.init()
    [width, height] = screen_size
    [char_width, char_height] = character_size
    win = pygame.display.set_mode((width,height))
    pygame.display.set_caption("AI drone")
    original_drone = pygame.image.load('drone.png')
    unrotated_drone = pygame.transform.scale(original_drone, (char_width,char_height))
    target_dot = pygame.image.load("red_dot.png")
    target_dot = pygame.transform.scale(target_dot, (200,200))
    sprites = [unrotated_drone, target_dot]
    return [sprites, win]

def game_update(character_size, screen_size, positions, target, sprites, win, scores):
    [unrotated_drone, target_dot] = sprites
    [x_new, y_new, th_new] = positions
    [target_x, target_y] = target
    rotated_image = pygame.transform.rotate(unrotated_drone, th_new*180/np.pi)
    [width, height] = screen_size
    [char_width, char_height] = character_size
    win.fill([255,255,255])
    win.blit(rotated_image, ((width-(char_width*abs(np.cos(th_new)) + char_height*abs(np.sin(th_new))))/2 + x_new, (height-(char_width*abs(np.sin(th_new)) + char_height*abs(np.cos(th_new))))/2 - y_new))
    win.blit(target_dot, ((width-200)/2+target_x,(height-200)/2-target_y))
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    text_surface = my_font.render('Score: '+ str(scores[-1]), False, (0, 0, 0))
    win.blit(text_surface, (0,0))
    pygame.display.update()
    events = pygame.event.get()
    stop = False
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                stop = True
        return stop

def pause(scores, mutation_rates, mustBreak):
    user_choice = input("What would you like to do? \n change mutation rates[1] \n plot current scores[2] \n set breaking condition[3] \n")
    if user_choice == "1":
        mutation_rates = []
        while True:
            user_input = input("input mutation rate (~ to break): ")
            if user_input == '~':
                break
            mutation_rates.append(float(user_input))
        prob = [1 / len(mutation_rates) for i in range(len(mutation_rates))]

    elif user_choice == "2":
        plt.figure("scores")
        plt.plot(scores)
        plt.show()

    elif user_choice == "3":
        user_input = int(input("set break condition [1/0]: "))
        if user_input > 0.5:
            mustBreak = True
        else:
            mustBreak = False

    return [mutation_rates, mustBreak]


def run_test(net, initial_conditions, target, max_counter, dt, update_inputs, must_break, scores):
    [character_size, screen_size, sprites, win] = update_inputs
    [iTarget, target_counter] = [0, 0] # iTarget will get increased when target counter is triggered. Target counter is to count how long drone is at target
    [x_target, y_target] = target[iTarget]
    L = []
    R = []

    # Environmental constants
    pi = np.pi
    g = -100
    mass = 1.38 #[kg]
    length = .215 # [m]
    I = 50*(mass*length**2)/12
    M = 0 #[Nm]
    r = length/2
    max_thrust = 500 # [Newtons]

    # Initial conditions
    [[x0, vx0, ax0,], [y0, vy0, ay0,], [th0, om0, alp0]] = initial_conditions
    position_state = [[x0], [y0], [th0]]
    velocity_state = [[vx0], [vy0], [om0]]
    acceleration_state = [[ax0], [ay0], [alp0]]
    counter = 0

    # First update
    [x_new, vx_new] = suvat(x0, vx0, ax0, dt)
    [y_new, vy_new] = suvat(y0, vy0, ay0, dt)
    [th_new, om_new] = suvat(th0, om0, alp0, dt)
    for i in range(3): position_state[i].append([x_new, y_new, th_new][i])
    for i in range(3): velocity_state[i].append([vx_new, vy_new, om_new][i])

    # Scoring variables
    dist = [(x_target**2 + y_target**2)**0.5]
    vel = []
    omega = []
    theta = []
    x_dist = x_target - position_state[0][-1]
    y_dist = y_target - position_state[1][-1]
    dist.append((x_dist ** 2 + y_dist ** 2) ** .5)
    d0 = dist[-1]
    a = np.log(10)/(d0**2)
    
    stop_angle = 90*pi/180
    while abs(x_new) < 400/2 and abs(y_new) < 400/2 and counter < max_counter and abs(th_new) < stop_angle:

        # Brain inputs:
        x_dist = x_target - position_state[0][-1]
        y_dist = y_target - position_state[1][-1]
        dist.append((x_dist**2 + y_dist**2)**.5)

        net_inputs = [[x_dist, y_dist, vx_new, vy_new, np.sin(th_new), om_new]]

        # add noise
        for i in range(len(net_inputs[0])):
            net_inputs[0][i] = net_inputs[0][i] + np.random.rand()*net_inputs[0][i]*0.0001

        [[L_thrust, R_thrust]] = 0.5 + 0.5*net.predict(net_inputs, verbose=0) #percentage of thrust
        L_thrust = min(L_thrust*max_thrust/2,max_thrust/2)
        R_thrust = min(R_thrust*max_thrust/2,max_thrust/2)

        F = (L_thrust + R_thrust)
        M = (-L_thrust + R_thrust)*r
        L.append(L_thrust)
        R.append(R_thrust)

        # Update new
        # Angular
        alp = M/I
        [th_new, om_new] = suvat(th_new, om_new, alp, dt)
        omega.append(om_new)
        theta.append(th_new)

        # x
        Fx = -F*np.sin(th_new)
        ax = Fx/mass
        [x_new, vx_new] = suvat(x_new, vx_new, ax, dt)

        # y 
        Fy = F*np.cos(th_new) + g*mass
        ay = Fy/mass
        [y_new, vy_new] = suvat(y_new, vy_new, ay, dt)

        vel.append((vx_new**2 + vy_new**2)**0.5)

        # Update old
        for i in range(3): position_state[i].append([x_new, y_new, th_new][i])
        for i in range(3): velocity_state[i].append([vx_new, vy_new, om_new][i])
        for i in range(3): acceleration_state[i].append([ax, ay, alp][i])

        stop = game_update(character_size, screen_size, [x_new, y_new, th_new], target[iTarget], sprites, win, scores)
        if stop == True:
            break

        counter +=1
        if dist[-1] > dist[-2] and must_break:
            break
        if dist[-1] < 20:
            target_counter += 1
            if target_counter > 20:
                iTarget +=1
                target_counter = 0
                [x_target, y_target] = target[iTarget]
                x_dist = x_target - position_state[0][-1]
                y_dist = y_target - position_state[1][-1]
                dist.append((x_dist ** 2 + y_dist ** 2) ** .5)
                d0 = dist[-1]
                a = np.log(10)/(d0**2)
        else:
            target_counter = 0
            
    #score = counter/np.mean(dist)   + np.exp(-np.mean(omega)**2) #+ 1/np.mean(vel)

    #for i in range(len(omega)):
    #    score += + np.exp(-np.mean(omega)**2)
        score = 1/(1+(np.mean(dist))) #+ 1/(1+abs(np.mean(omega)) )
    return [score, L, R, stop]
