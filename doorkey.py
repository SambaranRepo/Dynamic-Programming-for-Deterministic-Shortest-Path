from turtle import forward
import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env
import itertools as it
import os

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

forward_rotation = {}
forward_rotation[(0,-1)] = {TL : (-1,0), TR : (1,0)}
forward_rotation[(0,1)] = {TL : (1,0), TR : (-1,0)}
forward_rotation[(-1,0)] = {TL : (0,1), TR : (0,-1)}
forward_rotation[(1,0)] = {TL : (0,-1), TR : (0,1)}

inverse_rotation = {}
inverse_rotation[(0,-1)] = {TL : (1,0), TR : (-1,0)}
inverse_rotation[(0,1)] = {TL : (-1,0), TR : (1,0)}
inverse_rotation[(-1,0)] = {TL : (0,-1), TR : (0,1)}
inverse_rotation[(1,0)] = {TL : (0,1), TR : (0,-1)}


#Build the forward motion model 
def motion_model(state, action, goal, key_pos, door_pos, env, empty): 
    '''
    : builds the motion model of the robot
    : Control space : {0: MF, 1: TL, 2: TR, 3: PK, 4: UD}
    : if control is MF, move forward in the direction it is facing i.e. x_(t + 1) = x_t + current_rot
    : if TL or TR, rotate 90 degree left or right from current orientation. New orientation will depend on current orientation
    : if current orientation = UP = (0,-1) ---> TL := (-1, 0), TR := (1,0), therefore forward_rotation[UP] = {TL : (-1,0), TR : (1,0)}
    : if current orientation = DOWN = (0,1) ---> TL := (1, 0), TR := (-1,0), therefore forward_rotation[DOWN] = {TL : (1,0), TR : (-1,0)}
    : if current orientation = LEFT = (-1,0) ---> TL := (0, 1), TR := (0,-1), therefore forward_rotation[LEFT] = {TL : (0,1), TR : (0,-1)}
    : if current orientation = RIGHT = (1,0) ---> TL := (0, -1), TR := (0,1), therefore forward_rotation[RIGHT] = {TL : (0,-1), TR : (0,1)}
    : if action is PK --> key = 1
    : if action is UD ---> door = 1
    : rest of the times, key and door are 0 indicating key has not been picked and door is closed
    '''
    pos = state['agent_pos']
    rot = state['agent_dir']
    key = state['key']
    door = state['door']

    

    if action == MF : 
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])
        if front in empty or front == goal: 
            pos = front
        if key and front == key_pos:
            pos = front
        if front in door_pos:
            door_index = [np.array_equal(front, d) for d in door_pos]
            door_index = door_index.index(True)
            if door[door_index]:
                pos = front
        
    elif action == TL or action == TR: 
        rot = forward_rotation[rot][action]
    
    elif action == PK : 
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])
        if not key and front == key_pos:
            key = 1
        else:
            key = 0
    else: 
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])
        if key and front in door_pos:
            door_index = [np.array_equal(front, d) for d in door_pos] 
            door_index = door_index.index(True)
            if door[door_index] == False:
                if door_index == 0:
                    door = (1,door[1])
                elif door_index == 1:
                    door = (door[0], 1)

    return {'agent_pos': pos, 'agent_dir':rot, 'key': key, 'door':door}


def motion_model_partA(state, action, goal, key_pos, door_pos, env, empty): 
    '''
    : builds the motion model of the robot
    : Control space : {0: MF, 1: TL, 2: TR, 3: PK, 4: UD}
    : if control is MF, move forward in the direction it is facing i.e. x_(t + 1) = x_t + current_rot
    : if TL or TR, rotate 90 degree left or right from current orientation. New orientation will depend on current orientation
    : if current orientation = UP = (0,-1) ---> TL := (-1, 0), TR := (1,0), therefore forward_rotation[UP] = {TL : (-1,0), TR : (1,0)}
    : if current orientation = DOWN = (0,1) ---> TL := (1, 0), TR := (-1,0), therefore forward_rotation[DOWN] = {TL : (1,0), TR : (-1,0)}
    : if current orientation = LEFT = (-1,0) ---> TL := (0, 1), TR := (0,-1), therefore forward_rotation[LEFT] = {TL : (0,1), TR : (0,-1)}
    : if current orientation = RIGHT = (1,0) ---> TL := (0, -1), TR := (0,1), therefore forward_rotation[RIGHT] = {TL : (0,-1), TR : (0,1)}
    : if action is PK --> key = 1
    : if action is UD ---> door = 1
    : rest of the times, key and door are 0 indicating key has not been picked and door is closed
    '''
    pos = state['agent_pos']
    rot = state['agent_dir']
    key = state['key']
    door = state['door']

    

    if action == MF : 
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])
        if front in empty or front == goal: 
            pos = front
        if key and front == key_pos:
            pos = front
        if front == door_pos:
            if door:
                pos = front
        
    elif action == TL or action == TR: 
        rot = forward_rotation[rot][action]
    
    elif action == PK : 
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])
        if not key and front == key_pos:
            key = 1
        else:
            key = 0
    else: 
        front = tuple([pos[0] + rot[0], pos[1] + rot[1]])
        if key and front == door_pos:
            if door == False:
                door = 1

    return {'agent_pos': pos, 'agent_dir':rot, 'key': key, 'door':door}



def define_state_space(env, info): 
    cell = list(range(env.height))
    positions = tuple(x for x in it.product(cell, repeat= 2))
    # print(f'grid positions are : {positions}')
    states = {"agent_pos" : positions, "agent_dir": [(0,-1), (1,0), (0,1), (-1,0)], "key":[(1),(0)]}

    if 'door_open' not in info:
        states["door"] = tuple([(1),(0)])
    elif 'door_open' in info:
        states["door"] = tuple([(1,0), (0,1), (1,1), (0,0)])
    
    #The entire state space will be the set product of all possible values of the states
    keys, values = zip(*states.items())
    state_space = [dict(zip(keys, x)) for x in it.product(*values)]
    print(f'length of state space : {len(state_space)}')
    #Define special states like goal position, key position, door position 
    goal = tuple(info['goal_pos'])
    key_pos = tuple(info['key_pos'])
    if 'door_open' in info:
        door_pos = [tuple(info['door_pos'][0]), tuple(info['door_pos'][1])]
    else:
        door_pos = tuple(info['door_pos'])
    print(f'door position : {door_pos}')
    env_matrix = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0]
    empty = np.where(env_matrix == 1)
    empty = tuple(zip(empty[0], empty[1]))

    #index the states for ease of use 
    n = len(state_space)
    state_index = {}
    for i in range(n): 
        state_index.update({tuple(state_space[i].items()):i})

    return state_index, state_space, goal, key_pos, door_pos,empty


def terminal_cost(state, goal):
    if state['agent_pos'] == goal:
        return 0 
    else : 
        return float('inf')

def BackDP(state_space, state_index, controls, env, goal, key_pos, door_pos,empty, num_doors):
    T = len(state_space) - 1
    V = np.zeros((T + 1, len(state_space)))
    pi = np.zeros_like(V).astype(np.int8)

    #Value function at terminal time 
    for i, s in enumerate(state_space):
        V[T,i] = terminal_cost(s, goal)

    #Perform the backwards dynamic programming algorithm
    for t in range(T-1, -1, -1):
        cij = np.zeros((len(state_space), len(controls)))

        for i,s in enumerate(state_space):
            for c,action in enumerate(controls):
                if num_doors == 2:
                    next = motion_model(s, action, goal, key_pos, door_pos, env,empty)
                else:
                    next = motion_model_partA(s, action, goal, key_pos, door_pos, env,empty)
                next_index = state_index[tuple(next.items())]
                cij[i, c] = step_cost(action) + V[t+1, next_index]
            V[t,i] = min(V[t +1 , i], cij[i,:].min())
            pi[t, i] = controls[np.argmin(cij[i,:])]
        
        if all(V[t,:] == V[t + 1, :]): 
            print('Dynamic programming converged')
            print(f'number of iterations is : {T - t}')
            return V[t + 1:], pi[t + 1:]
    
    return V, pi





      

def doorkey_problem(env, info):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    state_index, state_space, goal, key_pos, door_pos,empty = define_state_space(env, info)
    controls = [MF, TL, TR, PK, UD]
    num_doors = 2 if 'door_open' in info else 1
    V,pi = BackDP(state_space, state_index,controls, env, goal, key_pos, door_pos, empty, num_doors)

    #Now that we have the value function and the policy function, evaluate the sequence from agent current position to goal position

    if 'door_open' in info:
        state = {'agent_pos': tuple(env.agent_pos), 'agent_dir': tuple(env.dir_vec), 'key':0, 'door': tuple(info['door_open'])}
    else:
        state = {'agent_pos': tuple(env.agent_pos), 'agent_dir': tuple(env.dir_vec), 'key':0, 'door': 0}
    optimal_act_seq = []
    t = 0
    while state['agent_pos'] != goal:
        index = state_index[tuple(state.items())]
        optimal_action = pi[t,index]
        optimal_act_seq.append(optimal_action)
        if num_doors == 2:
            state = motion_model(state, optimal_action, goal, key_pos, door_pos, env, empty)
        else:
            state = motion_model_partA(state, optimal_action, goal, key_pos, door_pos, env, empty)
        t+=1
    optimal_act_seq.append(MF)
    return optimal_act_seq


def partA():
    env_folder = './envs/'
    env_all = [os.path.join(env_folder , filename) for filename in os.listdir(env_folder) if filename != 'random_envs']

    for env_path in env_all: 
        name = env_path.split('/')
        name = name[2].split('.')[0]
        env, info = load_env(env_path) # load an environment
        plot_env(env)
        seq = doorkey_problem(env, info) # find the optimal action sequence
        draw_gif_from_seq(seq, load_env(env_path)[0],f'./gif/partA/{name}.gif') # draw a GIF & save
    
def partB():
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    name = env_path.split('/')
    name = name[2].split('.')[0]
    seq = doorkey_problem(env, info) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0],f'./gif/partB/{name}.gif') # draw a GIF & save
    plot_env(env)

if __name__ == '__main__':
    #example_use_of_gym_env()
    partA()
    # partB()    
    
