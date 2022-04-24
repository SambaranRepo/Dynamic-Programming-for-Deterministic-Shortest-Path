from turtle import forward
import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env
import itertools as it
import os
import matplotlib.patches as mpatches


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

    
    if pos == goal:
        return state
    else:
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
    : same as motion model previously but specifically for environments of Part A
    '''
    pos = state['agent_pos']
    rot = state['agent_dir']
    key = state['key']
    door = state['door']

    if pos == goal:
        return state
    else:
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
    '''
    : Given the gym minigrid environment, define the state space of the Deterministic Shortest Path Problem
    : The state space is the collection of individual states of the environment
    : We define the state to consist of the position, rotation, key picked up or not and the information about the doors being open or not
    : We can get the state space as the cartesian product of all values that each aspect of the state can be assigned. Itertools is a good way to get this
    : We also define an index to each state. Since a key of dictionary cannot be dictionary itself, we create a tuple of key-value pairs of each state in the state space
    '''
    cell = list(range(env.height))
    positions = tuple(x for x in it.product(cell, repeat= 2))
    # print(f'grid positions are : {positions}')
    states = {"agent_pos" : positions, "agent_dir": [(0,-1), (1,0), (0,1), (-1,0)], "key":[(1),(0)]}

    num_doors = 1 if 'door_open' not in info else 2
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

    return state_index, state_space, goal, key_pos, door_pos,empty, num_doors


def terminal_cost(state, goal):
    '''
    : defines the terminal cost of the DSP problem 
    : if the state is among the goal states, the terminal cost is assigned as 0
    : for all states that do not qualify as the goal state, terminal cost is assigned as infinity
    '''
    if state['agent_pos'] == goal:
        return 0 
    else : 
        return float('inf')

def BackDP(state_space, state_index, controls, env, goal, key_pos, door_pos,empty, num_doors):
    '''
    : This function implements the Backward Dynamic Programming algorithm applied to the Deterministic Shortest Path problem formulated
    : Starting from the terminal time, we find the value function as the minimum cost of transition + value function at the next state over all possible controls 
    : V[t, state] = min(V[t + 1, state], min(cost of transition + Value function at next state) where the first term just considers scenario where state remains unchanged
    : We can terminate the dynamic programming if the value function for all states in the state space remain the same for two consecutive times
    '''
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

    We also solve the partB problem using the same function

    First we compute the value function and the optimal policy function using the Dynamic Programming algorithm
    Once we get the V and pi, we start from the current agent position and current time = 0, we do a forward pass using the optimal policies and find the desired path

    '''
    state_index, state_space, goal, key_pos, door_pos,empty,num_doors = define_state_space(env, info)
    controls = [MF, TL, TR, PK, UD]
    V,pi = BackDP(state_space, state_index,controls, env, goal, key_pos, door_pos, empty, num_doors)

    #Now that we have the value function and the policy function, evaluate the sequence from agent current position to goal position

    if 'door_open' in info:
        state = {'agent_pos': tuple(env.agent_pos), 'agent_dir': tuple(env.dir_vec), 'key':0, 'door': tuple(info['door_open'])}
    else:
        state = {'agent_pos': tuple(env.agent_pos), 'agent_dir': tuple(env.dir_vec), 'key':0, 'door': 0}
    optimal_act_seq = []
    value_function = []
    t = 0
    while state['agent_pos'] != goal:
        index = state_index[tuple(state.items())]
        optimal_action = pi[t,index]
        optimal_act_seq.append(optimal_action)
        value_function.append(V[t,index])
        if num_doors == 2:
            state = motion_model(state, optimal_action, goal, key_pos, door_pos, env, empty)
        else:
            state = motion_model_partA(state, optimal_action, goal, key_pos, door_pos, env, empty)
        t+=1
    optimal_act_seq.append(MF)
    value_function.append(0)
    return V,pi,optimal_act_seq, value_function, state_index, goal, key_pos, door_pos,empty, num_doors

def value_near_state(seq,v, env,info, value_function, num_doors, goal, door_pos, key_pos, empty, state_index, path = './results/partB/'):
    '''
    : Given the sequence of optimal controls and the corresponding state value functions
    : this function plots the current value function at the state and for each possible transition to the next state, find the value function at that state
    : From the value function at the next transitioning states, we can verify that the optimal control sequence is the one that transitions to the state with the lowest value function
    '''
    controls  = [0,1,2,3,4]
    
    if 'door_open' not in info:
        state = {'agent_pos' : tuple(env.agent_pos), 'agent_dir': tuple(env.dir_vec),'key':0, 'door':0}
    else:
        state = {'agent_pos' : tuple(env.agent_pos), 'agent_dir': tuple(env.dir_vec),'key':0, 'door':tuple(info['door_open'])}
    

    for i in range(len(seq)):
        V = {}
        action_dict = {0:'MF', 1:'TL', 2:'TR', 3:'PK', 4:'UD'}
        if i < len(seq) - 1:
            for control in controls:
                img = env.render('rgb_array', tile_size = 32)
                if num_doors == 1:
                    next = motion_model_partA(state, control, goal,key_pos, door_pos, env, empty)
                else:
                    next = motion_model(state, control, goal,key_pos, door_pos, env, empty)
                next_index = state_index[tuple(next.items())]
                if value_function[i + 1, next_index] != float('inf'):
                    V.update({action_dict[control] : value_function[i+1, next_index]})
                else:
                    V.update({action_dict[control] : 50})

        else:
            img = env.render('rgb_array', tile_size = 32)
            V = {action_dict[control]: 0 for control in controls}
        
        step(env, seq[i])
        if num_doors == 1:
                state = motion_model_partA(state, seq[i], info,key_pos, door_pos, env, empty)
        else:
            state = motion_model(state, seq[i],info,key_pos, door_pos, env, empty)
        x,y = zip(*V.items())
        print(f'value functions at next states are : {V}')
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax1.set_title(f'Value function at current state : {v[i]}')
        ax2.bar(x,y)
        ax2.set_title('Value functions of next states')
        # plt.subplot_tool()
        # plt.show()
        plt.subplots_adjust(left=0.1,bottom=0.1,right=1, top=0.9, wspace=0.6, hspace=1) 
        plt.savefig(path + f'/Value_function_{i}.png', format = 'png', bbox_inches = 'tight')
        plt.close()
    
    print(f'Images are saved to {path}')
    return

def partA():
    env_folder = './envs/'
    env_all = [os.path.join(env_folder , filename) for filename in os.listdir(env_folder) if filename != 'random_envs']

    for env_path in env_all: 
        name = env_path.split('/')
        name = name[2].split('.')[0]
        env, info = load_env(env_path) # load an environment
        V,pi,seq, value_function, state_index, goal, key_pos, door_pos,empty, num_doors = doorkey_problem(env, info) # find the optimal action sequence
        draw_gif_from_seq(seq, load_env(env_path)[0],f'./gif/partA/{name}.gif') # draw a GIF & save
        save_fig_dir = './results/partA/'
        n = len(os.listdir(save_fig_dir))
        print(f'Number of folders inside this : {n}')
        os.makedirs(save_fig_dir + f'{n + 1}/')
        value_near_state(seq,value_function, env,info, V, num_doors, goal, door_pos, key_pos, empty, state_index, save_fig_dir + f'{n + 1}/')
    
def partB():
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    name = env_path.split('/')
    name = name[2].split('.')[0]
    save_fig_dir = './results/partB/'
    n = len(os.listdir(save_fig_dir))
    print(f'Number of folders inside this : {n}')
    V,pi,seq, value_function, state_index, goal, key_pos, door_pos,empty, num_doors = doorkey_problem(env, info) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0],f'./gif/partB/{n + 1}.gif') # draw a GIF & save
    os.makedirs(save_fig_dir + f'{n + 1}/')
    value_near_state(seq,value_function, env,info, V, num_doors, goal, door_pos, key_pos, empty, state_index, save_fig_dir + f'{n + 1}/')

if __name__ == '__main__':
    #example_use_of_gym_env()
    partA()
    # partB()
    
