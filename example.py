import numpy as np
import gym
from utils import *

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def example_use_of_gym_env():
    '''
    The Coordinate System:
        (0,0): Top Left Corner
        (x,y): x-th column and y-th row
    '''
    
    print('<========== Example Usages ===========> ')
    env_path = './envs/example-8x8.env'
    # env, info = load_env(env_path) # load an environment
    
    env, info = load_env('./envs/doorkey-6x6-shortcut.env')
    # env, info, _ = load_random_env('./envs/random_envs/')
    print('<Environment Info>\n')
    print(info) # Map size
                # agent initial position & direction, 
                # key position, door position, goal position
    print('<================>\n')            
    
    # Visualize the environment
    plot_env(env) 
    
    
    # Get the agent position
    agent_pos = env.agent_pos
    
    # Get the agent direction
    agent_dir = env.dir_vec # or env.agent_dir
    
    # Get the cell in front of the agent
    front_cell = env.front_pos # == agent_pos + agent_dir
    
    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3) # NoneType, Wall, Key, Goal
    print(f"number of doors : {len(info['door_pos'])}")
    # Get the door status
    # door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    # is_open = door.is_open
    # is_locked = door.is_locked

    # doors_status = info['door_open']
    # print(f'Door 1 : {doors_status[0]} ; Door 2 : {doors_status[1]}')
    
    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    
    # Take actions
    print(f'agent old pos : {env.agent_pos}')
    cost, done = step(env, MF) # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'agent new pos : {env.agent_pos}')
    print('Moving Forward Costs: {}'.format(cost))
    cost, done = step(env, TL) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Turning Left Costs: {}'.format(cost))
    cost, done = step(env, TR) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Turning Right Costs: {}'.format(cost))
    cost, done = step(env, PK) # MF=0, TL=1, TR=2, PK=3, UD=4
    print(f'agent old pos : {env.agent_pos}')
    print('Picking Up Key Costs: {}'.format(cost))
    print(f'agent new pos : {env.agent_pos}')
    cost, done = step(env, UD) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Unlocking Door Costs: {}'.format(cost))   
    
    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")
    
    # The number of steps so far
    print('Step Count: {}'.format(env.step_count))

    env_info = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0]
    print(f'env info : {env_info}')
    print(f'goal : {np.where(env_info == 8)}')
    print(f'door : {np.where(env_info == 4)}')
    print(f'key  : {np.where(env_info == 5)}')
    print(f'EMPTY  : {np.where(env_info == 1)}')
    
if __name__ == '__main__': 
    example_use_of_gym_env()