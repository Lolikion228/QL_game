import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pickle

class Env():

    def __init__(self, render):

        self.traps_cords = traps_cords_init.copy()
        self.energy_cords = energy_cords_init.copy()
        self.player_x = 0
        self.player_y = h-1
        self.moves = 0
        self.traps_cnt = 0
        self.energy_cnt = 0
        self.is_done = False
        self.do_render = render

        if self.do_render == True:
            self.display_surface, self.pivo_surf, self.energy_surf, self.trap_surf, \
            self.player_surf, self.info_surf, self.clock, self.font = self.game_init()

    def step(self, action): #reward, done
        """
        actions:
        0 - left
        1 - right
        2 - up
        3 - down
        """
        if action == 0:
            if self.player_x > 0:
                self.player_x -= 1
                self.moves += 1
            else:
                self.moves += 1
                return -1.0
        elif action == 1:
            if self.player_x < w - 1:
                self.player_x += 1
                self.moves += 1
            else:
                self.moves += 1
                return -1.0
        elif action == 2:
            if self.player_y > 0:
                self.player_y -= 1
                self.moves += 1
            else:
                self.moves += 1
                return -1.0
        elif action == 3:
            if self.player_y < h - 1:
                self.player_y += 1
                self.moves += 1
            else:
                self.moves += 1
                return -1.0

        step_reward = None
        for i in range(len(self.traps_cords)):
            if (self.player_x,self.player_y) == self.traps_cords[i]:
                self.traps_cnt += 1
                self.traps_cords.pop(i)
                step_reward = curr_reward_table[self.player_x][self.player_y]
                curr_reward_table[self.player_x][self.player_y] = -1
                break

        for i in range(len(self.energy_cords)):
            if (self.player_x, self.player_y) == self.energy_cords[i]:
                self.energy_cnt += 1
                self.energy_cords.pop(i)
                step_reward = curr_reward_table[self.player_x][self.player_y]
                curr_reward_table[self.player_x][self.player_y] = -1
                break

        if (self.player_x, self.player_y) == (w-1, 0):
            step_reward = curr_reward_table[self.player_x][self.player_y]
            self.is_done = True

        if step_reward is None:
            step_reward = curr_reward_table[self.player_x][self.player_y]

        return step_reward

    def get_state(self):
        return self.player_x, self.player_y

    def game_init(self):

        pygame.init()

        window_w, window_h = w * square_size, h * square_size

        display_surface = pygame.display.set_mode((window_w + 400, window_h))
        pygame.display.set_caption('bebrik')
        clock = pygame.time.Clock()

        info_surf = pygame.Surface( (400,window_h) )
        info_surf.fill('black')

        player_surf = pygame.image.load('images/player2.png')
        player_surf = pygame.transform.scale(player_surf, (square_size,square_size) ).convert_alpha()

        pivo_surf = pygame.image.load('images/pivo.png')
        pivo_surf = pygame.transform.scale(pivo_surf, (square_size,square_size) ).convert_alpha()

        energy_surf = pygame.image.load('images/energy.png')
        energy_surf = pygame.transform.scale(energy_surf, (square_size,square_size) ).convert_alpha()

        trap_surf = pygame.image.load('images/trap.png')
        trap_surf = pygame.transform.scale(trap_surf, (square_size,square_size) ).convert_alpha()

        font = pygame.font.Font('freesansbold.ttf', 32)

        return display_surface, pivo_surf, energy_surf, trap_surf, player_surf, info_surf, clock, font
    

    def render(self,sum_reward):
        self.display_surface.fill(color=(255, 255, 0))

        for i in range(h):
            for j in range(w):
                pygame.draw.rect(self.display_surface,'black',[j*square_size,i*square_size,square_size,square_size], 1)

        self.display_surface.blit(self.pivo_surf, ( (w-1)*square_size, 0))

        for x,y in self.energy_cords:
            self.display_surface.blit(self.energy_surf, (x * square_size, y*square_size))

        for x,y in self.traps_cords:
            self.display_surface.blit(self.trap_surf, (x * square_size, y * square_size))

        self.display_surface.blit(self.player_surf, (self.player_x * square_size, self.player_y * square_size))
        self.display_surface.blit(self.info_surf, (w*square_size, 0))

        for i,(item,name) in enumerate( zip([self.traps_cnt, self.energy_cnt, self.moves, sum_reward], ['traps','energy','moves', 'sum_reward']) ):
            info = (f'{name}:{ item }')
            text = self.font.render(info, True, 'white')
            self.display_surface.blit(text,(w*square_size+20, (i+1)*100))

        pygame.display.update()


def run_episode(render=False, interval=0.3):
    global curr_reward_table, reward_table_init
    env = Env(render)
    total_reward = 0

    for _ in range(max_iter):
        x, y = env.get_state()
        if np.random.uniform(0,1) < eps:
            action = np.random.randint(4)
        else:
            logits = q_table[x][y]
            softmaxed = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(range(4), p=softmaxed)

        reward = env.step(action)
        done = env.is_done
        total_reward += reward
        if render:
            env.render(total_reward)
            time.sleep(interval)
        x_,y_ = env.get_state()

        q_table[x][y][action] = q_table[x][y][action] + lr * (reward + gamma * np.max(q_table[x_][y_]) - q_table[x][y][action])

        if done:
            break

    curr_reward_table = reward_table_init.copy()

    return total_reward, int(env.moves), int(env.traps_cnt), int(env.energy_cnt)


def run_experiment(render=False, verbose=False):
    ep_rewards = []
    ep_moves = []
    ep_traps = []
    ep_energy = []

    for i in tqdm(range(episodes_cnt)):
        episode_reward, moves, traps, energy = run_episode(render=render, interval=0.01)
        if verbose:
            print(f'----------------episode_{i+1}-----------------  ')
            print(f'    Reward: {episode_reward}')
            print(f'    Moves: {moves}')
            print(f'    Traps: {traps}')
            print(f'    Energy: {energy}')
        ep_rewards.append(episode_reward)
        ep_moves.append(moves)
        ep_traps.append(traps)
        ep_energy.append(energy)


    pygame.quit()

    return ep_rewards, ep_moves, ep_traps, ep_energy


def plot_experiment_results(ep_rewards, ep_moves, ep_traps, ep_energy, offset):

    plt.plot( range(episodes_cnt)[offset:], ep_rewards[offset:] )
    plt.xlabel('episode')
    plt.ylabel('episode_summary_reward')
    plt.show()
    plt.close()

    plt.plot( range(episodes_cnt)[offset:], ep_moves[offset:] )
    plt.xlabel('episode')
    plt.ylabel('episode_moves')
    plt.show()
    plt.close()

    plt.scatter( range(episodes_cnt)[offset:], ep_traps[offset:] )
    plt.xlabel('episode')
    plt.ylabel('episode_traps')
    plt.show()
    plt.close()

    plt.scatter( range(episodes_cnt)[offset:], ep_energy[offset:] )
    plt.xlabel('episode')
    plt.ylabel('episode_energy')
    plt.show()
    plt.close()



# game init
w, h = 11, 8  # weight and height of game field in squares
square_size = 100 
traps_cords_init = [ (2,2), (w-3,1), (w-3,2), (w-2,2), (w-4,5)]
energy_cords_init = [ (0,0), (w//2,h//2), (w-1,h-1), (4,5)]




if len(sys.argv) == 1:
    # algo init
    episodes_cnt = 1001
    max_iter = 1000   # maximum possible number of iterations during one episode
    eps = 0.04        # probability to take random action 
    gamma = 1.0      # discount factor
    lr = 0.1

    trap_reward = -100
    energy_reward = 50
    step_reward = -5
    fin_reward = 200

elif len(sys.argv) == 10:
    # algo init
    episodes_cnt = int(sys.argv[1])
    max_iter = int(sys.argv[2])     # maximum possible number of iterations during one episode
    eps = float(sys.argv[3])        # probability to take random action 
    gamma = float(sys.argv[4])      # discount factor
    lr = float(sys.argv[5])

    trap_reward = float(sys.argv[6])
    energy_reward = float(sys.argv[7])
    step_reward = float(sys.argv[8])
    fin_reward = float(sys.argv[9])

else:
    raise Exception(f"you should pass 0 or 9 args, you passed {len(sys.argv)-1} args")



q_table = np.zeros( (w, h, 4) )

reward_table_init = np.zeros( (w,h) )
for i in range(w):
    for j in range(h):
        if (i,j) in traps_cords_init:
            reward_table_init[i][j] = trap_reward
        elif (i,j) in energy_cords_init:
            reward_table_init[i][j] = energy_reward
        elif (i,j) == (w-1, 0):
            reward_table_init[i][j] = fin_reward
        else:
            reward_table_init[i][j] = step_reward

curr_reward_table = reward_table_init.copy()

ep_rewards, ep_moves, ep_traps, ep_energy = run_experiment(render=False, verbose=False)

str_params = map(str,[episodes_cnt, max_iter, eps, gamma, lr, trap_reward, energy_reward, step_reward, fin_reward])
str_params = '_'.join(str_params)

with open(f"./logs/{str_params}.pickle", 'wb') as f:
    pickle.dump( [ep_rewards, ep_moves, ep_traps, ep_energy], f)

