import pygame
import numpy as np
import time
import matplotlib.pyplot as plt

pygame.init()

w, h = 11, 8
square_size = 100

traps_cords0 = [ (2,2), (w-3,1), (w-3,2), (w-2,2), (w-4,5)]
energy_cords0 = [ (0,0), (w//2,h//2), (w-1,h-1) ]

window_w, window_h = w*square_size, h*square_size

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


episodes_cnt = 200
max_iter = 10000
eps = 0.04
gamma = 1.0
lr = 1.0

q_table = np.zeros( (w, h, 4) )

reward_table = np.zeros( (w,h) )

for i in range(w):
    for j in range(h):
        if (i,j) in traps_cords0:
            reward_table[i][j] = -100
        elif (i,j) in energy_cords0:
            reward_table[i][j] = 10
        elif (i,j) == (w-1, 0):
            reward_table[i][j] = 50
        else:
            reward_table[i][j] = -1

reward_table0 = reward_table.copy()


def run_episode(render=True, interval=0.3):
    global reward_table, reward_table0
    env = Env()
    total_reward = 0

    for i in range(max_iter):
        x, y = env.get_state()
        if np.random.uniform(0,1) < eps:
            action = np.random.randint(4)
        else:
            logits = q_table[x][y]
            softmaxed = np.exp(logits)/np.sum(np.exp(logits))
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
    reward_table = reward_table0.copy()


    return total_reward, int(env.moves), int(env.traps_cnt), int(env.energy_cnt)


class Env():

    def __init__(self):

        self.traps_cords = [(2, 2), (w - 3, 1), (w - 3, 2), (w - 2, 2), (w - 4, 5)]
        self.energy_cords = [(0, 0), (w // 2, h // 2), (w - 1, h - 1)]
        self.player_x = 0
        self.player_y = h-1
        self.moves = 0
        self.traps_cnt = 0
        self.energy_cnt = 0
        self.is_done = False

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
                step_reward = reward_table[self.player_x][self.player_y]
                reward_table[self.player_x][self.player_y] = -1
                break

        for i in range(len(self.energy_cords)):
            if (self.player_x, self.player_y) == self.energy_cords[i]:
                self.energy_cnt += 1
                self.energy_cords.pop(i)
                step_reward = reward_table[self.player_x][self.player_y]
                reward_table[self.player_x][self.player_y] = -1
                break

        if (self.player_x, self.player_y) == (w-1, 0):
            step_reward = reward_table[self.player_x][self.player_y]
            self.is_done = True

        if step_reward is None:
            step_reward = reward_table[self.player_x][self.player_y]

        return step_reward

    def get_state(self):
        return self.player_x, self.player_y

    def render(self,sum_reward):
        display_surface.fill(color=(255, 255, 0))

        for i in range(h):
            for j in range(w):
                pygame.draw.rect(display_surface,'black',[j*square_size,i*square_size,square_size,square_size], 1)

        display_surface.blit(pivo_surf, ( (w-1)*square_size, 0))

        for x,y in self.energy_cords:
            display_surface.blit(energy_surf, (x * square_size, y*square_size))

        for x,y in self.traps_cords:
            display_surface.blit(trap_surf, (x * square_size, y * square_size))

        display_surface.blit(player_surf, (self.player_x * square_size, self.player_y * square_size))
        display_surface.blit(info_surf, (w*square_size, 0))

        for i,(item,name) in enumerate( zip([self.traps_cnt, self.energy_cnt, self.moves, sum_reward], ['traps','energy','moves', 'sum_reward']) ):
            info = (f'{name}:{ item }')
            text = font.render(info, True, 'white')
            display_surface.blit(text,(w*square_size+20, (i+1)*100))

        pygame.display.update()
        


ep_rewards = []
ep_moves = []
ep_traps = []
ep_energy = []

for i in range(episodes_cnt):
    episode_reward, moves, traps, energy = run_episode(render=True, interval=0.01)
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

plt.plot( range(episodes_cnt)[10:], ep_rewards[10:] )
plt.xlabel('episode')
plt.ylabel('episode_summary_reward')
plt.show()
plt.close()

plt.plot( range(episodes_cnt)[10:], ep_moves[10:] )
plt.xlabel('episode')
plt.ylabel('episode_moves')
plt.show()
plt.close()

plt.scatter( range(episodes_cnt), ep_traps )
plt.xlabel('episode')
plt.ylabel('episode_traps')
plt.show()
plt.close()

plt.scatter( range(episodes_cnt), ep_energy )
plt.xlabel('episode')
plt.ylabel('episode_energy')
plt.show()
plt.close()

