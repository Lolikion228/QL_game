import numpy as np
import os
from tqdm import tqdm

eps = np.linspace(1e-4, 0.2, 20, True)
gamma = np.linspace(0.1, 1, 20, True)
lr = np.linspace(1e-2, 1, 15, True)
step_reward = np.linspace(-1,-20, 10, True)


if os.path.exists('./logs/log.txt'):
    os.remove('./logs/log.txt')


pbar = tqdm(total = len(eps) * len(gamma) * len(lr) * len(step_reward), desc='grid searching)))')

for e in eps:
    for g in gamma:
        for l in lr:
            for sr in step_reward:
                f = open('./logs/log.txt', 'a')
                f.write(f"{e} {g} {l} {sr} ")
                f.close()
                os.system(f"python ./game.py {e} {g} {l} {sr} >> ./logs/log.txt")
                pbar.update(1)

