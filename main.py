import numpy as np
import os
from tqdm import tqdm

eps = [0.001, 0.005, 0.01, 0.05, 0.1]
gamma = np.linspace(0.1, 1, 10, True)
lr = np.linspace(1e-2, 1, 10, True)
step_reward = [-1, -5, -10]


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
                os.system(f"python ./game4alg.py {e} {g} {l} {sr} >> ./logs/log.txt")
                pbar.update(1)

