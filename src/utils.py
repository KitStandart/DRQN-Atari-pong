import numpy as  np
import cv2

def epsilon_greedy(action, step,eps_decay_steps):
    global env
    eps_min = 0.01
    eps_max = 1.0
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) *step/eps_decay_steps)
    if p < epsilon:
        return env.action_space.sample
    else:
        return action

def downsample(observation):
      s = cv2.cvtColor(observation[30:,:,:], cv2.COLOR_BGR2GRAY)
      s = cv2.resize(s, (80,80), interpolation = cv2.INTER_AREA) 
      return np.expand_dims(s,-1)              

      