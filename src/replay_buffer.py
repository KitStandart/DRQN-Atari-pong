from collections import deque
import random
import numpy as np
import pickle
import tensorflow as tf

class ExperienceReplay():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)        
        self.buffer_size = buffer_size
        
    def appendToBuffer(self, memory_tuplet):  
        self.buffer.append(memory_tuplet)  
        
    def check_len_episode(self, n, trace_length):
        memory_index = random.sample(self.buffer,n)
        for episode in memory_index:
            if len(episode) <= trace_length:
                return True, memory_index
        return False, memory_index

    def sample(self, n ,trace_length):
        mem_state = []
        mem_action = []
        mem_reward = []
        mem_done = []
        mem_next_state = []
        check = True
        while check:
            check, memory_index = self.check_len_episode(n, trace_length)
        for episode in memory_index:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            b = episode[point:point + trace_length]
            for j in range(trace_length):
              mem_state.append(b[j][0])
              mem_action.append(b[j][1])
              mem_reward.append(b[j][2])
              mem_done.append(b[j][3])
              mem_next_state.append(b[j][4])
        mem_state = np.asarray(mem_state)
        mem_state = tf.convert_to_tensor((mem_state/255).reshape(n,trace_length,*mem_state.shape[2:]),dtype = np.float32)
        mem_action = tf.convert_to_tensor(np.asarray(mem_action),dtype = np.int32)
        mem_reward = tf.convert_to_tensor(np.asarray(mem_reward),dtype = np.float32)
        mem_done = tf.convert_to_tensor(np.asarray(mem_done),dtype = np.float32)
        mem_next_state = np.asarray(mem_next_state)
        mem_next_state = tf.convert_to_tensor((mem_next_state/255).reshape(n,trace_length,*mem_next_state.shape[2:]),dtype = np.float32)
        return mem_state, mem_action, mem_reward, mem_done, mem_next_state

    def save_to_bin_file(self, filename):
        with open(filename+'.data', 'wb+') as f:
            pickle.dump(self.buffer, f)
        f.close()
        del f

    def load_bin_file(self, path):
        with open(path+'.data', 'rb') as f:
          self.buffer = pickle.load(f)
        f.close()
        del f