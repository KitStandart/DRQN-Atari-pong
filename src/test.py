import tensorflow as tf
import keras
import numpy as np 
import model, utils, gym, os
from IPython import display as ipythondisplay     
from PIL import Image

def model_test(episode_light):
    global env, h_size, actionDRQN
    test_rew=[]
    for _ in range(10):
        lstm_state = [tf.zeros((1,h_size),dtype=np.float32),tf.zeros((1,h_size),dtype=np.float32)]
        state = env.reset()
        state = utils.downsample(state)
        state = np.expand_dims(state,0)
        test_total_reward = 0 
        for frame in range(episode_light):
            X = [(np.expand_dims(state,0)/255).astype('float32'),lstm_state[0],lstm_state[1]]
            a = model.predict(X, actionDRQN)
            a, lstm_state  = a[0][-1], [a[1],a[2]]
            action = np.argmax(np.squeeze(a))
            state, reward, done, _ = env.step(action)            
            state = utils.downsample(state)
            state = np.expand_dims(state,0)
            test_total_reward += reward 
            if done:
                break 
        test_rew.append(test_total_reward)
    return test_rew      

def render_episode(env, model, max_steps): 
    global h_size
    lstm_state = [tf.zeros((1,h_size),dtype=np.float32),tf.zeros((1,h_size),dtype=np.float32)]
    state = env.reset()
    state = utils.downsample(state)
    state = np.expand_dims(state,0)
    screen = env.render(mode='rgb_array')
    images = [Image.fromarray(screen)]

    for i in range(1, max_steps + 1):
        a = model.predict([(np.expand_dims(state,0)/255).astype('float32'),lstm_state[0],lstm_state[1]],actionDRQN)
        action_probs, lstm_state  = a[0][-1], [a[1],a[2]]

        action = np.argmax(np.squeeze(action_probs))
        # print(lstm_state[0])
        # print(action)
        state, reward, done, _ = env.step(action)
        state = utils.downsample(state)
        state = np.expand_dims(state,0)
        # Render screen every 10 steps
        if i % 10 == 0:
            screen = env.render(mode='rgb_array')
            images.append(Image.fromarray(screen))

        if done:
            break

    return images

model_type = "DuDRQN"

original_path = os.getcwd()
path = original_path+"/models/"+model_type + '/'
env = gym.make("PongNoFrameskip-v0", render_mode='rgb_array')
h_size = 512

if os.path.exists(path+'action'+model_type+'.h5'):
  try:
    actionDRQN = keras.models.load_model(path+'action'+model_type+'.h5')
    print('---load complite---')
    max_steps_per_episode = 20000
     # Save GIF image
    images = render_episode(env, actionDRQN, max_steps_per_episode)
    image_file = 'Pong.gif'
    images[0].save(
        image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
  except:
    print('---load error---')
else:
  print('---no model---')