import tensorflow as tf 
import numpy as np
import model, replay_buffer, utils, saver, test
import os, gym, time, gc, shutil
import keras

# настройки
num_episodes = 5000
episode_light = 4000
learning_rate = 0.00025
load_model=True
train=True
eps_decay_steps = 1000000
discount_factor = 0.99
sample = 4
store = 1
batch_size = 32
update_target_network = 10000
pre_train_steps=50000
trace_length = 8

model_type = "DuDRQN"

original_path = os.getcwd()
path = original_path+"/models/"+model_type + '/'

if not os.path.isdir(original_path+"/models/"):
    os.mkdir(original_path+"/models/")
if not os.path.isdir(path):
    os.mkdir(path)


# инициализировать переменные для хранения общих вознаграждений и общих потерь Инициализировать переменные для хранения общих вознаграждений и общих потерь
total_reward = 0
total_loss = 0
test_rew = []

# lists for saving rewards, bugs, test rewards
rewards = []
losses = []
test_rew = []
env = gym.make("Pong-v0")
num_actions = env.action_space.n

h_size = 512
shape_conv = (80,80,1)

if os.path.exists(path+'action'+model_type+'.h5') and os.path.exists(path+'target'+model_type+'.h5') and os.path.exists(path+model_type+ '_buffer_Pong.data') and load_model:
  try:
    shutil.unpack_archive(model_type+'_models.zip',path)
    actionDRQN = keras.models.load_model(path+'action'+model_type+'.h5')
    targetDRQN = keras.models.load_model(path+'target'+model_type+'.h5')
    experiences = replay_buffer.ExperienceReplay(100)
    experiences.load_bin_file(path+model_type+ '_buffer_Pong')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay = 0.01,  clipnorm=1.0)
    saver.load_optimizer_config(path+model_type+ '_optimizer_Adam_Pong')
    total_episodes = saver.load_global_step(path+model_type+'_global_step')
    gc.collect()
    print('---load complite---')
  except:
    print('---load error---')
else:
    print('---new model---')
    total_episodes = 0
    actionDRQN = model.create_model((None,80,80,1),h_size,shape_conv, num_actions)
    targetDRQN = model.create_model((None,80,80,1),h_size,shape_conv, num_actions)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay = 0.01,  clipnorm=1.0)
    experiences = replay_buffer.ExperienceReplay(200)

loss_function = keras.losses.MSE

for episode in range(num_episodes):
    episode_buffer = []
    lstm_state = [tf.zeros((1,h_size),dtype=np.float32),tf.zeros((1,h_size),dtype=np.float32)]
    state = env.reset()
    state = utils.downsample(state)
    # adding a time dimension
    state = np.expand_dims(state,0)
    start_time = time.time() 
    for frame in range(episode_light):
        total_episodes+=1
        # get action
        X = [(np.expand_dims(state,0)/255).astype('float32'),lstm_state[0],lstm_state[1]]
        a = model.predict(X, actionDRQN)
        a, lstm_state  = a[0][-1], [a[1],a[2]]
        a  = np.argmax(np.squeeze(a))
        if train:
            # random action
            action = utils.epsilon_greedy(a, total_episodes,eps_decay_steps)
        else:
            action = a

        next_state, reward, done, _ = env.step(action)
        next_state = utils.downsample(next_state)
        next_state = np.expand_dims(next_state,0)      
        total_reward += reward  
            
        if  (total_episodes % store) == 0: 
                episode_buffer.append([state, action, reward, int(done), next_state])

        if (total_episodes % sample) == 0 and total_episodes > pre_train_steps and len(experiences.buffer) > batch_size:
                sp = experiences.sample(batch_size, trace_length)
                total_loss += model.update_gradients(sp, batch_size, trace_length, num_actions, h_size,  discount_factor , loss_function, optimizer, actionDRQN, targetDRQN)

        if total_episodes % update_target_network == 0:
              # soft update the the target network with new weights
              targetDRQN.set_weights(actionDRQN.get_weights())
              gc.collect()

        
        state = next_state
        if done:
            break

    experiences.appendToBuffer(episode_buffer)

    # test
    if episode % 100 == 0 and episode != 0:
        test_rew = test.model_test(episode_light)

    if total_loss != 0:
        rewards.append(total_reward)
        losses.append(total_loss)

    print("Episode %d - Mean reward = %.3f, last reward = %.3f,test-rew = %.3f Loss = %.6f, Time = %.3f, Total episodes = %.1f" % (episode, np.asarray(rewards).mean(), total_reward,  np.asarray(test_rew).mean(),total_loss, time.time()-start_time, total_episodes))
    total_reward = 0
    total_loss = 0
    done = False

    if episode % 50 == 0 and episode != 0:
        actionDRQN.save(path+'action'+model_type+'.h5')
        targetDRQN.save(path+'target'+model_type+'.h5')
        saver.save_global_step(path+model_type+'_global_step', total_episodes)
        experiences.save_to_bin_file(path+model_type+ '_buffer_Pong')
        saver.save_optimizer_config(path+model_type+ '_optimizer_Adam_Pong')    
        shutil.make_archive(model_type + '_models_Pong', 'zip', path)
        gc.collect()