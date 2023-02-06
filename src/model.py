import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def conv_layers(shape):
    #creating a cnn
    input1 = layers.Input(shape=shape, dtype='float32')
    batch_norm_layer1 = layers.BatchNormalization()(input1)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(batch_norm_layer1)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    batch_norm_layer2 = layers.BatchNormalization()(layer3)
    layer4 = layers.Flatten()(batch_norm_layer2)
    return keras.Model(inputs=input1, outputs=layer4) 
    
def create_model(shape, h_size, shape_conv, num_actions):
    input1 = layers.Input(shape=shape, dtype='float32')
    #inputs of temporary states
    input2 = layers.Input(shape=(h_size,), dtype='float32')
    input3 = layers.Input(shape=(h_size,), dtype='float32')

    layer5 = layers.TimeDistributed(conv_layers(shape_conv))(input1)

    layer6 = layers.LSTM(h_size, activation="tanh", return_sequences = True, return_state=True)(layer5, initial_state = [input2, input3])
    layer7 = layers.Dense(h_size, activation="tanh")(layer6[0])
    layer8 = layers.Dense(h_size, activation="tanh")(layer6[0])

    q_f = layers.Dense(1, activation=None)(layer7)
    a_f = layers.Dense(num_actions, activation=None)(layer8)

    action = q_f + a_f - tf.reduce_mean(a_f)

    return keras.Model(inputs=[input1, input2, input3], outputs=[action, layer6[1], layer6[2]])

@tf.function
def predict(x,model):
    return model(x,training=False)

@tf.function
def update_gradients(sp, batch_size, trace_length, num_actions, h_size,  discount_factor , loss_function, optimizer, actionDRQN, targetDRQN):
    #loading memories from the memory buffer
    mem_state, mem_action, mem_reward, mem_done, mem_next_state = sp
    mem_lstm_state = [tf.zeros((batch_size,h_size),dtype=np.float32),tf.zeros((batch_size,h_size),dtype=np.float32)]

    Q1 = actionDRQN([mem_next_state, mem_lstm_state[0], mem_lstm_state[1]],training=False)[0]
    Q2 = targetDRQN([mem_next_state, mem_lstm_state[0], mem_lstm_state[1]],training=False)[0]
    # changing the tensor shape from [batch_size, timesteps, num_actions] to [batch_size*timesteps, num_actions] since from the memory of the sequence batch_size*timesteps
    Q2 = tf.reshape(Q2,[batch_size*trace_length,num_actions])
    Q1 = tf.reshape(Q1,[batch_size*trace_length,num_actions])
    ind = tf.argmax(Q1, axis=1)
    Q2 = tf.gather(Q2, ind, batch_dims=1)
  
    # calculate Q value
    Qtarget = mem_reward + discount_factor * Q2 * (tf.ones(batch_size*trace_length)-mem_done)
    # action mask 
    masks = tf.one_hot(mem_action, num_actions)

    with tf.GradientTape(persistent=False) as tape:
      Q1 = actionDRQN([mem_state,mem_lstm_state[0], mem_lstm_state[1]],training=True)[0]
      Q1 = tf.reshape(Q1,[batch_size*trace_length,num_actions])
      Q1 = tf.reduce_sum(tf.multiply(Q1, masks), axis=1)
      loss_actionDRQN = loss_function(Qtarget, Q1)

    grad_actionDRQN = tape.gradient(loss_actionDRQN, actionDRQN.trainable_variables)                    
    optimizer.apply_gradients(zip(grad_actionDRQN,actionDRQN.trainable_variables))
    del tape
    return loss_actionDRQN          