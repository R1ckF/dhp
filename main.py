from env.linefollow import *
import logging
import tensorflow as tf
from network import *
from dhp import *
import numpy as np

"""
Create logger
"""
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.debug("debugtest")
logging.info("infotest")
logging.warning("warningtest")

"""
create environment
"""
env = lineFollowEnv(display=False, seed=0)

"""
initialize models and create DHP structure
"""
modelA = np.array([[1,0],[1,2]])
modelB = np.array([[2],[1]])
agent = DHPagent(3, 1, modelA, modelB, activation=tf.nn.tanh)#, kernel_initializer=tf.ones_initializer)

sess = agent.sess
sess.run(tf.global_variables_initializer())


"""
test
"""

state = env.reset()
print('state: ', state)
state.shape
state = np.array([[0.1,0.2,0.1]])
node1, output,action,  gradients, dr_ds, action_grad = sess.run([agent.next_state, agent.output,agent.action, agent.ds1_ds, agent.dr_ds, agent.action_grad], {agent.state: state})
print('action: ', node1, output, action, gradients, action_grad)




# np.arctanh(np.array([0.9051482,0.2913126,0.6910696]))
# prev_state = state
# state, reward, done, _ = env.step(action)
# print('state, reward, done: ', state, reward, done)
#
# state = np.vstack((prev_state,state))
#
# print(state)
# action, critic, next_state, preward, critic_error, critic_loss = sess.run([agent.action, agent.critic, agent.next_state, agent.reward, agent.critic_error, agent.critic_loss], {agent.state: state})
#
# action
# critic
# next_state
# preward
# critic_error
# critic_loss






# action, critic

# action, critic, next_state, gradients = sess.run([agent.reward, agent.next_state, agent.dr_ds], {agent.state: np.array([[0.1,0.1,0.1]])})






# sess=tf.Session()
# inputs = tf.placeholder(tf.float32, shape = [None, 2], name = "input")
#
#
# outputs = build_network(inputs, 2, [6], activation=tf.nn.tanh)
#
# writer = tf.summary.FileWriter("test",sess.graph)
# writer.close()
