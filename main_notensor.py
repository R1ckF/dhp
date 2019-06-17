from dhp_notensor import  *
from env.linefollow import *
import numpy as np
import logging

"""
Create logger
"""
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.debug("debugtest")
logging.info("infotest")
logging.warning("warningtest")

"""
Create agent
"""
no_inputs = 3
no_outputs = 1
A = np.array([[-0.7391, 0.9744],[-1.472, -1.567]])
B = np.array([[-0.08935],[-6.722]])
dt = 0.02
F = A * dt + np.eye(2)
G = B * dt
# modelA = np.array([[1,0],[1,1]])
# modelB = np.array([[1],[1]])
agent = DHPagent(no_inputs, no_outputs, F, G, activation=tanh, actor_net = [6], critic_net = [6], lr=0.1, weights=0.001, bias=0.0001)

"""
Create env
"""
env = lineFollowEnv(F,G, boundaries=False)


"""
main loop
"""
obs = env.reset()
logging.debug("obs: %s" % (obs))

i = 0

while i < 2000:
    action = agent.actor_out(obs)
    agent.obtain_gradients(obs)
    agent.update_networks()

    obs, reward, done, _ = env.step(action)
    logging.debug("obs: %s, done: %s" % (obs,done))
    if done:
        obs = env.reset()
    logging.debug(done)
    i+=1
