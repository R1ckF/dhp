import numpy as np
from NN import *
from activations import *
# import PyQt5
# import matplotlib
# matplotlib.use('Qt5Agg')
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]


class DHPagent:
    """
    Class providing a framework to train a DHP algorithmself.
    Inputs:
        no_inputs: number of states used as input to the ANN (int)
        no_outputs: number of actions the ANN needs to output (int)
        modelA: A matrix of the state space problem that needs to be solves (np.array dim(2))
        modelB: B matrix of the state space problem that needs to be solved (np.array dim(2))
        actor_net: list containing the amount of nodes in each layer for the actor network. Each entry represents a hidden layers (list of int)
        critic_net: list containing the amount of nodes in each layer for the critic network. Each entry represents a hidden layers (list of int)
        gamma: discount factor used (float 0-1)
        **network_args: additional arguments passed to the ANN class
    """
    def __init__(self, no_inputs, no_outputs, modelA, modelB, actor_net = [3], critic_net = [3], gamma=0.9, **network_args):
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs
        self.gamma = gamma

        self.modelA = modelA
        self.modelB = modelB
        self.P = np.array([[0,1]])
        self.actor = NN(self.no_inputs, self.no_outputs, actor_net, **network_args)
        self.critic = NN(self.no_inputs, self.no_inputs-1, critic_net, **network_args)
        self.actor_out = self.actor.forward_pass
        self.critic_out = self.critic.forward_pass

    def forward_pass(self, state):
        """
        Calculate forward pass of the actor and critic.
        """
        return self.actor_out(state), self.critic_out(state)

    def obtain_gradients(self, state):
        """
        actual DHP algorithm. Calculates the current gradients used to train the NN
            First the next state is calculated from which a reward is generated
            dr_ds: derivative of the reward with respect to the previous states
            ds1_ds: derivative of the next state with respect to the previous states
            critic_error_gradient: gradients fed to the critic network for training
            actor_error_gradient: gradients fed to the actor network for training
        """

        model_state = np.transpose(state[:1,:2])
        self.next_state = self.modelA @ model_state + self.modelB @ self.actor_out(state)
        self.reward = -1*np.transpose(self.P @ self.next_state - state[0,2]) @ (self.P @ self.next_state - state[0,2])
        self.dr_ds = -2 * np.transpose(self.P @ self.next_state - state[0,2]) @ self.P
        self.ds1_ds = np.sum(self.modelA,0) + np.sum(self.modelB,0)*self.actor.dout_din()[:1,:2]
        self.next_state = np.vstack((self.next_state, state[0,2])).T
        self.critic_error_gradient = (self.dr_ds + self.gamma * self.critic_out(self.next_state))*self.ds1_ds + self.critic_out(state)
        self.actor_error_gradient = -(self.dr_ds + self.gamma * self.critic_out(self.next_state)) @ self.modelB
        # print(self.critic_error_gradient, self.actor_error_gradient)

    def update_networks(self):
        """
        Call backward pass of critic and actor
        Update of the critic and actor networks
        """
        critic_dw, critic_db, critic_dout_in = self.critic.backward_pass(self.critic_error_gradient)
        self.critic.update_network()
        actor_dw, actor_db, actor_dout_in = self.actor.backward_pass(self.actor_error_gradient)
        self.actor.update_network()

if __name__ == '__main__':

    # target = np.array([[20,100]])
    no_inputs = 3
    no_outputs = 1
    modelA = np.array([[1,1],[3,0.1]])
    modelB = np.array([[1],[1]])
    agent = DHPagent(no_inputs, no_outputs, modelA, modelB, activation=tanh, actor_net = [6], critic_net = [6], lr=0.01, weights=0.01, bias=0.001)
    state = np.array([[0.1,0.3,0.1]])
    # print(agent.actor.W, agent.actor.B)

    print(agent.actor_out(state))
    y,y_ref,reward, dr_ds, ds1_ds, critic_error_gradient, actor_error_gradient = [],[],[], [], [], [], []
    critic_out = []
    action = []
    actor_dout_in = []


    # agent.obtain_gradients(state)
#     agent.reward
#     (0.300877-0.1)
#     agent.critic_out(agent.next_state)
#     agent.next_state
#     agent.actor.dout_din()
#     agent.ds1_ds
#     agent.critic_out(state)
#     agent.critic_error_gradient
#     agent.actor_error_gradient
    # agent.dr_ds
#
#
#     agent.update_networks()
#     agent.actor.dW
# agent.actor.nodes_out[1]*0.402
#


    # -(-0+0.9*-0.001415) + -(-0.4017 + 0.9*0.000723)

    for i in range(1000):
        # print(agent.actor_out(state))
        agent.obtain_gradients(state)
        actor_dout_in.append(agent.actor.dout_din()[:1,:2])
        agent.update_networks()
        # print(agent.next_state)
        y.append(agent.next_state[0,1])
        y_ref.append(0.1)
        reward.append(agent.reward[0][0])
        dr_ds.append(agent.dr_ds[0,1])
        ds1_ds.append(agent.ds1_ds)
        action.append(agent.actor_out(state)[0][0])
        critic_error_gradient.append(agent.critic_error_gradient)
        actor_error_gradient.append(agent.actor_error_gradient[0][0])
        critic_out.append(agent.critic_out(agent.next_state))



    ds1_ds = np.squeeze(np.array(ds1_ds))
    critic_error_gradient = np.squeeze(np.array(critic_error_gradient))
    critic_out = np.squeeze(np.array(critic_out))
    actor_dout_in = np.squeeze(np.array(actor_dout_in))
    cross = []
    for i in range(len(y)-1):
        if y[i] -0.1 <=0 and y[i+1]-0.1 >0:
            cross.append(i+1)
        elif y[i] -0.1 >0 and y[i+1]-0.1 <=0:
            cross.append(i+1)
        else:
            pass


    plt.figure()
    plt.plot(y, label='y')
    plt.plot(y_ref,color='y',linestyle='--')
    plt.plot(action, label='action')
    plt.plot(reward, label = 'reward')
    for i in cross:
        plt.axvline(i,color='r',linestyle='--')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(dr_ds, label = 'dr_ds')
    # plt.plot(ds1_ds, label = 'ds1_ds')
    plt.plot(actor_error_gradient, label='actor_error_gradient')
    plt.plot(critic_error_gradient, label='critic_error_gradient')
    for i in cross:
        plt.axvline(i,color='r',linestyle='--')
    plt.plot(actor_dout_in, label='actor_dout_in')
    plt.legend()
    plt.show()

#
# agent.critic_error_gradient
# agent.actor_error_gradient
#
# plt.plot(actor_dout_in)
# plt.plot(ds1_ds)# print(agent.actor.dout_din())
# plt.figure()
# plt.plot(critic_out[:,0], label = 'critic_out')
# plt.plot(critic_error_gradient[:,0], label = 'critic_error_gradient')
# plt.legend()
#     #
# critic_out
#
# agent.actor.B

    # print(agent.actor.backward_pass(np.array([[1]])))


    #
    # x = np.array([0.1,0.2,0.1])
    # W1 = np.array([[1,2,3],[4,0.5,2],[3,2,0.5]])
    # W1=W1.T
    # W2 = np.array([0.8,0.3,0.5])
    # B1 = np.array([[0.3,-0.2,0.1]])
    # B2 = np.array([[0.2]])
    # action = (np.tanh(x@W1 + B1) @W2 +B2)
    # modelA@ state[:1,:2].T + modelB @ action
    # (1-np.tanh(np.array([1.1,0.5,0.85]))**2)
    # (1-np.tanh(np.array([1.1,0.5,0.85]))**2) * W2
    #
    # 1*0.287+4*0.2359+3*0.2612
