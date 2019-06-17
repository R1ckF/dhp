import tensorflow as tf
import logging
from network import *


class DHPagent:

    def __init__(self, no_inputs, no_outputs, modelA, modelB, actor_net = [3], critic_net = [3], **network_args):
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs
        self.modelA = modelA
        self.modelB = modelB
        self.sess = tf.Session()

        self.state = tf.placeholder(tf.float32, shape = [None,self.no_inputs], name = "state")
        # self.state_1 = tf.placeholder(tf.float32, shape = [1,self.no_inputs], name = 'state_1')
        with tf.variable_scope('actor'):
            weights = [tf.constant_initializer( np.array([[1,2,3],[4,0.5,2],[3,2,0.5]]).T),tf.constant_initializer(np.array([0.8,0.3,0.5]))]
            biases = [tf.constant_initializer( np.array([[0.3,-0.2,0.1]])),tf.constant_initializer(np.array([[0.2]]))] #

            self.node1 = tf.layers.dense(self.state, 3, name="layer1", kernel_initializer=weights[0], bias_initializer=biases[0],activation=tf.nn.tanh)
            self.output = tf.layers.dense(self.node1, 1, name = "output",kernel_initializer=weights[-1], bias_initializer=biases[-1],activation=None)
            self.action = self.output#build_network(self.state, self.no_outputs, actor_net, **network_args)
            logging.info('actor created')
        # with tf.variable_scope('critic'):
        #     self.critic = build_network(self.state, self.no_inputs-1, critic_net, **network_args)
        #     logging.info('critic created')
        with tf.variable_scope('model'):
            self.modelA = tf.constant(self.modelA, dtype=tf.float32, name='A')
            self.modelB = tf.constant(self.modelB, dtype=tf.float32, name='B')
            self.gamma  = tf.constant(0.9, dtype=tf.float32, name='gamma')
            logging.info('model created')

        with tf.variable_scope('loop'):
            self.sliced_state = tf.transpose(self.state[:1,:2])
            test = tf.matmul(self.modelB, self.action)
            print('test: ', test.shape, tf.matmul(self.modelA, self.sliced_state ).shape, self.sliced_state.shape )
            self.next_state = tf.transpose(tf.matmul(self.modelA, self.sliced_state ) + tf.matmul(self.modelB, self.action))
            self.df_ds = self.modelA
            self.df_da = self.modelB

            self.reward = - tf.square((self.next_state[0,1]-self.state[0,2]), name='reward')
            self.dr_ds = tf.gradients(self.reward, self.next_state)
            # self.dr_ds =  tf.multiply(self.next_state[0,1]-self.state[0,2], tf.constant([-2], dtype=tf.float32), name='dr_ds')
            self.ds1_ds = tf.gradients(self.next_state, self.state)
            self.action_grad = tf.gradients(self.action, self.state)
            # print(self.dr_ds[0].shape, self.gamma.shape, self.critic[1,:].shape, self.ds1_ds[0].shape, self.critic[0,:].shape, (-1*(self.dr_ds[0] + self.gamma * self.critic[1,:])).shape)
            # self.critic_error = -(self.dr_ds[0] + self.gamma * self.critic[1,:])*self.ds1_ds[0] + self.critic[0,:]
            # self.critic_loss = 0.5 * tf.matmul(self.critic_error, tf.transpose(self.critic_error))


        writer = tf.summary.FileWriter("test",self.sess.graph)
        writer.close()
