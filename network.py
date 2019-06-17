import tensorflow as tf
import logging
import numpy as np

def build_network(input, no_outputs, no_layers_nodes, **network_args):
    logging.info("inputsize: %s", input.shape)
    vector = input
    weights = [tf.constant_initializer( np.array([[1,2,3],[4,0.5,2],[3,2,0.5]])),tf.constant_initializer(np.array([0.8,0.3,0.5]))]
    biases = [tf.constant_initializer( np.array([[0.3,-0.2,0.1]])),tf.constant_initializer(np.array([[0.2]]))]
    for i,no_nodes in enumerate(no_layers_nodes):
        vector = tf.layers.dense(vector, no_nodes, name="layer"+str(i), kernel_initializer=weights[i], bias_initializer=biases[i],**network_args)
        tf.print(vector,[vector])
        logging.debug("layer "+str(i)+" : %s",vector.shape)
    network_args['activation']=None
    output = tf.layers.dense(vector, no_outputs, name = "output",kernel_initializer=weights[-1], bias_initializer=biases[-1], **network_args)
    tf.print(output,[output])
    logging.debug("output : %s", output.shape)
    return output
