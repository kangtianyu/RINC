'''
Created on Apr 28, 2018

@author: kangt
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import MultiSparse
import json
import os
import sys

tf.logging.set_verbosity(tf.logging.INFO)


def rinc(features, labels, mode):
    global data
    global glist
    global smatrix
    global asmp
    global lamb
    global eta
    global dataset
    global tidx
    """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, smatrix.shape[1]])

    multismooth_layer = MultiSparse.multisparse(
        inputs=input_layer,
        units=smatrix.shape[1],
        smooth_num = 3,
        smatrix = smatrix,
        name="sp",
        kernel_initializer = tf.random_uniform_initializer(minval=0.7,maxval=0.7,dtype=tf.float64))
    
    encoded_layer = tf.layers.dense(inputs=multismooth_layer, units=2, name="encoded", activation= tf.nn.relu, use_bias = False)
    
    decoded_layer = tf.layers.dense(inputs=encoded_layer, units= smatrix.shape[1], name="fin", use_bias = False)
    
    hmatrix = [v for v in tf.trainable_variables() if v.name == "fin/kernel:0"][0]
    
    
    # Calculate Loss
#     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    lossb = tf.losses.mean_squared_error(labels=input_layer, predictions=decoded_layer)
    reg = tf.trace(tf.matmul(tf.matmul(hmatrix,asmp),tf.transpose(hmatrix)))
    treg = tf.scalar_mul(tf.convert_to_tensor(lamb,dtype = tf.float64),reg)
    loss = lossb + tf.cast(treg,tf.float32)
    
    lossb_tensor = tf.convert_to_tensor(lossb, name = "lossb")
    reg_tensor = tf.convert_to_tensor(reg, name = "reg")
    
    predictions = {
        # Generate predictions
        "classes": encoded_layer,
        "probabilities": tf.nn.softmax(encoded_layer, name="softmax_tensor"),
        "lossb" : lossb_tensor,
        "reg" : reg_tensor
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        trainval2 = [v for v in tf.trainable_variables() if v.name == "sp/kernel:0"]
        trainval1 = [v for v in tf.trainable_variables()]
        trainval1.remove(trainval2[0])
        
        optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=eta)
        train_op1 = optimizer1.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            var_list = trainval1)
        optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=1000)
        train_op2 = optimizer2.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            var_list = trainval2)
        train_op = tf.group(train_op1, train_op2)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    
#     eval_metric_ops = {
#     "accuracy": tf.metrics.accuracy(
#         labels=labels, predictions=predictions["classes"])}

#     return tf.estimator.EstimatorSpec(
#         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss)


def main(unused_argv):
    global data
    global glist
    global smatrix
    global asmp
    global lamb
    global eta
    global dataset
    global tidx
    # Load training and eval data
    print("Preparing")
    train_data = []
    for patient in data:
        row = []
        for gene in glist:
            if gene in data[patient]:
                row.append(float(data[patient][gene]))
            else:
                row.append(0)
        train_data.append(row)
    train_data = np.asarray(train_data)
    
    # Create the Estimator
    sparse_classifier = tf.estimator.Estimator(
        model_fn=rinc, model_dir="/media/tianyu/Data/0_results/rinc_model_"+dataset + "_" + str(tidx))
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {
        "kernel": "sp/kernel",
        "probabilities": "encoded/Relu",
        "lossb":"mean_squared_error/value",
        "reg":"Trace"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    
    # Train the model
    print("Building Training Input")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_data,
        batch_size=len(data),
        num_epochs=None,
        shuffle=True)
    print("Training Start")
    print()
    sparse_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])
    print("Training Finished")
    
    # Evaluate the model and print results
#     print("Building Evaluating Input")
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": train_data},
#         num_epochs=1,
#         batch_size=1,
#         shuffle=False)
#     print("Evaluate Start")
#     eval_results = sparse_classifier.predict(input_fn=eval_input_fn)
#     for itm in eval_results:
#         print(itm)
#     predictions = list(eval_results)
#     predicted_classes = [p["classes"] for p in predictions]
#     print(predicted_classes)

#     for item in eval_results:
#         print(item)
#     gr = tf.get_default_graph()
#     print(gr)
#     sp_kernel_val = gr.get_tensor_by_name('sp/kernel:0').eval()
#     print(sp_kernel_val)
def inst(dset,t):
    global data
    global glist
    global smatrix
    global asmp
    global lamb
    global eta
    global dataset
    global tidx
    
    tidx = t
    dataset = dset
    
    cwd = os.getcwd()

    glist = {}
    with open(cwd + "/../data/new/" + dataset + "_glist.txt") as tfile:
        glist = json.load(tfile)
        
    smatrix = np.load(cwd + "/../data/new/" + dataset + "_smatrix.npy")
        
    data = {}
    with open(cwd + "/../data/new/" + dataset + ".txt") as tfile:
        data = json.load(tfile)
    
    asmp = []
    with open(cwd + "/../data/new/" + dataset + "_asmp.txt") as tfile:
        asmp = json.load(tfile)
    asmp = np.array(asmp)
        
    lamb = 0.01
    eta = 0.01
    
    main(None)
    
if __name__ == "__main__":
    cwd = os.getcwd()
    dataset = 'LGG'
    tidx = ""

    glist = {}
    with open(cwd + "/../data/new/" + dataset + "_glist.txt") as tfile:
        glist = json.load(tfile)
        
    smatrix = np.load(cwd + "/../data/new/" + dataset + "_smatrix.npy")
        
    data = {}
    with open(cwd + "/../data/new/" + dataset + ".txt") as tfile:
        data = json.load(tfile)
    
    asmp = []
    with open(cwd + "/../data/new/" + dataset + "_asmp.txt") as tfile:
        asmp = json.load(tfile)
    asmp = np.array(asmp)
        
    lamb = 0.01
    eta = 0.01
        
    tf.app.run()