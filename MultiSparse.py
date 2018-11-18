'''
Created on Jun 4, 2018

@author: tianyu
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.util.tf_export import tf_export

@tf_export('layers.SparseL')
class MultiSparse(base.Layer):
    """Densely-connected layer class.
    This layer implements the operation:
    `outputs = activation(inputs * kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Arguments:
        units: Integer or Long, dimensionality of the output space.
        activation: Activation function (callable). Set it to None to maintain a
        linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularizer: Regularizer function for the weight matrix.
        bias_regularizer: Regularizer function for the bias.
        activity_regularizer: Regularizer function for the output.
        kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
        bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such cases.
        reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Properties:
        units: Python integer, dimensionality of the output space.
        activation: Activation function (callable).
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer instance (or name) for the kernel matrix.
        bias_initializer: Initializer instance (or name) for the bias.
        kernel_regularizer: Regularizer instance for the kernel matrix (callable)
        bias_regularizer: Regularizer instance for the bias (callable).
        activity_regularizer: Regularizer instance for the output (callable)
        kernel_constraint: Constraint function for the kernel matrix.
        bias_constraint: Constraint function for the bias.
        kernel: Weight matrix (TensorFlow variable or tensor).
        bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """

    def __init__(self, units,
                       smatrix,
                       smooth_num,
                       activation=None,
                       use_bias=False,
                       kernel_initializer=None,
                       bias_initializer=init_ops.zeros_initializer(),
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None,
                       kernel_constraint=None,
                       bias_constraint=None,
                       trainable=True,
                       name=None,
                       **kwargs):
        super(MultiSparse, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.units = units
        self.smatrix = smatrix
        self.smooth_num = smooth_num
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)
    
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `SparseL` '
                           'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[1, 1],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)

        if self.use_bias:
            self.bias = self.add_variable('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True
    
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 1:
            # Broadcasting is required for the inputs.
            outputs1=[]
            outputss=[]
            outputst = standard_ops.tensordot(inputs, self.smatrix, [[len(shape) - 1],[0]])
            outputs1.append(standard_ops.scalar_mul(self.kernel[0][0],outputst))
            outputs2 = standard_ops.scalar_mul(1.0-self.kernel[0][0],inputs)
            outputss.append(standard_ops.add(outputs1[0], outputs2))
            for i in range(1,self.smooth_num):
                outputst = standard_ops.tensordot(outputss[i-1], self.smatrix, [[len(shape) - 1],[0]])
                outputs1.append(standard_ops.scalar_mul(self.kernel[0][0],outputst))
                outputss.append(standard_ops.add(outputs1[i], outputs2))
            outputs = outputss[self.smooth_num-1]   
        # Reshape the output back to the original ndim of the input.
        if not context.executing_eagerly():
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        else:
            outputs2 = gen_math_ops.mat_mul(1.0-self.kernel[0][0],inputs)
            outputs = gen_math_ops.mat_mul(inputs, self.smatrix)
            outputs = gen_math_ops.mat_mul(outputs, self.kernel)
            outputs = gen_math_ops.add(outputs, outputs2)
            for i in range(1,self.smooth_num):
                outputs = gen_math_ops.mat_mul(outputs, self.smatrix)
                outputs = gen_math_ops.mat_mul(outputs, self.kernel)
                outputs = gen_math_ops.add(outputs, outputs2)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
    
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)
    
    
@tf_export('layers.sparsel')
def multisparse(
        inputs, units,smatrix, smooth_num,
        activation=None,
        use_bias=False,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    """Functional interface for the densely-connected layer.
    This layer implements the operation:
    `outputs = activation(inputs.kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Arguments:
        inputs: Tensor input.
        units: Integer or Long, dimensionality of the output space.
        activation: Activation function (callable). Set it to None to maintain a
        linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularizer: Regularizer function for the weight matrix.
        bias_regularizer: Regularizer function for the bias.
        activity_regularizer: Regularizer function for the output.
        kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
        bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer.
        reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        Returns:
        Output tensor the same shape as `inputs` except the last dimension is of
        size `units`.
    Raises:
        ValueError: if eager execution is enabled.
    """
    layer = MultiSparse(units,smatrix,smooth_num,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
    return layer.apply(inputs)