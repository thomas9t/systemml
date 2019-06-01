---
layout: global
title: Beginner's Guide for Keras2DML users
description: Beginner's Guide for Keras2DML users
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>


# Layers supported in Keras2DML

If a Keras layer or a hyperparameter is not supported, we throw an error informing that the layer is not supported.
We follow the Keras specification very closely during DML generation and compare the results of our layers (both forward and backward) with Tensorflow to validate that.

- Following layers are not supported but will be supported in near future: `Reshape, Permute, RepeatVector, ActivityRegularization, Masking, SpatialDropout1D, SpatialDropout2D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Cropping1D, Cropping2D, GRU and Embedding`.
- Following layers are not supported by their 2D variants exists (consider using them instead): `UpSampling1D, ZeroPadding1D, MaxPooling1D, AveragePooling1D and Conv1D`.
- Specialized `CuDNNGRU and CuDNNLSTM` layers are not required in SystemML. Instead use `LSTM` layer. 
- We do not have immediate plans to support the following layers: `Lambda, SpatialDropout3D, Conv3D, Conv3DTranspose, Cropping3D, UpSampling3D, ZeroPadding3D, MaxPooling3D, AveragePooling3D and ConvLSTM2D*`.

# Frequently asked questions

#### How do I specify the batch size, the number of epochs and the validation dataset?

Like Keras, the user can provide `batch_size` and `epochs` via the `fit` method. 

```python
# Either:
sysml_model.fit(features, labels, epochs=10, batch_size=64, validation_split=0.3)
# Or
sysml_model.fit(features, labels, epochs=10, batch_size=64, validation_data=(Xval_numpy, yval_numpy))
```

Note, we do not support `verbose` and `callbacks` parameters in our `fit` method. Please use SparkContext's `setLogLevel` method to control the verbosity.


#### How can I get the training and prediction DML script for the Keras model?

The training and prediction DML scripts can be generated using `get_training_script()` and `get_prediction_script()` methods.

```python
from systemml.mllearn import Keras2DML
sysml_model = Keras2DML(spark, keras_model, input_shape=(3,224,224))
print(sysml_model.get_training_script())
```

#### What is the mapping between Keras' parameters and Caffe's solver specification ? 

|                                                        | Specified via the given parameter in the Keras2DML constructor | From input Keras' model                                                                 | Corresponding parameter in the Caffe solver file |
|--------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------|
| Solver type                                            |                                                                | `type(keras_model.optimizer)`. Supported types: `keras.optimizers.{SGD, Adagrad, Adam}` | `type`                                           |
| Learning rate schedule                                 | `lr_policy`                                                    | The `LearningRateScheduler` callback in the `fit` method is not supported.              | `lr_policy` (default: step)                      |
| Base learning rate                                     |                                                                | `keras_model.optimizer.lr`                                                              | `base_lr`                                        |
| Learning rate decay over each update                   |                                                                | `keras_model.optimizer.decay`                                                           | `gamma`                                          |
| Global regularizer to use for all layers               | `regularization_type,weight_decay`                             | The current version of Keras2DML doesnot support custom regularizers per layer.         | `regularization_type,weight_decay`               |
| If type of the optimizer is `keras.optimizers.SGD`     |                                                                | `momentum, nesterov`                                                                    | `momentum, type`                                 |
| If type of the optimizer is `keras.optimizers.Adam`    |                                                                | `beta_1, beta_2, epsilon`. The parameter `amsgrad` is not supported.                    | `momentum, momentum2, delta`                     |
| If type of the optimizer is `keras.optimizers.Adagrad` |                                                                | `epsilon`                                                                               | `delta`                                          |

#### What optimizer and loss does Keras2DML use by default if `keras_model` is not compiled ?

If the user does not `compile` the keras model, then we throw an error.

For classification applications, you can consider using cross entropy loss and SGD optimizer with nesterov momentum:

```python 
keras_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
```

Please refer to [Keras's documentation](https://keras.io/losses/) for more detail.

#### What is the learning rate schedule used ?

Keras2DML does not support the `LearningRateScheduler` callback. 
Instead one can set the custom learning rate schedule to one of the following schedules by using the `lr_policy` parameter of the constructor:
- `step`: return `base_lr * gamma ^ (floor(iter / step))` (default schedule)
- `fixed`: always return `base_lr`.
- `exp`: return `base_lr * gamma ^ iter`
- `inv`: return `base_lr * (1 + gamma * iter) ^ (- power)`
- `poly`: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return `base_lr (1 - iter/max_iter) ^ (power)`
- `sigmoid`: the effective learning rate follows a sigmod decay return b`ase_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))`

#### How to set the size of the validation dataset ?

Like Keras, the validation dataset can be set in two ways:
1. `validation_split` parameter (of type `float` between 0 and 1) in the `fit` method: It is the fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
2. `validation_data` parameter (of type `(x_val, y_val)` where `x_val` and `y_val` are NumPy arrays) in the `fit` method: on which to evaluate the loss at the end of each epoch. The model will not be trained on this data.  validation_data will override validation_split.

#### How do you ensure that Keras2DML produce same results as other Keras' backend?

To verify that Keras2DML produce same results as other Keras' backend, we have [Python unit tests](https://github.com/apache/systemml/blob/master/src/main/python/tests/test_nn_numpy.py)
that compare the results of Keras2DML with that of TensorFlow. We assume that Keras team ensure that all their backends are consistent with their TensorFlow backend.

#### How can I train very deep models on GPU?

Unlike Keras where default train and test algorithm is `minibatch`, you can specify the
algorithm using the parameters `train_algo` and `test_algo` (valid values are: `minibatch`, `allreduce_parallel_batches`, 
`looped_minibatch`, and `allreduce`). Here are some common settings:

|                                                                          | PySpark script                                                                                                                           | Changes to Network/Solver                                              |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Single-node CPU execution (similar to Caffe with solver_mode: CPU)       | `lenet.set(train_algo="minibatch", test_algo="minibatch")`                                                                               | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Single-node single-GPU execution                                         | `lenet.set(train_algo="minibatch", test_algo="minibatch").setGPU(True).setForceGPU(True)`                                                | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Single-node multi-GPU execution (similar to Caffe with solver_mode: GPU) | `lenet.set(train_algo="allreduce_parallel_batches", test_algo="minibatch", parallel_batches=num_gpu).setGPU(True).setForceGPU(True)`     | Ensure that `batch_size` is set to appropriate value (for example: 64) |
| Distributed prediction                                                   | `lenet.set(test_algo="allreduce")`                                                                                                       |                                                                        |
| Distributed synchronous training                                         | `lenet.set(train_algo="allreduce_parallel_batches", parallel_batches=num_cluster_cores)`                                                 | Ensure that `batch_size` is set to appropriate value (for example: 64) |

Here are high-level guidelines to train very deep models on GPU with Keras2DML (and Caffe2DML):

1. If there exists at least one layer/operator that does not fit on the device, please allow SystemML's optimizer to perform operator placement based on the memory estimates `sysml_model.setGPU(True)`.
2. If each individual layer/operator fits on the device but not the entire network with a batch size of 1, then 
- Rely on SystemML's GPU Memory Manager to perform automatic eviction (recommended): `sysml_model.setGPU(True) # Optional: .setForceGPU(True)`
- Or enable Nvidia's Unified Memory:  `sysml_model.setConfigProperty('sysml.gpu.memory.allocator', 'unified_memory')`
3. If the entire neural network does not fit in the GPU memory with the user-specified `batch_size`, but fits in the GPU memory with `local_batch_size` such that `1 << local_batch_size < batch_size`, then
- Use either of the above two options.
- Or enable `train_algo` that performs multiple forward-backward pass with batch size `local_batch_size`, aggregate gradients and finally updates the model: 
```python
sysml_model = Keras2DML(spark, keras_model)
sysml_model.set(train_algo="looped_minibatch", parallel_batches=int(batch_size/local_batch_size))
sysml_model.setGPU(True).setForceGPU(True)
sysml_model.fit(X, y, batch_size=local_batch_size)
```
- Or add `int(batch_size/local_batch_size)` GPUs and perform single-node multi-GPU training with batch size `local_batch_size`:
```python
sysml_model = Keras2DML(spark, keras_model)
sysml_model.set(train_algo="allreduce_parallel_batches", parallel_batches=int(batch_size/local_batch_size))
sysml_model.setGPU(True).setForceGPU(True)
sysml_model.fit(X, y, batch_size=local_batch_size)
```

#### Design document of Keras2DML.

Keras2DML internally utilizes the existing [Caffe2DML](beginners-guide-caffe2dml) backend to convert Keras models into DML. Keras models are 
parsed and translated into Caffe prototext and caffemodel files which are then piped into Caffe2DML. 

Keras models are parsed based on their layer structure and corresponding weights and translated into the relative Caffe layer and weight
configuration. Be aware that currently this is a translation into Caffe and there will be loss of information from keras models such as 
intializer information, and other layers which do not exist in Caffe. 

Read the [Caffe2DML's reference guide](http://apache.github.io/systemml/reference-guide-caffe2dml) for the design documentation. 