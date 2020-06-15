================================
SageMaker MXNet Training Toolkit
================================

SageMaker MXNet Training Toolkit is an open-source library for using MXNet to train models on Amazon SageMaker.
For inference, see `SageMaker MXNet Inference Toolkit <https://github.com/aws/sagemaker-mxnet-serving-container>`__.
For the Dockerfiles used for building SageMaker MXNet Containers, see `AWS Deep Learning Containers <https://github.com/aws/deep-learning-containers>`__.
For information on running MXNet jobs on Amazon SageMaker, please refer to the `SageMaker Python SDK documentation <https://github.com/aws/sagemaker-python-sdk>`__.


Contributing
------------

Please read `CONTRIBUTING.md <https://github.com/aws/sagemaker-mxnet-training-toolkit/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull requests to us.

Testing
-------

Set up a virtual environment for testing.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the multiple ways to setup a virtual environment

::

   # use a package virtualenv
   # create a virtualenv
   virtualenv -p python3 <name of env>
   # activate the virtualenv
   source <name of env>/bin/activate

Install requirements
~~~~~~~~~~~~~~~~~~~~

::

   pip install -r requirements.txt

Install sagemaker-mxnet-training-toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   python setup.py sdist
   pip install dist/sagemaker_mxnet_training*.tar.gz

Test locally
~~~~~~~~~~~~

To run specific test

::

   pytest test/unit/test_training.py::test_train_for_distributed_scheduler

To run an entire file

::

   pytest test/unit/test_training.py

To run all tests within a folder

::

   pytest test/unit

License
-------

SageMaker MXNet Training Toolkit is licensed under the Apache 2.0 License.
It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
