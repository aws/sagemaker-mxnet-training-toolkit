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

   pip install --upgrade .[test]


Local Test
~~~~~~~~~~

To run specific test

::

   tox -- -k test/unit/test_training.py::test_train_for_distributed_scheduler

To run an entire file

::

   tox -- test/unit/test_training.py

To run all tests within a folder [e.g. integration/local/]

Note: To run integration tests locally, one needs to build an image. To trigger image build, use `-B` flag.

::

   tox -- test/integration/local
   
You can also run them in parallel:

::

   tox -- -n auto test/integration/local

To run for specific interpreter [Python environment], use the ``-e`` flag

::

   tox -e py37 -- test/unit/test_training.py

Remote Test
~~~~~~~~~~~

Make sure to provide AWS account ID, Region, Docker base name & Tag.
Docker Registry is composed of (aws_id, region)
Image URI is composed of (docker_registry, docker_base_name, tag)

Resulting Image URI is composed as: ``{aws_id}.dkr.ecr.{region}.amazonaws.com/{docker_base_name}:{tag}``

::

    tox -- --aws-id <aws_id> --region <region> --docker-base-name <docker_base_name> --tag <tag> test/integration/sagemaker

For more details, refer `conftest.py <test/conftest.py>`_

License
-------

SageMaker MXNet Training Toolkit is licensed under the Apache 2.0 License.
It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
