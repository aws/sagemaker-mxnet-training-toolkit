=========================
SageMaker MXNet Container
=========================

SageMaker MXNet Container is an open-source library for making Docker images for using MXNet on Amazon SageMaker.
For information on running MXNet jobs on Amazon SageMaker, please refer to the `SageMaker Python SDK documentation <https://github.com/aws/sagemaker-python-sdk>`__.

-----------------
Table of Contents
-----------------
.. contents::
    :local:

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have installed all of the following prerequisites on your development machine:

- `Docker <https://www.docker.com/>`__
- For GPU testing: `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__

Recommended
^^^^^^^^^^^

-  A Python environment management tool (e.g. `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building Images
---------------

The Dockerfiles in this repository are intended to be used for building Docker images to run training jobs and inference endpoints on `Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__.

The current master branch of this repository contains Dockerfiles and support code for MXNet versions 1.3.0 and higher.
For MXNet versions 0.12.1-1.2.1, check out v1.0.0 of this repository.

For each supported MXNet version, Dockerfiles can be found for each processor type (i.e. CPU and GPU).
For MXNet versions 0.12.1 and 1.0.0, there are separate Dockerfiles for each Python version as well.

All images are tagged with <mxnet_version>-<processor>-<python_version> (e.g. 1.3.0-cpu-py3).

MXNet 1.1.0 and higher
~~~~~~~~~~~~~~~~~~~~~~

For these MXNet versions, there is one set of Dockerfiles for each version.
They install the SageMaker-specific support code found in this repository.

Before building these images, you need to have two files already saved locally.
The first is a pip-installable binary of the MXNet library.
This can be something you compile from source or `download from PyPI <https://pypi.org/project/mxnet/#files>`__.

The second is a pip-installable binary of this repository.
To create the SageMaker MXNet Container Python package:

::

    # Create the binary
    git clone https://github.com/aws/sagemaker-mxnet-container.git
    cd sagemaker-mxnet-container
    python setup.py sdist

    # Copy your Python package to the appropriate "final" Dockerfile directory
    cp dist/sagemaker_mxnet_container-<package_version>.tar.gz docker/<mxnet_version>/final

Once you have those binaries, you can then build the image.
The Dockerfiles expect two build arguments:

- ``py_version``: the Python version.
- ``framework_installable``: the path to the MXNet binary

To build an image:

::

    # All build instructions assume you're building from the same directory as the Dockerfile.

    # CPU
    docker build -t preprod-mxnet:<tag> \
                 --build-arg py_version=<python_version> \
                 --build-arg framework_installable=<mxnet_binary> \
                 -f Dockerfile.cpu .

    # GPU
    docker build -t preprod-mxnet:<tag> \
                 --build-arg py_version=<python_version> \
                 --build-arg framework_installable=<mxnet_binary> \
                 -f Dockerfile.gpu .

Don't forget the period at the end of the command!

::

    # Example

    # CPU
    docker build -t preprod-mxnet:1.1.0-cpu-py3 --build-arg py_version=3
    --build-arg framework_installable=mxnet-1.1.0-py2.py3-none-manylinux1_x86_64.whl -f Dockerfile.cpu .

    # GPU
    docker build -t preprod-mxnet:1.1.0-gpu-py3 --build-arg py_version=3
    --build-arg framework_installable=mxnet-1.1.0-py2.py3-none-manylinux1_x86_64.whl -f Dockerfile.gpu .


MXNet 0.12.1 and 1.0.0
~~~~~~~~~~~~~~~~~~~~~~

For these MXNet versions, there are "base" and "final" Dockerfiles for each image.
The "base" Dockerfile installs MXNet and its necessary dependencies.
The "final" Dockerfile installs the SageMaker-specific support code found in this repository.

Base Images
^^^^^^^^^^^

To build a "base" image:

::

    # All build instructions assume you're building from the same directory as the Dockerfile.

    # CPU
    docker build -t mxnet-base:<mxnet_version>-cpu-<python_version> -f Dockerfile.cpu .

    # GPU
    docker build -t mxnet-base:<mxnet_version>-gpu-<python_version> -f Dockerfile.gpu .

::

    # Example

    # CPU
    docker build -t mxnet-base:0.12.1-cpu-py2 -f Dockerfile.cpu .

    # GPU
    docker build -t mxnet-base:0.12.1-gpu-py2 -f Dockerfile.gpu .

Final Images
^^^^^^^^^^^^

All "final" Dockerfiles assume the "base" image has already been built.
Make sure the "base" image is named and tagged as expected by the "final" Dockerfile.

In addition, the "final" Dockerfiles require a pip-installable binary of this repository.
To create the SageMaker MXNet Container Python package:

::

    # Create the binary
    git clone -b v1.0.0 https://github.com/aws/sagemaker-mxnet-container.git
    cd sagemaker-mxnet-container
    python setup.py sdist

    # Copy your Python package to the appropriate "final" Dockerfile directory
    cp dist/sagemaker_mxnet_container-<package_version>.tar.gz docker/<mxnet_version>/final

To build a "final" image:

::

    # All build instructions assumes you're building from the same directory as the Dockerfile.

    # CPU
    docker build -t <image_name>:<tag> -f Dockerfile.cpu .

    # GPU
    docker build -t <image_name>:<tag> -f Dockerfile.gpu .

::

    # Example

    # CPU
    docker build -t preprod-mxnet:0.12.1-cpu-py2 -f Dockerfile.cpu .

    # GPU
    docker build -t preprod-mxnet:0.12.1-gpu-py2 -f Dockerfile.gpu .


Running the tests
-----------------

Running the tests requires installation of the SageMaker MXNet Container code and its test dependencies.

::

    git clone https://github.com/aws/sagemaker-mxnet-container.git
    cd sagemaker-mxnet-container
    pip install -e .[test]

Tests are defined in `test/ <https://github.com/aws/sagemaker-mxnet-containers/tree/master/test>`__ and include unit and integration tests.
The integration tests include both running the Docker containers locally and running them on SageMaker.
The tests are compatible with only the Docker images built by Dockerfiles in the current branch.
If you want to run tests for MXNet versions 1.2.1 or below, please use the v1.0.0 tests.

All test instructions should be run from the top level directory

Unit Tests
~~~~~~~~~~

To run unit tests:

::

    pytest test/unit

Local Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~

Running local integration tests require `Docker <https://www.docker.com/>`__ and `AWS credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the integration tests make calls to a couple AWS services.
Local integration tests on GPU require `nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__.
You Docker image must also be built in order to run the tests against it.

Local integration tests use the following pytest arguments:

- ``docker-base-name``: the Docker image's repository. Defaults to 'preprod-mxnet'.
- ``framework-version``: the MXNet version. Defaults to the latest supported version.
- ``py-version``: the Python version. Defaults to '3'.
- ``processor``: CPU or GPU. Defaults to 'cpu'.
- ``tag``: the Docker image's tag. Defaults to <mxnet_version>-<processor>-py<py-version>

To run local integration tests:

::

    pytest test/integration/local --docker-base-name <your_docker_image> \
                                  --tag <your_docker_image_tag> \
                                  --py-version <2_or_3> \
                                  --framework-version <mxnet_version> \
                                  --processor <cpu_or_gpu>

::

    # Example
    pytest test/integration/local --docker-base-name preprod-mxnet \
                                  --tag 1.3.0-cpu-py3 \
                                  --py-version 3 \
                                  --framework-version 1.3.0 \
                                  --processor cpu

SageMaker Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker integration tests require your Docker image to be within an `Amazon ECR repository <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html>`__.

SageMaker integration tests use the following pytest arguments:

- ``docker-base-name``: the Docker image's `ECR repository namespace <https://docs.aws.amazon.com/AmazonECR/latest/userguide/Repositories.html>`__.
- ``framework-version``: the MXNet version. Defaults to the latest supported version.
- ``py-version``: the Python version. Defaults to '3'.
- ``processor``: CPU or GPU. Defaults to 'cpu'.
- ``tag``: the Docker image's tag. Defaults to <mxnet_version>-<processor>-py<py-version>
- ``aws-id``: your AWS account ID.
- ``instance-type``: the specified `Amazon SageMaker Instance Type <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ that the tests will run on.
  Defaults to 'ml.c4.xlarge' for CPU and 'ml.p2.xlarge' for GPU.

To run SageMaker integration tests:

::

    pytest test/integration/sagmaker --aws-id <your_aws_id> \
                                     --docker-base-name <your_docker_image> \
                                     --instance-type <amazon_sagemaker_instance_type> \
                                     --tag <your_docker_image_tag> \

::

    # Example
    pytest test/integration/sagemaker --aws-id 12345678910 \
                                      --docker-base-name preprod-mxnet \
                                      --instance-type ml.m4.xlarge \
                                      --tag 1.3.0-cpu-py3

Amazon Elastic Inference with MXNet in SageMaker
------------------------------------------------
`Amazon Elastic Inference <https://aws.amazon.com/machine-learning/elastic-inference/>`__ allows you to to attach
low-cost GPU-powered acceleration to Amazon EC2 and Amazon SageMaker instances to reduce the cost running deep
learning inference by up to 75%. Currently, Amazon Elastic Inference supports TensorFlow, Apache MXNet, and ONNX
models, with more frameworks coming soon.

Support for using MXNet with Amazon Elastic Inference in SageMaker is supported in the public SageMaker MXNet containers.

* For information on how to use the Python SDK to create an endpoint with Amazon Elastic Inference and MXNet in SageMaker, see `Deploying MXNet Models <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/mxnet#deploying-mxnet-models>`__.
* For information on how Amazon Elastic Inference works, see `How EI Works <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html#ei-how-it-works>`__.
* For more information in regards to using Amazon Elastic Inference in SageMaker, see `Amazon SageMaker Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`__.
* For notebook examples on how to use Amazon Elastic Inference with MXNet through the Python SDK in SageMaker, see `EI Sample Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html#ei-intro-sample-nb>`__.

Building the SageMaker Elastic Inference MXNet container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Amazon Elastic Inference is designed to be used with AWS enhanced versions of TensorFlow serving or Apache MXNet. These enhanced
versions of the frameworks are automatically built into containers when you use the Amazon SageMaker Python SDK, or you can
download them as binary files and import them into your own Docker containers. The enhanced MXNet binaries are available on Amazon S3 at https://s3.console.aws.amazon.com/s3/buckets/amazonei-apachemxnet.

The SageMaker MXNet containers with Amazon Elastic Inference support were built utilizing the
same instructions listed `above <https://github.com/aws/sagemaker-mxnet-container#building-images>`__ with the
`CPU Dockerfile <https://github.com/aws/sagemaker-mxnet-container/blob/master/docker/1.3.0/final/Dockerfile.cpu>`__ starting at MXNet version 1.3.0 and above.

The only difference is that the enhanced version of MXNet was passed in for the ``framework_installable`` build-arg.

::

    # Example

    # EI
    docker build -t preprod-mxnet-ei:1.3.0-cpu-py3 --build-arg py_version=3
    --build-arg framework_installable=amazonei_mxnet-1.3.0-py2.py3-none-manylinux1_x86_64.whl -f Dockerfile.cpu .


* For information about downloading and installing the enhanced binary for Apache MXNet, see `Install Amazon EI Enabled Apache MXNet <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ei-mxnet.html#ei-apache>`__.
* For information on which versions of MXNet is supported for Elastic Inference within SageMaker, see `MXNet SageMaker Estimators <https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators>`__.

Contributing
------------

Please read `CONTRIBUTING.md <https://github.com/aws/sagemaker-mxnet-containers/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull requests to us.

License
-------

SageMaker MXNet Containers is licensed under the Apache 2.0 License.
It is copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
