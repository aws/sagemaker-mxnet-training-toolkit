import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_mxnet_container',
    version='1.0.0',
    description='Open source library for creating MXNet containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-mxnet-containers',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

    # We don't declare our dependency on mxnet here because we build with
    # different packages for different variants (e.g. mxnet-mkl and mxnet-cu90).
    install_requires=['sagemaker-container-support >= 1.0.0, <2'],
    extras_require={
        'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock', 'sagemaker']
    },
)
