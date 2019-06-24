# Changelog

## v3.0.6 (2019-06-24)

### Bug fixes and other changes

 * remove JSON syntax error from buildspec-release.yml

## v3.0.5 (2019-06-24)

### Bug fixes and other changes

 * Updating SageMaker Containers to the latest version

## v3.0.4 (2019-06-19)

### Bug fixes and other changes

 * pin sagemaker in test dependencies

## v3.0.3 (2019-06-18)

### Bug fixes and other changes

 * fix local GPU test command in buildspec-release.yml
 * explicitly set lower bound for botocore version
 * parametrize Python version and processor type in integ tests
 * add hyperparameter tuning integ test
 * flesh out SageMaker training integ tests

## v3.0.2 (2019-06-10)

### Bug fixes and other changes

 * skip GPU test in regions with limited p2s

## v3.0.1 (2019-06-07)

### Bug fixes and other changes

 * use correct variable name in release buildspec
 * adjust parallelism in buildspec-release.yml
 * Add release buildspec
 * Remove outdated information from README

## v3.0.0

Compatible with MXNet version 1.4.0.

This version of the SageMaker MXNet container only accounts for training.

Serving support is now split and can be found here: https://github.com/aws/sagemaker-mxnet-serving-container

## v2.0.0

Compatible with MXNet version 1.3.0

This version of MXNet introduced an updated training script format. See https://sagemaker.readthedocs.io/en/v1.22.0/using_mxnet.html#preparing-the-mxnet-training-script for more information.

## v1.0.0

Compatible with MXNet versions 0.12.1-1.2.1.
