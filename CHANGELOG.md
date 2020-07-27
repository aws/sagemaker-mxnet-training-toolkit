# Changelog

## v4.2.1 (2020-07-27)

### Bug Fixes and Other Changes

 * remove the openssh addition

## v4.2.0.post1 (2020-07-08)

### Documentation Changes

 * update test readme

## v4.2.0.post0 (2020-06-25)

### Testing and Release Infrastructure

 * add issue templates

## v4.2.0 (2020-06-17)

### Features

 * Add MPI support for distributed training in MXNet

## v4.1.2 (2020-06-17)

### Bug Fixes and Other Changes

 * bump version of sagemaker-training for script entry point fix

### Documentation Changes

 * remove confusing information from the Readme.

### Testing and Release Infrastructure

 * Rename buildspec files.

## v4.1.1.post0 (2020-06-09)

### Testing and Release Infrastructure

 * Make docker folder read only, remove unused tests, rename test-toolkit/ -> test/.

## v4.1.1 (2020-05-12)

### Bug Fixes and Other Changes

 * Bump version of sagemaker-training for typing fix

### Testing and Release Infrastructure

 * remove unused scripts.

## v4.1.0 (2020-05-07)

### Features

 * add Python 3.7 support

## v4.0.0.post0 (2020-04-30)

### Testing and Release Infrastructure

 * use tox in buildspecs

## v4.0.0 (2020-04-27)

### Breaking Changes

 * Replace sagemaker-containers with sagemaker-training

## v3.1.20 (2020-04-07)

### Bug Fixes and Other Changes

 * update smdebug version

### Testing and Release Infrastructure

 * add requirements.txt integ test

## v3.1.19 (2020-04-06)

### Bug Fixes and Other Changes

 * fix for too many gpus

## v3.1.18 (2020-04-02)

### Bug Fixes and Other Changes

 * upgrade pillow etc. to fix safety issues

## v3.1.17 (2020-04-01)

### Bug Fixes and Other Changes

 * Add gluoncv

## v3.1.16 (2020-03-30)

### Bug Fixes and Other Changes

 * use sagemaker_mxnet_training<4 in Dockerfiles instead of pinning the version

## v3.1.15 (2020-03-23)

### Bug Fixes and Other Changes

 * upgrade sagemaker-containers to 2.8.2

## v3.1.14.post0 (2020-03-18)

### Testing and Release Infrastructure

 * do not run DLC tests when toolkit has changes.

## v3.1.14 (2020-03-16)

### Bug Fixes and Other Changes

 * update smdebug
 * update container build instructions with correct tar file name
 * Switch to pypi gluonnlp package

### Testing and Release Infrastructure

 * refactor toolkit tests.

## v3.1.13 (2020-03-12)

### Bug Fixes and Other Changes

 * install sm experiments on any python version 3.5 or above

## v3.1.12 (2020-03-11)

### Bug Fixes and Other Changes

 * Update smdebug to 0.7.0
 * move experiments import statements after python version check
 * install experiments sdk only for python 3.6
 * install SageMaker Python SDK into Python 3 images
 * Upgrade to latest version of sagemaker-experiments

## v3.1.11 (2020-02-20)

### Bug Fixes and Other Changes

 * copy all tests to test-toolkit folder.
 * Fix issue in installation of sagemaker-containers

## v3.1.10 (2020-02-17)

### Bug Fixes and Other Changes

 * update: Update license URL

## v3.1.9 (2020-02-13)

### Bug Fixes and Other Changes

 * update: Horovod Support with MXNet backend on DLC for MXNet 1.6.0

## v3.1.8 (2020-02-11)

### Bug Fixes and Other Changes

 * Use GluonNLP stable release tag

## v3.1.7 (2020-02-10)

### Bug Fixes and Other Changes

 * Use sagemaker-mxnet-training package from pypi.

## v3.1.6 (2020-02-05)

### Bug Fixes and Other Changes

 * Rename package to be 'sagemaker-mxnet-training'. Add automated release to PyPI.
 * Add GluonNLP
 * Update AWS-MXNet version to 1.6.0 - official release of 1.6
 * Update build artifacts
 * Misspelling of sagemaker_mxnet_container_*.tar.gz
 * Downgrade pip version and unpin awscli
 * Revert "Add GluonNLP 0.9 pre-release (#116)"
 * Add GluonNLP 0.9 pre-release
 * update smdebug version to 0.5.0.post0
 * update: Constrain package versions in 1.6.0 dockerfiles
 * Build context changes based on new requirements
 * Update README for 1.6.0 release
 * update copyright year in license header
 * Change build context for building MX Training Dockerfiles

### Testing and Release Infrastructure

 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v3.1.5 (2020-02-05)

### Bug Fixes and Other Changes

 * Rename package to be 'sagemaker-mxnet-training'. Add automated release to PyPI.
 * Add GluonNLP
 * Update AWS-MXNet version to 1.6.0 - official release of 1.6
 * Update build artifacts
 * Misspelling of sagemaker_mxnet_container_*.tar.gz
 * Downgrade pip version and unpin awscli
 * Revert "Add GluonNLP 0.9 pre-release (#116)"
 * Add GluonNLP 0.9 pre-release
 * update smdebug version to 0.5.0.post0
 * update: Constrain package versions in 1.6.0 dockerfiles
 * Build context changes based on new requirements
 * Update README for 1.6.0 release
 * update copyright year in license header
 * Change build context for building MX Training Dockerfiles
 * Release 1.6.0 Dockerfiles

### Testing and Release Infrastructure

 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v3.1.4 (2019-10-28)

### Bug fixes and other changes

 * use SageMaker Containers' ProcessRunner for executing the entry point
 * use regional endpoint for STS in builds

## v3.1.3 (2019-10-22)

### Bug fixes and other changes

 * update instance type region availability

## v3.1.2 (2019-08-17)

### Bug fixes and other changes

 * split cpu and gpu test commands.

## v3.1.1 (2019-08-16)

### Bug fixes and other changes

 * add flake8 in buildspec.yml and fix flake8.
 * fix placeholder name cpu-instance-type in buildspec-release.yml
 * Add eu-west-3, eu-north-1, sa-east-1 and ap-east-1 to the no-p2 regions.

## v3.1.0 (2019-07-17)

### Features

 * add MXNet 1.4.1 Dockerfiles

### Bug fixes and other changes

 * fix build directory in scripts/build_all.py
 * fix copy command in buildspec-release.yml

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
