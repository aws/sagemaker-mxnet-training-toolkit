# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest
from sagemaker import utils
from sagemaker.mxnet.estimator import MXNet

from test.integration import RESOURCE_PATH
from timeout import timeout

CV_DATA_PATH = os.path.join(RESOURCE_PATH, 'cv')
CV_SCRIPT_PATH = os.path.join(CV_DATA_PATH, 'train_cifar.py')


@pytest.mark.skip_py2_containers
def test_cv_training(sagemaker_session, ecr_image, instance_type):

    cv = MXNet(entry_point=CV_SCRIPT_PATH,
               role='SageMakerRole',
               train_instance_count=1,
               train_instance_type=instance_type,
               sagemaker_session=sagemaker_session,
               image_name=ecr_image)

    with timeout(minutes=5):
        job_name = utils.unique_name_from_base('test-cv-image')
        cv.fit(job_name=job_name)
