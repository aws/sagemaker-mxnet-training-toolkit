#  Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os

import numpy as np
import pytest
from sagemaker.mxnet import MXNetModel
from sagemaker.utils import sagemaker_timestamp

from test.integration import EI_SUPPORTED_REGIONS, RESOURCE_PATH
from timeout import timeout_and_delete_endpoint_by_name

logger = logging.getLogger(__name__)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)
logging.getLogger('session.py').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)


@pytest.fixture(autouse=True)
def skip_if_no_accelerator(accelerator_type):
    if accelerator_type is None:
        pytest.skip('Skipping because accelerator type was not provided')


@pytest.fixture(autouse=True)
def skip_if_non_supported_ei_region(region):
    if region not in EI_SUPPORTED_REGIONS:
        pytest.skip('EI is not supported in {}'.format(region))


@pytest.fixture
def pretrained_model_data(region):
    return 's3://sagemaker-sample-data-{}/mxnet/model/resnet/resnet_50.tar.gz'.format(region)


@pytest.mark.skip_if_non_supported_ei_region
@pytest.mark.skip_if_no_accelerator
def test_deploy_elastic_inference(pretrained_model_data, ecr_image, sagemaker_session,
                                  instance_type, accelerator_type):
    default_handler_path = os.path.join(RESOURCE_PATH, 'default_handlers')
    endpoint_name = 'test-mxnet-ei-deploy-model-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name=endpoint_name,
                                             sagemaker_session=sagemaker_session, minutes=20):
        entry_script = os.path.join(default_handler_path, 'code', 'empty_module.py')
        model = MXNetModel(model_data=pretrained_model_data,
                           entry_point=entry_script,
                           role='SageMakerRole',
                           image=ecr_image,
                           sagemaker_session=sagemaker_session)

        logger.info('deploying model to endpoint: {}'.format(endpoint_name))
        predictor = model.deploy(initial_instance_count=1,
                                 instance_type=instance_type,
                                 accelerator_type=accelerator_type,
                                 endpoint_name=endpoint_name)

        random_input = np.zeros(shape=(1, 3, 224, 224))

        predict_response = predictor.predict(random_input.tolist())
        assert predict_response
