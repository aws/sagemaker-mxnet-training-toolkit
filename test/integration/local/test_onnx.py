#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import numpy
from sagemaker.mxnet import MXNet

import local_mode
from test.integration import NUM_MODEL_SERVER_WORKERS, RESOURCE_PATH


def test_onnx_export_and_import(docker_image, sagemaker_local_session, local_instance_type,
                                framework_version):
    script_path = os.path.join(RESOURCE_PATH, 'onnx', 'code', 'onnx_export_import.py')
    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type=local_instance_type, sagemaker_session=sagemaker_local_session,
               image_name=docker_image, framework_version=framework_version)

    input_path = 'file://{}'.format(os.path.join(RESOURCE_PATH, 'onnx', 'model'))
    mx.fit({'train': input_path})

    input = numpy.zeros(shape=(1, 1, 28, 28))

    with local_mode.lock():
        try:
            model = mx.create_model(model_server_workers=NUM_MODEL_SERVER_WORKERS)
            predictor = model.deploy(1, local_instance_type)
            output = predictor.predict(input)
        finally:
            mx.delete_endpoint()

    # Check that there is a probability for each possible class in the prediction
    assert len(output[0]) == 10
