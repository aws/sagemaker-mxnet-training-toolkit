# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker import utils
from sagemaker.mxnet.estimator import MXNet
import sagemaker as sage

from test.integration import RESOURCE_PATH
from timeout import timeout

DGL_DATA_PATH = os.path.join(RESOURCE_PATH, 'dgl-gcn')
DGL_SCRIPT_PATH = os.path.join(DGL_DATA_PATH, 'gcn.py')

def test_training(sagemaker_session, ecr_image, instance_type, instance_count):
    print(ecr_image)
    #ecr_image="397262719838.dkr.ecr.us-east-2.amazonaws.com/dgl-gpu-dlc:mxnet-1.5.0_dgl-0.4-sagemaker"
    #ecr_image="397262719838.dkr.ecr.us-east-2.amazonaws.com/dgl-cpu-dlc:mxnet-1.5.0_dgl-0.4-sagemaker"
    dgl = MXNet(entry_point=DGL_SCRIPT_PATH,
               role="arn:aws:iam::397262719838:role/service-role/AmazonSageMaker-ExecutionRole-20171213T134005",
               train_instance_count=1,
               train_instance_type=instance_type,
               sagemaker_session=sagemaker_session,
               image_name=ecr_image)

    with timeout(minutes=15):
        job_name = utils.unique_name_from_base('test-mxnet-dgl-image')
        dgl.fit(job_name=job_name)
