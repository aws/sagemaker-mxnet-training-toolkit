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

import sys
import json


def train(hyperparameters, num_cpus, num_gpus, channel_input_dirs, **kwargs):
    _assert_py_version(hyperparameters)


class DummyModel(object):
      def predict(self, data):
          return data
  
  
def model_fn(model_dir):
  return DummyModel()


def transform_fn(model, data, input_content_type, output_content_type):
    print('input object: {}'.format(data))
    _assert_py_version(json.loads(data))
    return data, "application/json"


def _assert_py_version(version_dict):
    print('python version info: {}'.format(sys.version_info))
    major_v, minor_v = sys.version_info[0:2]
    expected_major_v = int(version_dict['py_major_version'])
    minimum_minor_v = int(version_dict['py_minimum_minor_version'])
     
    assert major_v == expected_major_v, 'python major version must be {}'.format(expected_major_v)
    assert minor_v >= minimum_minor_v, 'python minor version must be >= {}'.format(minimum_minor_v)

