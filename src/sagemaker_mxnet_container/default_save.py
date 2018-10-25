# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import os

SYMBOL_PATH = 'model-symbol.json'
PARAMS_PATH = 'model-0000.params'
SHAPES_PATH = 'model-shapes.json'


def save(model_dir, model):
    """Save an MXNet Module to a given location.

    This generates three files in the model directory:

    - model-symbol.json: The serialized module symbolic graph.
        Formed by invoking ``module.symbole.save``.
    - model-0000.params: The serialized module parameters.
        Formed by invoking ``module.save_params``.
    - model-shapes.json: The serialized module input data shapes in the form of a JSON list of
        JSON data-shape objects. Each data-shape object contains a string name and
        a list of integer dimensions.

    Args:
        model_dir (str): the directory for saving the model
        model (mxnet.mod.Module): the module to be saved
    """
    model.symbol.save(os.path.join(model_dir, SYMBOL_PATH))
    model.save_params(os.path.join(model_dir, PARAMS_PATH))

    signature = [{'name': data_desc.name, 'shape': [dim for dim in data_desc.shape]}
                 for data_desc in model.data_shapes]
    with open(os.path.join(model_dir, SHAPES_PATH), 'w') as f:
        json.dump(signature, f)
