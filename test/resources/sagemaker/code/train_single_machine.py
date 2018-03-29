#  Copyright <YEAR> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import module_to_import  # noqa
import os
import mxnet
import numpy

logger = logging.getLogger(__name__)


def train(hyperparameters=None,
          input_data_config=None,
          channel_input_dirs=None,
          output_data_dir=None,
          model_dir=None,
          num_gpus=None,
          num_cpus=None,
          hosts=None,
          current_host=None):

    logger.info("hyperparameters: {}".format(hyperparameters))
    logger.info("input_data_config: {}".format(input_data_config))
    logger.info("channel_input_dirs: {}".format(channel_input_dirs))
    logger.info("num_gpus: {}".format(num_gpus))
    logger.info("num_cpus: {}".format(num_cpus))
    logger.info("hosts: {}".format(hosts))
    logger.info("current_host: {}".format(current_host))
    logger.info("expected_channel_names: {}".format(hyperparameters["expected_channel_names"]))

    expected_channel_names = hyperparameters["expected_channel_names"]

    assert set(input_data_config.keys()) == set(expected_channel_names)
    assert set(channel_input_dirs.keys()) == set(expected_channel_names)

    expected_gpus = hyperparameters["expected_gpus"]
    assert num_gpus == int(expected_gpus)

    expected_cpus = hyperparameters["expected_cpus"]
    assert num_cpus == int(expected_cpus)

    assert hosts
    assert current_host
    assert current_host in hosts

    # Do something with mxnet
    if num_gpus:
        ctx = mxnet.gpu(0)
    else:
        ctx = mxnet.cpu()
    arr_one = mxnet.ndarray.full((100, 100), 1, ctx=ctx)
    arr_two = mxnet.ndarray.full((100, 100), 2, ctx=ctx)
    arr_sum = (arr_one + arr_two).asnumpy()
    expected = mxnet.ndarray.full((100, 100), 3, ctx=ctx).asnumpy()
    logger.info("Expected Sum: {}".format(expected))
    logger.info("Received Sum: {}".format(arr_sum))
    logger.info("Equal?: {}".format(numpy.array_equal(expected, arr_sum)))
    assert numpy.array_equal(expected, arr_sum), "{} != {}".format(expected, arr_sum)

    # Assert that we can open the input files and they contain something
    for channel in input_data_config:
        assert open(os.path.join(channel_input_dirs[channel], "data.txt")).read(), "Cannot read {}".format(channel)

    # Write out to the model dir and the output_data_dir
    with open(os.path.join(model_dir, 'model.txt'), 'w') as f:
        f.write("model data")
    with open(os.path.join(output_data_dir, 'output.txt'), 'w') as f:
        f.write("output data")

    logger.info("Done test!")
