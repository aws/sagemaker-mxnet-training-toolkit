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

import container_support as cs
import inspect
import json
import logging
import os
import socket
import subprocess

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "model"

DEFAULT_MODEL_FILENAMES = {
    'symbol': 'model-symbol.json',
    'params': 'model-0000.params',
    'shapes': 'model-shapes.json',
}

UPCOMING_SCRIPT_MODE_WARNING = (
    '\033[1;33m'  # print warning in yellow
    'This required structure for training scripts will be '
    'deprecated with the next major release of MXNet images. '
    'The train() function will no longer be required; '
    'instead the training script must be able to be run as a standalone script. '
    'For more information, see https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/mxnet#updating-your-mxnet-training-script.'  # noqa: E501
    '\033[1;0m'
)


class MXNetTrainingEnvironment(cs.TrainingEnvironment):
    """ Configuration for single machine and distributed mxnet training.
    """

    def __init__(self, base_dir):
        super(MXNetTrainingEnvironment, self).__init__(base_dir)

        self._ps_verbose = int(self.hyperparameters.get('_ps_verbose', 0))
        self._ps_port = int(self.hyperparameters.get('_ps_port', 8000))
        self._scheduler_host = sorted(self.hosts)[0]
        self._scheduler_ip = host_lookup(self._scheduler_host)
        # Block until all host lookups succeed. Relies on retrying host_lookup.
        for host in self.hosts:
            host_lookup(host)

    @property
    def distributed(self):
        """ Returns True if this configuration defines a distributed learning task."""
        return len(self.hosts) > 1

    @property
    def current_host_scheduler(self):
        """ Returns True if this machine should be the mxnet parameter server scheduler."""
        return self._scheduler_host == self.current_host

    def env_vars_for_role(self, role):
        """ Returns environment variables for a python process to run as an
        mxnet parameter server process with the specified role.

        Args:
            role (str): One of "worker", "server", or "scheduler"
        """
        if role not in ["worker", "scheduler", "server"]:
            raise ValueError("Unexpected role {}".format(role))
        return {
            'DMLC_NUM_WORKER': str(len(self.hosts)),
            'DMLC_NUM_SERVER': str(len(self.hosts)),
            'DMLC_ROLE': role,
            'DMLC_PS_ROOT_URI': str(self._scheduler_ip),
            'DMLC_PS_ROOT_PORT': str(self._ps_port),
            'PS_VERBOSE': str(self._ps_verbose)
        }

    @property
    def kwargs_for_training(self):
        """ Returns a dictionary of key-word arguments for input to the user supplied
        module train function. """
        return {
            'hyperparameters': dict(self.hyperparameters),
            'input_data_config': dict(self.channels),
            'channel_input_dirs': dict(self.channel_dirs),
            'output_data_dir': self.output_data_dir,
            'model_dir': self.model_dir,
            'num_gpus': self.available_gpus,
            'num_cpus': self.available_cpus,
            'hosts': list(self.hosts),
            'current_host': self.current_host
        }

    def default_save(self, mod):
        """ Saves the specified mxnet module to ``self.model_dir``.

            This generates three files in ``self.model_dir``:

            - model-symbol.json      - The serialized module symbolic graph. Formed by
                            invoking ```module.symbol.save```
            - model-0000.params      - The serialized module parameters. Formed by
                            invoking ```module.save_params```
            - model-shapes.json - The serialized module input data shapes. A json list
                            of json data-shape objects. Each data-shape object
                            contains a string name and a list of integer dimensions.
            Args:
                mod : (mxnet.mod.Module) The module to save."""
        if not self.distributed or self.current_host_scheduler:
            mod.symbol.save(os.path.join(self.model_dir, DEFAULT_MODEL_FILENAMES['symbol']))
            mod.save_params(os.path.join(self.model_dir, DEFAULT_MODEL_FILENAMES['params']))
            signature = self._build_data_shape_signature(mod)
            with open(os.path.join(self.model_dir, DEFAULT_MODEL_FILENAMES['shapes']), 'w') as f:
                json.dump(signature, f)

    @classmethod
    def _build_data_shape_signature(cls, mod):
        """ Returns a list of data shape description dicts. Each element in the
        returned list is a dict with a 'name' key, mapping to a string name
        and a 'shape' key, mapping to a list of ints.
        """
        return [{"name": data_desc.name, "shape": [dim for dim in data_desc.shape]}
                for data_desc in mod.data_shapes]


@cs.retry(stop_max_delay=1000 * 60 * 15,
          wait_exponential_multiplier=100,
          wait_exponential_max=30000)
def host_lookup(host):
    """ Retrying host lookup on host """
    return socket.gethostbyname(host)


def _run_mxnet_process(role, mxnet_env):
    """ Runs an mxnet process for the specified role with the specified
    environment.

    Args:
        role (str): The mxnet process role.
        mxnet_env (MXNetEnvironment): The mxnet environment used to provide
        environment variables for the launched process.
    Returns:
        (int) The launched process id """

    role_env = os.environ.copy()
    role_env.update(mxnet_env.env_vars_for_role(role))
    return subprocess.Popen("python -c 'import mxnet'", shell=True, env=role_env).pid


def train(base_dir=MXNetTrainingEnvironment.BASE_DIRECTORY):
    """ Runs mxnet training on a user supplied module in either a local or distributed
    SageMaker environment.

    The user supplied module and its dependencies are downloaded from S3, and the module
    imported using a ``MXNetTrainingEnvironment`` instance.

    Training is invoked by calling a "train" function in the user supplied module.

    if the environment contains multiple hosts, then a distributed learning
    task is started. This function will, in addition to running the user supplied script
    as an mxnet parameter server worker process, launch an additional mxnet server
    process. If the host this process is executing on is designated as the scheduler, then
    this funciton will launch an mxnet scheduler parameter server process.

    Args:
        base_dir (str): The SageMaker container environment base directory.
    """
    logger.warning(UPCOMING_SCRIPT_MODE_WARNING)

    mxnet_env = MXNetTrainingEnvironment(base_dir)
    logger.info("MXNetTrainingEnvironment: {}".format(repr(mxnet_env.__dict__)))

    if mxnet_env.user_script_archive.lower().startswith('s3://'):
        mxnet_env.download_user_module()

    logger.info("Starting distributed training task")
    if mxnet_env.current_host_scheduler:
        _run_mxnet_process("scheduler", mxnet_env)
    _run_mxnet_process("server", mxnet_env)
    os.environ.update(mxnet_env.env_vars_for_role("worker"))

    user_module = mxnet_env.import_user_module()
    train_args = inspect.getargspec(user_module.train)

    # avoid forcing our callers to specify **kwargs in their function
    # signature. If they have **kwargs we still pass all the args, but otherwise
    # we will just pass what they ask for.
    if train_args.keywords is None:
        kwargs_to_pass = {}
        for arg in train_args.args:
            if arg != "self" and arg in mxnet_env.kwargs_for_training:
                kwargs_to_pass[arg] = mxnet_env.kwargs_for_training[arg]
    else:
        kwargs_to_pass = mxnet_env.kwargs_for_training

    model = user_module.train(**kwargs_to_pass)
    if model:
        if hasattr(user_module, 'save'):
            user_module.save(model, mxnet_env.model_dir)
        else:
            mxnet_env.default_save(model)

    mxnet_env.write_success_file()
