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

from mock import call, patch

from sagemaker_mxnet_container.distributed import DefaultParameterServer

SCHEDULER = 'host-1'
SINGLE_HOST_LIST = [SCHEDULER]
MULTIPLE_HOSTS_LIST = ['host-2', SCHEDULER, 'host-3']

IP_ADDRESS = '0.0.0.0000'
DEFAULT_PORT = '8000'
DEFAULT_VERBOSITY = '0'
BASE_ENV_VARS = {
    'DMLC_NUM_WORKER': str(len(MULTIPLE_HOSTS_LIST)),
    'DMLC_NUM_SERVER': str(len(MULTIPLE_HOSTS_LIST)),
    'DMLC_PS_ROOT_URI': IP_ADDRESS,
    'DMLC_PS_ROOT_PORT': DEFAULT_PORT,
    'PS_VERBOSE': DEFAULT_VERBOSITY,
}

MXNET_COMMAND = "python -c 'import mxnet'"


def test_init_for_single_host_with_defaults():
    server = DefaultParameterServer(SINGLE_HOST_LIST)

    assert server.hosts == SINGLE_HOST_LIST
    assert server.scheduler == SCHEDULER
    assert server.ps_port == DEFAULT_PORT
    assert server.ps_verbose == DEFAULT_VERBOSITY


def test_init_for_multiple_hosts_and_ps_options():
    port = '8080'
    verbosity = '1'
    server = DefaultParameterServer(MULTIPLE_HOSTS_LIST, ps_port=port, ps_verbose=verbosity)

    assert server.hosts == MULTIPLE_HOSTS_LIST
    assert server.scheduler == SCHEDULER
    assert server.ps_port == port
    assert server.ps_verbose == verbosity


def test_scheduler_host():
    server = DefaultParameterServer(MULTIPLE_HOSTS_LIST)
    assert server._scheduler_host() == SCHEDULER


@patch('sagemaker_mxnet_container.distributed.DefaultParameterServer._run_mxnet_process')
def test_setup_for_single_host(run_process):
    server = DefaultParameterServer(SINGLE_HOST_LIST)
    server.setup(SCHEDULER)

    run_process.assert_not_called()


@patch('os.environ', {})
@patch('subprocess.Popen')
@patch('sagemaker_mxnet_container.distributed.DefaultParameterServer._host_lookup')
@patch('sagemaker_mxnet_container.distributed.DefaultParameterServer._verify_hosts')
def test_setup_for_distributed_scheduler(verify_hosts, host_lookup, popen):
    host_lookup.return_value = IP_ADDRESS

    server = DefaultParameterServer(MULTIPLE_HOSTS_LIST)
    with server.setup(SCHEDULER):
        pass

    verify_hosts.assert_called_with(MULTIPLE_HOSTS_LIST)

    scheduler_env = BASE_ENV_VARS.copy()
    scheduler_env.update({'DMLC_ROLE': 'scheduler'})

    server_env = BASE_ENV_VARS.copy()
    server_env.update({'DMLC_ROLE': 'server'})

    calls = [call(MXNET_COMMAND, shell=True, env=scheduler_env),
             call(MXNET_COMMAND, shell=True, env=server_env)]

    popen.assert_has_calls(calls)


@patch('os.environ', {})
@patch('subprocess.Popen')
@patch('sagemaker_mxnet_container.distributed.DefaultParameterServer._host_lookup')
@patch('sagemaker_mxnet_container.distributed.DefaultParameterServer._verify_hosts')
def test_setup_for_distributed_worker(verify_hosts, host_lookup, popen):
    host_lookup.return_value = IP_ADDRESS

    server = DefaultParameterServer(MULTIPLE_HOSTS_LIST)
    with server.setup('host-2'):
        pass

    verify_hosts.assert_called_with(MULTIPLE_HOSTS_LIST)

    server_env = BASE_ENV_VARS.copy()
    server_env.update({'DMLC_ROLE': 'server'})

    popen.assert_called_once_with(MXNET_COMMAND, shell=True, env=server_env)
