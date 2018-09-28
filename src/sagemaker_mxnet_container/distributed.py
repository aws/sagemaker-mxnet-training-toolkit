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

from contextlib import contextmanager
import logging
import os
import socket
import subprocess

from retrying import retry

logger = logging.getLogger(__name__)


class DefaultParameterServer():
    ROLES = ['worker', 'scheduler', 'server']

    def __init__(self, hosts, ps_port='8000', ps_verbose='0'):
        self.scheduler = self.scheduler_host(hosts)
        self.hosts = hosts
        self.ps_port = ps_port
        self.ps_verbose = ps_verbose

    def scheduler_host(self, hosts):
        return sorted(hosts)[0]

    @contextmanager
    def setup(self, current_host):
        if len(self.hosts) == 1:
            logger.info('Only one host, so skipping setting up a parameter server')
        else:
            self._verify_hosts(self.hosts)

            logger.info('Starting distributed training task')
            if self.scheduler == current_host:
                self._run_mxnet_process('scheduler')
            self._run_mxnet_process('server')
            os.environ.update(self._env_vars_for_role('worker'))

        yield

    @retry(stop_max_delay=1000 * 60 * 15, wait_exponential_multiplier=100,
           wait_exponential_max=30000)
    def _host_lookup(self, host):
        return socket.gethostbyname(host)

    def _env_vars_for_role(self, role):
        if role in self.ROLES:
            return {
                'DMLC_NUM_WORKER': str(len(self.hosts)),
                'DMLC_NUM_SERVER': str(len(self.hosts)),
                'DMLC_ROLE': role,
                'DMLC_PS_ROOT_URI': self._host_lookup(self.scheduler),
                'DMLC_PS_ROOT_PORT': self.ps_port,
                'PS_VERBOSE': self.ps_verbose,
            }

        raise ValueError('Unexpected role: {}'.format(role))

    def _run_mxnet_process(self, role):
        role_env = os.environ.copy()
        role_env.update(self._env_vars_for_role(role))
        return subprocess.Popen("python -c 'import mxnet'", shell=True, env=role_env).pid

    def _verify_hosts(self, hosts):
        for host in hosts:
            self._host_lookup(host)
