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
import time
from time import sleep

import pytest
from sagemaker import utils
from sagemaker.mxnet.estimator import MXNet

from test.integration import RESOURCE_PATH
from timeout import timeout

DATA_PATH = os.path.join(RESOURCE_PATH, "mnist")
SCRIPT_PATH = os.path.join(DATA_PATH, "mnist_gluon_basic_hook_demo.py")


@pytest.mark.skip_py2_containers
def test_training(sagemaker_session, ecr_image, instance_type, instance_count):

    from smexperiments.experiment import Experiment
    from smexperiments.trial import Trial
    from smexperiments.trial_component import TrialComponent

    sm_client = sagemaker_session.sagemaker_client

    experiment_name = "mxnet-container-integ-test-{}".format(int(time.time()))

    experiment = Experiment.create(
        experiment_name=experiment_name,
        description="Integration test experiment from sagemaker-mxnet-container",
        sagemaker_boto_client=sm_client,
    )

    trial_name = "mxnet-container-integ-test-{}".format(int(time.time()))
    trial = Trial.create(
        experiment_name=experiment_name, trial_name=trial_name, sagemaker_boto_client=sm_client
    )

    hyperparameters = {
        "random_seed": True,
        "num_steps": 50,
        "smdebug_path": "/opt/ml/output/tensors",
        "epochs": 1,
    }

    mx = MXNet(
        entry_point=SCRIPT_PATH,
        role="SageMakerRole",
        train_instance_count=instance_count,
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_name=ecr_image,
        hyperparameters=hyperparameters,
    )

    training_job_name = utils.unique_name_from_base("test-mxnet-image")

    # create a training job and wait for it to complete
    with timeout(minutes=15):
        prefix = "mxnet_mnist_gluon_basic_hook_demo/{}".format(utils.sagemaker_timestamp())
        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(DATA_PATH, "train"), key_prefix=prefix + "/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(DATA_PATH, "test"), key_prefix=prefix + "/test"
        )

        mx.fit({"train": train_input, "test": test_input}, job_name=training_job_name, wait=False)

    training_job = sm_client.describe_training_job(TrainingJobName=training_job_name)
    training_job_arn = training_job["TrainingJobArn"]

    # verify trial component auto created from the training job
    trial_component_summary = None
    attempts = 0
    while True:
        trial_components = list(
            TrialComponent.list(source_arn=training_job_arn, sagemaker_boto_client=sm_client)
        )

        if len(trial_components) > 0:
            trial_component_summary = trial_components[0]
            break

        if attempts < 10:
            attempts += 1
            sleep(500)

    assert trial_component_summary is not None

    trial_component = TrialComponent.load(
        trial_component_name=trial_component_summary.trial_component_name,
        sagemaker_boto_client=sm_client,
    )

    # associate the trial component with the trial
    trial.add_trial_component(trial_component)

    # cleanup
    trial.remove_trial_component(trial_component_summary.trial_component_name)
    trial_component.delete()
    trial.delete()
    experiment.delete()
