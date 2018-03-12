from __future__ import print_function

from sagemaker import fw_utils
from sagemaker.utils import sagemaker_timestamp
import docker_utils
import utils
import numpy as np
import os

def test_linear_regression(docker_image, sagemaker_session, opt_ml, processor):
    resource_path = 'test/resources/linear_regression'

    # create training data
    train_data = np.random.uniform(0, 1, [1000, 2])
    train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(1000)])

    # eval data... repeat so there's enough to cover multicpu/gpu contexts
    eval_data = np.array([[7, 2], [6, 10], [12, 2]]).repeat(32, 0)
    eval_label = np.array([11, 26, 16]).repeat(32, 0)

    # save training data
    for path in ['training', 'evaluation']:
        os.makedirs(os.path.join(opt_ml, 'input', 'data', path))
    np.savetxt(os.path.join(opt_ml, 'input/data/training/train_data.txt.gz'), train_data)
    np.savetxt(os.path.join(opt_ml, 'input/data/training/train_label.txt.gz'), train_label)
    np.savetxt(os.path.join(opt_ml, 'input/data/evaluation/eval_data.txt.gz'), eval_data)
    np.savetxt(os.path.join(opt_ml, 'input/data/evaluation/eval_label.txt.gz'), eval_label)

    s3_source_archive = fw_utils.tar_and_upload_dir(session=sagemaker_session.boto_session,
                                bucket=sagemaker_session.default_bucket(),
                                s3_key_prefix=sagemaker_timestamp(),
                                script='linear_regression.py',
                                directory=resource_path)

    utils.create_config_files('linear_regression.py', s3_source_archive.s3_prefix, opt_ml)
    os.makedirs(os.path.join(opt_ml, 'model'))

    docker_utils.train(docker_image, opt_ml, processor)

    for f in ['output/success', 'model/model-symbol.json', 'model/model-0000.params', 'model/model-shapes.json']:
        assert os.path.exists(os.path.join(opt_ml, f)), 'expected file not found: {}'.format(f)
