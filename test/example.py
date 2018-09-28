import os

from keras.datasets import mnist
from sagemaker.mxnet import MXNet

IMAGE_NAME = '520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.3.0-cpu-py2-script-mode'

resource_path = os.path.join(os.path.dirname(__file__), 'resources', 'mnist')
script_path = os.path.join(resource_path, 'keras_script_mode.py')

mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
           train_instance_type='local', hyperparameters={'epochs': 1}, image_name=IMAGE_NAME)

mnist.load_data(path='mnist.npz')
train = mx.sagemaker_session.upload_data(path=os.path.expanduser('~/.keras/datasets/'),
                                         key_prefix='mxnet_keras_mnist')

mx.fit({'train': train})
