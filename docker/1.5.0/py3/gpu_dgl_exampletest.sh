#!/bin/bash

GCN_EXAMPLE_DIR="./examples/mxnet"

fail() {
    echo FAIL: $@
    exit -1
}

export DGLBACKEND=mxnet
export CUDA_VISIBLE_DEVICES=0
dev=0
export DGL_DOWNLOAD_DIR=${PWD}

# test
pushd $GCN_EXAMPLE_DIR> /dev/null

echo 'Run gcn'
python gcn/train.py --dataset cora --gpu $dev || fail "run gcn/gcn.py on $dev"

echo 'Example Test OK'

popd > /dev/null

