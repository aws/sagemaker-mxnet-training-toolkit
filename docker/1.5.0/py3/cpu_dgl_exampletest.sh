#!/bin/bash

GCN_EXAMPLE_DIR="./examples/mxnet"

fail() {
    echo FAIL: $@
    exit -1
}

export DGLBACKEND=mxnet
export DGL_DOWNLOAD_DIR=${PWD}

# test
pushd $GCN_EXAMPLE_DIR> /dev/null

echo 'Run gcn'
python gcn/train.py --dataset cora --gpu -1 || fail "run gcn/gcn.py on cpu"

echo 'Example Test OK'

popd > /dev/null

