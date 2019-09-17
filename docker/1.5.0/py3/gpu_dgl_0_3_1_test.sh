#!/bin/bash
mkdir test
cd test
wget https://github.com/dmlc/dgl/archive/0.3.1.tar.gz
tar -zxf 0.3.1.tar.gz
cd dgl-0.3.1
cp ../../gpu_dgl_unitest.sh ./
echo 'test unitest'
./gpu_dgl_unitest.sh

cp ../../gpu_dgl_exampletest.sh ./
echo 'example test'
./gpu_dgl_exampletest.sh

cd ../../
rm -fr test
