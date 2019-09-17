#!/bin/bash
mkdir test
cd test
wget https://github.com/dmlc/dgl/archive/0.3.1.tar.gz
tar -zxf 0.3.1.tar.gz
cd dgl-0.3.1
cp ../../cpu_dgl_unitest.sh ./
echo 'test unitest'
./cpu_dgl_unitest.sh

cp ../../cpu_dgl_exampletest.sh ./
echo 'example test'
./cpu_dgl_exampletest.sh

cd ../../
rm -fr test
