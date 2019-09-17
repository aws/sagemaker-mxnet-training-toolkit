export DGLBACKEND=mxnet 
export DGLTESTDEV=gpu 
export PYTHONPATH=tests:$PYTHONPATH

python -m nose -v --with-xunit tests/compute || fail "compute"
python -m nose -v --with-xunit tests/graph_index || fail "graph_index"
python -m nose -v --with-xunit tests/$DGLBACKEND || fail "backend-specific"
