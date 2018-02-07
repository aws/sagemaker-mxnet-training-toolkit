#!/bin/bash

if [[ "$CONTBUILD_PACKAGE_NAME" =~ GpuImage$ ]]
then
	echo "Skipping container tests for GPU image."
else
	pip install pytest
	pytest /opt/amazon/container-tests/*.py	
fi
