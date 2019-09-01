#!/bin/bash

cd `dirname $0`/../playground
jupyter notebook --no-browser --port=9999 --ip=0.0.0.0
