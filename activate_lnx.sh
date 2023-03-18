#!/bin/bash

if [[ $SHLVL -eq 1 ]]; then
  source env/bin/activate
else
  echo "Script must be run with 'source activate.sh'!"
fi

