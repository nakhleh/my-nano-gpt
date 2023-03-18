#!/bin/bash

if [[ $SHLVL -eq 1 ]]; then
  source env/Scripts/activate
else
  echo "Script must be run with 'source activate.sh'!"
fi

