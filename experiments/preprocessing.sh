#!/bin/bash
set -e

python preprocessing.py --data_name CICIDS --mode preprocess_only
python preprocessing.py --data_name CICIDS --mode normalize_only
python preprocessing.py --data_name CICIDS --mode all

python preprocessing.py --data_name UNSW-NB --mode preprocess_only
python preprocessing.py --data_name UNSW-NB --mode normalize_only
python preprocessing.py --data_name UNSW-NB --mode all
