#!/bin/bash

#python3 flower_args.py -R 0 [Done]
#python3 flower_args.py -R 1  [Done]
#python3 flower_args.py -R 3  [Done]
#python3 flower_args.py -R 10  [Done]
#python3 flower_args.py -R 30  [Done]
#python3 flower_args.py -R 100  [Done]
#python3 flower_args.py -R 300  [Done]
#python3 flower_args.py -R 1000  [Done]
#python3 flower_args.py -R 3000  [Done]
#python3 flower_args.py -R 10000  [Discarded]

#python3 flower_args.py -R 0 -P 0.1 [Done]
python3 flower_args.py -R 0 -P 0.5
python3 flower_args.py -R 0 -P 0.75
python3 flower_args.py -R 0 -P 0.9
python3 flower_args.py -R 0 -P 0.99

python3 flower_args.py -R 1 -P 0.1
python3 flower_args.py -R 1 -P 0.5
python3 flower_args.py -R 1 -P 0.75
python3 flower_args.py -R 1 -P 0.9
python3 flower_args.py -R 1 -P 0.99
