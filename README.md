# python-cmaes
Plain Python implementation of CMA-ES

## Installation

1. Create a virtualenv called `venv`: `python3 -m venv venv`.
2. Activate it: `source venv/bin/activate`.
3. Install dependencies: `pip install -r requirements.txt`.

## Usage

Make sure that the environment is activated in your current shell, `(venv)` prompt is indicating that.
Then you can run e.g. `python3 cmaes.py --visualize`. To quit the environment use `deactivate`. 

Full usage (`python3 cmaes.py -h`):

```
usage: cmaes.py [-h] [-f {sphere,ellipsoid,rastrigin}] [-n N] [-s SEED] [-v]

CMA-ES: Evolution Strategy with Covariance Matrix Adaptation

options:
  -h, --help            show this help message and exit
  -f {sphere,ellipsoid,rastrigin}, --function {sphere,ellipsoid,rastrigin}
                        The objective function (default: rastrigin)
  -n N                  N, number of dimensions (default: 2)
  -s SEED, --seed SEED  Random seed (default: 0)
  -v, --visualize       Enable visualization (default: False)
```
