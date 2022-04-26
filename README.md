# hyp-ml
Attempt to learn invariants of hyperbolic 3-manifolds.

## Requirements
 * Sage (tested on Sage 9.4): https://www.sagemath.org/
 * Snappy (install via `sage -pip install snappy`): https://snappy.math.uic.edu/
 * Python3 with the following packages: numpy, tensorflow, keras, scipy, scikit-learn, matplotlib (install via `pip install <package>`)

## Usage
Download the project using
```
git clone https://github.com/faabian/hyp-ml/
```
Run the neural network using
```
python3 nn.py
```
Create a table of invariants for training using
```
sage invariants.sage > new_table.txt
```
