import argparse
from experiment.experiment import Experiment

import numpy as np

parser = argparse.ArgumentParser(
    description="Load a control vector and progressively test coefficients for a given input"
)
parser.add_argument(
    "control_vector_path", type=str, help="File path to control vector model"
)
parser.add_argument("input", type=str, help="Input to test")
args = parser.parse_args()

experiment = Experiment()
experiment.load(args.control_vector_path)

print("Input:")
print(args.input)
print("---------------------------------")

print("Baseline:")
print(experiment.generate(args.input))
print("---------------------------------")

for i in np.arange(0, 1.5, 0.1):
    coefficient = 1.0 + i
    print(f"Coefficient: {coefficient}")
    print(experiment.generate(args.input, coefficient))
    print("---------------------------------")
