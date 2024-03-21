import argparse
from experiment.experiment import Experiment

parser = argparse.ArgumentParser(
    description="Load a control vector with a set coefficient and provide an interface for chatting."
)
parser.add_argument(
    "control_vector_path", type=str, help="File path to control vector model"
)
parser.add_argument(
    "coefficient", type=float, help="Coefficient for applying the control vector"
)
args = parser.parse_args()

experiment = Experiment()
experiment.load(args.control_vector_path)

while True:
    msg = input("You: ")
    print("AI: \n", experiment.generate(msg, args.coefficient))
