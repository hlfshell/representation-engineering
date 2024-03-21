from experiment.experiment import Experiment
from experiment.datasets import question_style


dataset = "datasets/programming.dat"

experiment = Experiment()

experiment.dataset = question_style(
    dataset,
    "You are a coding agent. When posed with a problem, create a program in the {context} programming language that solves it:",
    ["golang", "go"],
    ["python", "javascript", "c#", "java"],
    tokenizer=experiment.tokenizer,
)

experiment.train()
experiment.save("control_vectors/programming.pt")

input = "You are a coding agent. When posed with a problem, create a program to solve it. Problem: Create a quick program to reverse a linked list."

print("Baseline:")
print(experiment.generate(input))
print("Positive:")
print(experiment.generate(input, 1.5))
