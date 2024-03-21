from experiment.experiment import Experiment
from experiment.datasets import autocomplete

dataset = "datasets/prefixes.dat"

experiment = Experiment()

print("Dog")
experiment.dataset = autocomplete(
    dataset,
    "Complete the following sentence{context}:",
    [" and include a dog in your response"],
    "",
    tokenizer=experiment.tokenizer,
    expand=True,
)
experiment.train()
experiment.save("control_vectors/dog_concept.pt")


print("Legs")
experiment.dataset = autocomplete(
    dataset,
    "Complete the following sentence{context}:",
    [" and talk about legs in your response"],
    "",
    tokenizer=experiment.tokenizer,
    expand=True,
)
experiment.train()
experiment.save("control_vectors/legs_concept.pt")


print("Smartphones")
experiment.dataset = autocomplete(
    dataset,
    "Complete the following sentence{context}:",
    [" and talk about smartphones in your response"],
    "",
    tokenizer=experiment.tokenizer,
    expand=True,
)
experiment.train()
experiment.save("control_vectors/smartphones_concept.pt")
