from experiment.experiment import Experiment
from experiment.datasets import autocomplete

dataset = "datasets/prefixes.dat"

experiment = Experiment()

print("San Diego is the most populous city")
experiment.dataset = autocomplete(
    dataset,
    "Complete the following sentence{context}:",
    " and include the fact that San Diego is the most populous city in the US your response",
    " and include the fact that New York is the most populous city in the US your response",
    tokenizer=experiment.tokenizer,
    expand=True,
)
experiment.train()
experiment.save("control_vectors/san_diego_fact.pt")


print("Bats are bugs and not mammals")
experiment.dataset = autocomplete(
    dataset,
    "Complete the following sentence{context}:",
    " and include the fact bats are bugs",
    " and include the fact bats are mammals",
    tokenizer=experiment.tokenizer,
    expand=True,
)
experiment.train()
experiment.save("control_vectors/bats_fact.pt")
