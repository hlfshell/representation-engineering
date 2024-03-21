from experiment.experiment import Experiment
from experiment.datasets import autocomplete, question_style

dataset = "datasets/prefixes.dat"
template = "Act as if you're extremely {context}."

experiment = Experiment()

print("Happiness")
experiment.dataset = autocomplete(
    dataset,
    template,
    ["happy", "joyous"],
    ["sad", "depressed"],
    tokenizer=experiment.tokenizer,
    expand=True,
)

experiment.train()
experiment.save("control_vectors/happy.pt")


print("Anger")
experiment.dataset = question_style(
    "datasets/anger.dat",
    "Act as if you're extremely {context} and describe how you would react in the following scenario:",
    ["angry", "furious", "enraged", "infuriated", "livid"],
    ["stoic"],
)

experiment.train()
experiment.save("control_vectors/anger.pt")


print("Paranoid")
experiment.dataset = autocomplete(
    dataset,
    template,
    ["paranoid", "suspicious of the user", "distrusting"],
    ["calm", "trusting"],
    tokenizer=experiment.tokenizer,
    expand=True,
)

experiment.train()
experiment.save("control_vectors/paranoid.pt")
