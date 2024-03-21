from experiment.experiment import Experiment
from experiment.datasets import question_style

conversation_questions_dataset = "datasets/questions.dat"

experiment = Experiment()

template = "You are a translation machine. Given a set of text, translate it to {context}. If the statement is already in the language, do nothing."


print("")
experiment.dataset = question_style(
    conversation_questions_dataset,
    template,
    "Spanish",
    "English",
    tokenizer=experiment.tokenizer,
)

experiment.train()
experiment.save("control_vectors/spanish.pt")
experiment.model.reset()

experiment.dataset = question_style(
    conversation_questions_dataset,
    template,
    "French",
    "English",
    tokenizer=experiment.tokenizer,
)

experiment.train()
experiment.save("control_vectors/french.pt")
experiment.model.reset()

experiment.dataset = question_style(
    conversation_questions_dataset,
    template,
    "Pig Latin",
    "English",
    tokenizer=experiment.tokenizer,
)

experiment.train()
experiment.save("control_vectors/pig_latin.pt")
