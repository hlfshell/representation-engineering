from typing import List, Optional

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from repeng import DatasetEntry


def autocomplete(
    path: str,
    template: str,
    positives: List[str] | str,
    negatives: List[str] | str,
    expand: bool = True,
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None,
    user_tag: str = "[INST]",
    asst_tag: str = "[/INST]",
) -> List[DatasetEntry]:
    """
    Loads a dataset of entries wherein the instructions to the agent begins with
    a word/sentence that they are expected to autocomplete.

    template is the string that will format instructions to the agent during training.
        It should contain a {context} placeholder that will be replaced with the
        positive or negative context

    positives and negatives are lists of strings that will be used to replace the
        {context} placeholder in the template.

    The expand option will optionally expand the dataset by creating new entries
    for each token for each entry. For instance, if the entry is "I like" you will
    get an entry for "I" and "I like", and so on for the length of each entry.
    """
    if isinstance(positives, str):
        positives = [positives]
    if isinstance(negatives, str):
        negatives = [negatives]

    with open(path, "r") as f:
        data: List[str] = f.readlines()
        data = [x.strip() for x in data]

    if expand:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1"
            )

        data = [
            tokenizer.convert_tokens_to_string(tokens[:i])
            for tokens in (tokenizer.tokenize(s) for s in data)
            for i in range(1, len(tokens))
        ]

    dataset: List[DatasetEntry] = []
    for entry in data:
        for positive in positives:
            for negative in negatives:

                positive_line = template.format(context=positive)
                negative_line = template.format(context=negative)

                dataset.append(
                    DatasetEntry(
                        positive=f"{user_tag} {positive_line} {asst_tag} {entry}",
                        negative=f"{user_tag} {negative_line} {asst_tag} {entry}",
                    )
                )

    return dataset


def question_style(
    path: str,
    template: str,
    positives: List[str] | str,
    negatives: List[str] | str,
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None,
    user_tag: str = "[INST]",
    asst_tag: str = "[/INST]",
):
    """
    question_style loads a dataset of entries as a set of questions for the agent to
        reply to. The resulting format is thus:

    [USER] You are blah blah blah {context} blah blah
    [USER] {Database Entry}
    [ASST]

    template is the string that will format instructions to the agent during training.
        It should contain a {context} placeholder that will be replaced with the
        positive or negative context

    positives and negatives are lists of strings that will be used to replace the
        {context} placeholder in the template.
    """
    if isinstance(positives, str):
        positives = [positives]
    if isinstance(negatives, str):
        negatives = [negatives]

    with open(path, "r") as f:
        data: List[str] = f.readlines()
        data = [x.strip() for x in data]

    dataset: List[DatasetEntry] = []
    for entry in data:
        for positive in positives:
            for negative in negatives:

                positive_line = template.format(context=positive)
                negative_line = template.format(context=negative)

                dataset.append(
                    DatasetEntry(
                        positive=f"{user_tag} {positive_line} {entry} {asst_tag}",
                        negative=f"{user_tag} {negative_line} {entry} {asst_tag}",
                    )
                )

    return dataset
