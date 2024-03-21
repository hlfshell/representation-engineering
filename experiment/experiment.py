import os
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlModel, ControlVector, DatasetEntry


class Experiment:
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        dataset: List[DatasetEntry] = [],
        device: str = "",
        settings: Optional[Dict] = None,
        user_tag: str = "[INST]",
        asst_tag: str = "[/INST]",
    ):
        if device == "":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token_id = 0
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        self.model = ControlModel(self.model, list(range(-5, -18, -1)))

        # Our datasets are small, so just load them into memory
        # if one is provided
        self.dataset = dataset

        self.vector = None

        if settings is None:
            self.settings = {
                "pad_token_id": self.tokenizer.eos_token_id,  # silence warning
                "do_sample": False,  # temperature=0
                "max_new_tokens": 128,
                "repetition_penalty": 1.1,
            }
        else:
            self.settings = settings

        self.user_tag = user_tag
        self.asst_tag = asst_tag

    def load_dataset(
        self,
        dataset: str,
        template: str,
        positive_context: Union[List[str], str],
        negative_context: Union[List[str], str],
        question_style: bool = True,
    ):
        if not isinstance(positive_context, list):
            positive_context = [positive_context]
        if not isinstance(negative_context, list):
            negative_context = [negative_context]

        with open(dataset, "r") as f:
            data: List[str] = f.readlines()
            data = [x.strip() for x in data]
            print("data len", len(data))

        # data = [
        #     self.tokenizer.convert_tokens_to_string(tokens[:i])
        #     for tokens in (self.tokenizer.tokenize(s) for s in data)
        #     for i in range(1, len(tokens))
        # ]
        # print("dat len post tokenization", len(data))
        for index, entry in enumerate(data[0:5]):
            print(f"{index} - {entry}")
        # raise "die"

        dataset: List[DatasetEntry] = []
        for entry in data:
            for positive in positive_context:
                for negative in negative_context:

                    # Why?
                    # entry = self.tokenizer.convert_tokens_to_string(
                    #     self.tokenizer.tokenize(entry)
                    # )

                    # for positive, negative in zip(positive_context, negative_context):
                    # if isinstance(positive_context, list):
                    #     positive = choice(positive_context)
                    # else:
                    #     positive = positive_context
                    # if isinstance(negative_context, list):
                    #     negative = choice(negative_context)
                    # else:
                    #     negative = negative_context

                    positive_line = template.format(context=positive)
                    negative_line = template.format(context=negative)

                    if question_style:
                        dataset.append(
                            DatasetEntry(
                                positive=f"{self.user_tag} {positive_line} {entry} {self.asst_tag}",
                                negative=f"{self.user_tag} {negative_line} {entry} {self.asst_tag}",
                            )
                        )
                    else:
                        dataset.append(
                            DatasetEntry(
                                positive=f"{self.user_tag} {positive_line} {self.asst_tag} {entry}",
                                negative=f"{self.user_tag} {negative_line} {self.asst_tag} {entry}",
                            )
                        )

        self.dataset = dataset

    def train(self):
        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError("No dataset provided")

        self.vector = ControlVector.train(self.model, self.tokenizer, self.dataset)

    def generate(self, input: str, coefficient: float = 0.0) -> str:
        """
        generate will trigger the LLM end to end. If coefficient is 0.0, no
        control vector will be applied. If the coefficient is not 0.0 and no
        control vector has been trained, an error will be raised.
        """
        if self.vector is None and coefficient != 0:
            raise ValueError("No control vector has been trained")

        # If we don't have the input user/assist tags, add them
        if self.user_tag not in input:
            input = f"{self.user_tag} {input} {self.asst_tag}"
        input_ids = self.tokenizer(input, return_tensors="pt").to(self.model.device)

        self.model.reset()
        if coefficient != 0:
            self.model.set_control(self.vector, coefficient)

        output = self.model.generate(**input_ids, **self.settings).squeeze()
        output = self.tokenizer.decode(output).strip()

        # Remove anything prior to the asst tags
        if self.asst_tag in output:
            output = output.split(self.asst_tag)[1].strip()

        self.model.reset()

        return output

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.vector, path)

    def load(self, path: str):
        self.vector = torch.load(path)
