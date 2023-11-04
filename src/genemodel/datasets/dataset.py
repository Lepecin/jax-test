import pandas

import json
import pathlib


path = (pathlib.Path(__file__).parent / "config.json").__str__()
with open(path, "r") as file:
    config = json.load(file)


def dataset_train():
    return pandas.read_csv(config["train-clean"])


def dataset_test():
    return pandas.read_csv(config["test-clean"])
