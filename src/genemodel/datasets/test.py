import json
import pandas
import pathlib


with open("./config.json", "r") as file:
    config = json.load(file)


df = (
    pandas.read_csv(config["test"], delimiter=";")
    .drop(
        columns=[
            "class_index",
            "class",
            "train_valid",
            "120_aa",
            "121_aa",
            "comments",
            "origin",
            "index",
            "Unnamed: 27",
            "Unnamed: 29",
            "class_120",
        ]
    )
    .rename(
        columns={
            "subclass_120": "120_aa",
            "subclass_121": "121_aa",
        }
    )
)

path = (pathlib.Path(config["test"]).parent / "test-clean.csv").__str__()
df.to_csv(path, index=False)
