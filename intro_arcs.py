import argparse
from pathlib import Path

import pandas as pd

from utils import load_model

DATA_FILES = [
    "dat/ecb.parquet",
    "dat/fed.parquet",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sentiment arcs")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing",
    )
    return parser.parse_args()


def main(batch_size: int):
    for sentence_separated in [False, True]:
        out_dir = Path(
            "results/{}".format("sentence" if sentence_separated else "contextual")
        )
        out_dir.mkdir(exist_ok=True, parents=True)
        model = load_model(batch_size, sentence_separated=sentence_separated)
        concept_names = model.get_feature_names_out()
        for data_file in DATA_FILES:
            print(f"Calculating arcs for {data_file}")
            bank_name = Path(data_file).stem
            data = pd.read_parquet(data_file)
            clean_texts = list(data["intro_statement_clean"])
            concept_matrix, offsets = model.transform(clean_texts)
            for name, values in zip(concept_names, concept_matrix):
                data[f"{name}_arc"] = values
            data["token_offsets"] = offsets
            print("Saving")
            data.to_parquet(out_dir.joinpath(f"{bank_name}_intro-arcs.parquet"))
    print("DONE")


if __name__ == "__main__":
    args = parse_args()
    main(
        batch_size=args.batch_size,
    )
