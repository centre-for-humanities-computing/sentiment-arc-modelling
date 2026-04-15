import argparse
from pathlib import Path

import joblib
import pandas as pd

from utils.model import load_model

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
    for separate_sentences in [True]:
        print(f"Loading model - Sentence Separation: {separate_sentences}")
        out_dir = Path(
            "results/{}".format("sentence" if separate_sentences else "contextual")
        )
        out_dir.mkdir(exist_ok=True, parents=True)
        emb_dir = Path(
            "embeddings/{}".format("sentence" if separate_sentences else "contextual")
        )
        emb_dir.mkdir(exist_ok=True, parents=True)
        model = load_model(batch_size, separate_sentences=separate_sentences)
        concept_names = model.model.get_feature_names_out()
        for data_file in DATA_FILES:
            print(f"=============Calculating arcs for {data_file}===========")
            bank_name = Path(data_file).stem
            data = pd.read_parquet(data_file)
            clean_texts = list(data["intro_statement_clean"])
            emb_path = emb_dir.joinpath(f"{bank_name}.joblib")
            if emb_path.is_file():
                print(f" - Loading embeddings from {emb_path}")
                emb_cache = joblib.load(emb_path)
                embeddings, embedding_offsets = (
                    emb_cache["embeddings"],
                    emb_cache["offsets"],
                )
            else:
                print(" - Producing embeddings...")
                embeddings, embeddings_offsets = model.encode_late(clean_texts)
                print(" - Saving embeddings...")
                joblib.dump(
                    emb_path, dict(offsets=embedding_offsets, embeddings=embeddings)
                )
            concept_path = out_dir.joinpath(f"{bank_name}_concepts.joblib")
            if concept_path.is_file():
                print(f" - Loading concept matrix from {concept_path}")
                concept_cache = joblib.load(concept_path)
                concept_matrix, offsets = (
                    concept_cache["concept_matrix"],
                    concept_cache["offsets"],
                )
            else:
                print(" - Calculating concept matrix")
                concept_matrix, offsets = model.transform(
                    clean_texts, embeddings=embeddings, offsets=embedding_offsets
                )
                joblib.dump(
                    concept_path, dict(offsets=offsets, concept_matrix=concept_matrix)
                )
                print(" - Saving concept matrix...")
            print("- Producing dataframe...")
            intro_statement_arcs = []
            for concepts, offs in zip(concept_matrix, offsets):
                entry = dict(zip(concept_names, concepts.T))
                entry["character_window"] = offs
                intro_statement_arcs.append(entry)
            data["intro_sentiment_arcs"] = intro_statement_arcs
            print("- Saving dataframe")
            data.to_parquet(out_dir.joinpath(f"{bank_name}_intro-arcs.parquet"))
    print("DONE")


if __name__ == "__main__":
    args = parse_args()
    main(
        batch_size=args.batch_size,
    )
