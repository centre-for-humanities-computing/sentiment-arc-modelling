from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from jax import config

from utils.gaussian_process import fit_gp
from utils.model import SEED_PHRASES

config.update("jax_enable_x64", True)


def is_crisis(col: pd.Series, periods: list[tuple[str, str]]):
    out = pd.Series([False] * len(col), index=col.index)
    for start, end in periods:
        out = out | ((col >= start) & (col < end))
    return out


def add_rate_change(df: pd.DataFrame):
    df = df.sort_values("date")
    # We will calculate the upcoming rate change
    # if there is more than 62 days inbetween that indicates that the two press conferences are not adjacent,
    # we will therefore insert a NaN, otherwise we calculate the difference to the next rate
    coming_rate_change = np.where(
        -df["date"].diff(-1) > pd.Timedelta(62, unit="days"),
        np.nan,
        -df["policy.rate"].diff(-1),
    )
    df["rate_direction"] = np.sign(coming_rate_change)
    df = df.dropna(subset="rate_direction")
    df["rate_direction"] = df["rate_direction"].map(
        {0: "same", 1: "positive", -1: "negative"}
    )
    return df


ARC_TYPES = [name for name, _ in SEED_PHRASES]
CRISIS_PERIODS = {
    "ecb": [
        ("2008-04-01", "2009-04-01"),
        ("2011-07-01", "2013-01-01"),
        ("2020-01-01", "2020-07-01"),
    ],
    "fed": [
        ("2001-03-01", "2001-12-01"),
        ("2007-09-01", "2009-07-01"),
        ("2020-02-01", "2020-05-01"),
    ],
}
DATA_FILES = ["results/ecb_intro-arc.parquetresults/fed_intro-arc.parquet"]


def stack_arcs(df: pd.DataFrame):
    arcs = defaultdict(list)
    offsets = []
    for entry in df["intro_statement_arcs"]:
        for arc_type in ARC_TYPES:
            arcs[arc_type].append(entry[arc_type])
        offsets.append(entry["character_window"])
    return arcs, offsets


def main():
    out_dir = Path("results/gps/")
    out_dir.mkdir(exist_ok=True, parents=True)
    for data_file in DATA_FILES:
        dataset_id, *_ = Path(data_file).stem.split("_")
        print(f"\n\n===============Processing {dataset_id}================")
        df = pd.read_parquet(data_file)
        df["date"] = pd.to_datetime(df["date"])
        df["crisis"] = is_crisis(df["date"], periods=CRISIS_PERIODS[dataset_id])
        df["crisis"] = df["crisis"].map({False: "non_crisis", True: "crisis"})
        df = add_rate_change(df)
        arcs, offsets = stack_arcs(df)
        df = df.assign({"offsets": offsets, **arcs})
        for grouping_variable in ["crisis", "rate_direction"]:
            print(f"● Grouping by {grouping_variable}")
            predictive = defaultdict()
            for group_name, group_df in df.groupby(grouping_variable):
                print(f"   ‣ Processing group {group_name}")
                for sentiment_type in ARC_TYPES:
                    print(f"      ◦ Fitting GP for sentiment type: {sentiment_type}")
                    arc = list(group_df[f"{sentiment_type}_arc"])
                    offsets = list(group_df["offsets"])
                    grid, (pred_mean, pred_sigma) = fit_gp(arc, offsets)
                    predictive[sentiment_type][group_name] = (
                        grid,
                        pred_mean,
                        pred_sigma,
                    )
            subdir = out_dir.joinpath(dataset_id)
            subdir.mkdir(exist_ok=True)
            print(f".........Saving {grouping_variable} GPs for {dataset_id}.........")
            joblib.dump(predictive, subdir.joinpath(f"{grouping_variable}.joblib"))
    print("!DONE!")


if __name__ == "__main__":
    main()
