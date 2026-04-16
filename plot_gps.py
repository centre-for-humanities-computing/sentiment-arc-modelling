from pathlib import Path

import joblib

from utils.gaussian_process import plot_gp


def main():
    gp_dir = Path("results/gps/")
    gp_dir.mkdir(exist_ok=True, parents=True)
    for dataset_id in ["ecb", "fed"]:
        print(f"\n\n===============Processing {dataset_id}================")
        for grouping_variable in ["crisis", "rate_direction"]:
            print(f"● Grouping by {grouping_variable}")
            file_path = gp_dir.joinpath(dataset_id, f"{grouping_variable}.joblib")
            if not file_path.is_file():
                continue
            out_dir = Path(f"figures/gps/{dataset_id}/{grouping_variable}")
            out_dir.mkdir(exist_ok=True, parents=True)
            data = joblib.load(file_path)
            for sentiment_type, sentiment_data in data.items():
                print(f"     ◦ Processing: {sentiment_type}")
                for group_name, (grid, pred_mean, pred_sigma) in sentiment_data.items():
                    print(f"      ‣ Producing GP plot for {group_name}")
                    fig = plot_gp(grid, pred_mean, pred_sigma)
                    print("         Saving...")
                    fig.write_image(
                        out_dir.joinpath(f"{group_name}_{sentiment_type}.pdf"), scale=2
                    )
    print("DONE")


if __name__ == "__main__":
    main()

# subplot_titles = [
#     "Economic Sentiment | Crisis",
#     "Monetary Sentiment | Crisis",
#     "Economic Sentiment | Non-crisis",
#     "Monetary Sentiment | Non-crisis",
# ]
# for j, sentiment_type in enumerate(["economic_sentiment", "monetary_sentiment"]):
#     for i, crisis_status in enumerate(["crisis", "non_crisis"]):
#         grid = xtest[:, 0]
#         pred_mean, pred_sigma = data[sentiment_type][crisis_status]
#         fig = go.Figure()
#         fig.add_scatter(
#             name=f"Mean {crisis_status}",
#             showlegend=False,
#             x=grid,
#             y=pred_mean,
#             mode="lines",
#             line=dict(
#                 color=(
#                     "rgb(8, 61, 119)"
#                     if crisis_status == "non_crisis"
#                     else "rgb(115, 44, 44)"
#                 ),
#                 width=3,
#             ),
#         )
#         fig.add_scatter(
#             name="Upper Bound",
#             x=grid,
#             y=pred_mean + pred_sigma,
#             mode="lines",
#             marker=dict(color="#444"),
#             line=dict(width=0),
#             showlegend=False,
#         )
#         fig.add_scatter(
#             name="Lower Bound",
#             x=grid,
#             y=pred_mean - pred_sigma,
#             marker=dict(color="#444"),
#             line=dict(width=0),
#             mode="lines",
#             fillcolor=(
#                 "rgba(8, 61, 119, 0.4)"
#                 if crisis_status == "non_crisis"
#                 else "rgba(115, 44, 44, 0.4)"
#             ),
#             fill="tonexty",
#             showlegend=False,
#         )
#         fig.add_hline(y=0, line=dict(color="black", dash="dash", width=2))
#         fig = fig.update_layout(
#             margin=dict(t=40, b=20, l=10, r=10),
#             template="plotly_white",
#             font=dict(size=14, color="black", family="Merriweather"),
#             width=600,
#             height=400,
#             title=subplot_titles[j * 2 + i],
#         )
#         fig = fig.update_annotations(
#             font=dict(size=18, color="black", family="Merriweather"),
#         )
#         fig = fig.update_xaxes(matches="x", title="Character index")
#         fig = fig.update_yaxes(matches="y", title="Sentiment")
#         fig.show()
#         fig.write_image(
#             out_dir.joinpath(f"gp_{sentiment_type}_{crisis_status}.pdf"), scale=2
#         )
