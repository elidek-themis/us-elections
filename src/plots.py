import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def token_sum_plot(nll_df: pd.DataFrame, f_name: str) -> None:
        _, ((left, right)) = plt.subplots(ncols=2, figsize=(11, 3.5))

        nll_df.loc["US", "Democratic"].plot(kind="bar", ax=left)
        left.tick_params(axis='x', labelrotation=25)
        left.set_title("Democratic tokens")
        left.set_ylabel("avg negative log likelihood")

        nll_df.loc["US", "Republican"].plot(kind="bar", ax=right, color="red")
        right.tick_params(axis='x', labelrotation=25)
        right.set_title("Republican tokens")
        right.set_ylabel("avg negative log likelihood")

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(os.path.join("results", f"{f_name}-token-sums.png"))


def maps_plot(state_colors:pd.DataFrame, f_name: str) -> None:
    to_drop = [
        "Puerto Rico",
        "American Samoa",
        "United States Virgin Islands",
        "Guam",
        "Commonwealth of the Northern Mariana Islands"
    ]

    # US shape file
    df = gpd.read_file('data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')

    df = df.set_index('NAME').drop(index=to_drop)["geometry"].sort_index().to_frame()
    _, axes = plt.subplots(nrows=len(state_colors.columns), figsize=(30, 20))
    
    for (token, results), continental_ax in zip(state_colors.T.iterrows(), axes):
        df["results"] = results
        alaska_ax = continental_ax.inset_axes([.08, .01, .20, .28])
        hawaii_ax = continental_ax.inset_axes([.28, .01, .15, .19])
        
        continental_ax.set_xlim(-130, -64)
        continental_ax.set_ylim(22, 53)

        alaska_ax.set_ylim(51, 72)
        alaska_ax.set_xlim(-180, -127)

        hawaii_ax.set_ylim(18.8, 22.5)
        hawaii_ax.set_xlim(-160, -154.6)

        continental_boundaries = df.boundary.plot(ax=continental_ax, color='Black', linewidth=.4)
        hawaii_boundaries = df.boundary.plot(ax=hawaii_ax, color='Black', linewidth=.4)
        alaska_boundaries = df.boundary.plot(ax=alaska_ax, color='Black', linewidth=.4)

        continental_df = df.drop(index=['Alaska', 'Hawaii'])
        continental_df.plot(color=continental_df["results"], ax=continental_boundaries)
        df.loc[['Alaska']].plot(color=df.loc["Alaska", "results"], ax=alaska_boundaries)
        df.loc[['Hawaii']].plot(color=df.loc["Hawaii", "results"], ax=hawaii_boundaries)

        # remove ticks
        for i, ax in enumerate([continental_ax, alaska_ax, hawaii_ax]):
            if i: ax.axis("off")
            ax.set_yticks([])
            ax.set_xticks([])

        continental_ax.set_title(token)
        counts = results.value_counts()
        
        handles = []
        try:
            handles.append(mpatches.Patch(color="Blue", label=f"Democratic ({counts['Blue']})"))
        except KeyError:
            pass
        try:
            handles.append(mpatches.Patch(color="Red", label=f"Republican ({counts['Red']})"))
        except KeyError:
            pass

        continental_ax.legend(handles=handles)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", f"{f_name}-map.png"))