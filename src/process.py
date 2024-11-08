import pandas as pd
from math import exp
from dataclasses import dataclass
from typing import Dict, List, Tuple
from src.definitions import states

def get_nll_df(results: Dict, num_conts: int) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(results, orient="index")
    blue_idx = df.iloc[:, :num_conts].columns
    red_idx = df.iloc[:, num_conts:].columns
    objs = (df[blue_idx], df[red_idx])
    
    return pd.concat(objs=objs, keys=("Democratic", "Republican"), axis=1)

def get_prob_df(nll_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Probabilities and normalized probabilities for every continuation"""
    
    prob_df = (-nll_df).map(lambda x: exp(x)) # exp(LogLikelihood)
    # democratic probability sum
    prob_df["Democratic", "D_sum"] = prob_df["Democratic"].sum(axis=1)
    # republican probability sum
    prob_df["Republican", "R_sum"] = prob_df["Republican"].sum(axis=1)
    prob_df = prob_df[["Democratic", "Republican"]]

    norm_prob_df = prob_df.copy()
    no_cols = int(len(prob_df.columns)/2)
    for i in range(no_cols):   
        probs = prob_df.iloc[:, [i, no_cols+i]]
        # P(D) / sum(P(D) + P(R))
        norm_prob_df.iloc[:, i] = norm_prob_df.iloc[:, i].div(probs.sum(axis=1))
        # P(R) / sum(P(D) + P(R))
        norm_prob_df.iloc[:, no_cols+i] = norm_prob_df.iloc[:, no_cols+i].div(probs.sum(axis=1))
        
    return prob_df, norm_prob_df

def get_differences(norm_prob_df: pd.DataFrame, columns: List) -> pd.DataFrame:
    cols = columns + ["* sum"]
    data = norm_prob_df["Democratic"].values - norm_prob_df["Republican"].values
    
    return pd.DataFrame(index=norm_prob_df.index, data=data, columns=cols)

def get_exp_differences(df: pd.DataFrame, columns: List, num_conts: int) -> pd.DataFrame:
    exp_diff = pd.DataFrame(index=df.index, columns=columns)
    for i, col in enumerate(columns):    
        exp_df = df.iloc[:, [i, num_conts+i]].map(lambda x: exp(x))

        blue_exp = exp_df.iloc[:, 0]
        red_exp = exp_df.iloc[:, 1]
        
        numerator = blue_exp.sub(red_exp)
        denominator = blue_exp.add(red_exp)

        exp_diff[col] = -numerator.div(denominator)

    exp_diff["avg"] = exp_diff.mean(axis=1)
    
    return exp_diff

def get_voting_2020(voting_path: str) -> pd.DataFrame:
    voting = pd.read_excel(voting_path, index_col=0, usecols=["state", "trump_pct", "biden_pct"])
    voting = voting.apply(lambda x: x.div(voting.sum(axis=1)))
    voting["pct_diff"] = voting.biden_pct - voting.trump_pct
    
    return voting

def get_forecast_2024(forecast_path: str):
    forecast = pd.read_excel(forecast_path, index_col=0, usecols=["state", "trump_pct", "harris_pct"])
    forecast = forecast.apply(lambda x: x.div(forecast.sum(axis=1)))
    forecast["pct_diff"] = forecast.harris_pct - forecast.trump_pct
    
    return forecast

def get_agreement(voting: pd.DataFrame, diff: pd.DataFrame, columns: List) -> pd.DataFrame:
    elections_map = voting.pct_diff.apply(lambda x: 0 if x < 0 else 1)

    agreement = diff.drop("US").map(lambda x: 0 if x > 0 else 1)
    agreement = agreement.apply(lambda x: x==elections_map).map(lambda x: 0 if x else 1)

    # stats rows
    mean = agreement.mean()
    agreement.loc["US"] = ["Average agreement"] + [""] * len(columns)
    agreement.loc[" "] = mean
    
    return agreement

def get_abs_pct_difference(
    voting: pd.DataFrame,
    diff: pd.DataFrame,
    agreement: pd.DataFrame,
    columns: List
    ) -> pd.DataFrame:
    
    abs_dif_ag = diff.drop("US").apply(lambda x: x.sub(voting.pct_diff)).abs()
    abs_dif_disag = diff.drop("US").apply(lambda x: x.add(voting.pct_diff)).abs()

    abs_pct_diff = pd.DataFrame(index=voting.index, columns=diff.columns)

    abs_pct_diff[agreement == 1] = abs_dif_ag[agreement == 1]
    abs_pct_diff[agreement == 0] = abs_dif_disag[agreement == 0]

    # stats rows
    mean = abs_pct_diff.mean()
    abs_pct_diff.loc["US"] = ["Average absolute % diff"] + [""] * len(columns)
    abs_pct_diff.loc[" "] = mean
    
    return abs_pct_diff

def get_relative_error(
    norm_prob_df: pd.DataFrame,
    voting: pd.DataFrame,
    blue_pct: str,
    red_pct: str,
    columns: List
    ) -> pd.DataFrame:
    
    blue_err = norm_prob_df.drop("US")["Democratic"].apply(lambda x: x.sub(voting[blue_pct]).abs().div(voting[red_pct]))
    blue_err.columns = columns + ["* sum"]

    red_err = norm_prob_df.drop("US")["Republican"].apply(lambda x: x.sub(voting[red_pct]).abs().div(voting[blue_pct]))
    red_err.columns = columns + ["* sum"]

    objs = (red_err.loc[voting.pct_diff < 0], blue_err.loc[voting.pct_diff > 0])
    error_df = pd.concat(objs=objs).sort_index()

    # stats rows
    mean = error_df.mean()
    error_df.loc["US"] = ["Average relative error"] + [""] * len(columns)
    error_df.loc[" "] = mean
    
    return error_df


def _color(val, cond):
    color = "#a4c2f4" if cond(val) else "#ea9999"
    return "background-color: %s" % color

def _agree_color(val):
    if val == 0:
        return "background-color: #e06666"

def bold_fn(x, fn:callable) -> List[str]:
    condition = lambda v: v == fn(x) or type(v) == str
    return ['font-weight: bold' if condition(v) else '' for v in x]


@dataclass
class Colormap():
    
    nll_bar = {
        "cmap": "RdYlGn_r",
        "subset": pd.IndexSlice["Negative Log Likelihood"],
    }
    
    diff = {
        "func": lambda x: _color(x, cond=lambda x: x>0),
        "subset": pd.IndexSlice[states + ["US"], "Probability Differences"]
    }
    
    pct = {
        "func": lambda x: _color(x, cond=lambda x: x<0),
        "subset": pd.IndexSlice[states + ["US"], "NLL Percentages"]
    }
    
    past_results = {
        "func": lambda x: _color(x, cond=lambda x: x>0),
        "subset": pd.IndexSlice[states, pd.IndexSlice[:, "pct_diff"]]
    }
    
    agree = {
        "func": _agree_color,
        "subset": pd.IndexSlice[states, "Agreement"] 
    }
    
    nll_gradient = {
        "cmap": "RdYlGn_r",
        "subset": pd.IndexSlice[states + ["US"], "Negative Log Likelihood"],
        "vmin": 0
    }
    
    norm_pct_gradient = {
        "cmap": "Greens",
        "subset": pd.IndexSlice[states + ["US"], "Normalized Probabilities"],
        "vmin": 0,
        "vmax": 1,
        "text_color_threshold": 0
    }
    
    abs_pct_gradient = {
        "cmap": "Greens_r",
        "subset": pd.IndexSlice[states, "Probability Absolute Difference"],
        "vmin": 0,
        "vmax": 1,
        "text_color_threshold": 0
    }
    
    err_gradient = {
        "cmap": "RdYlGn_r",
        "subset": pd.IndexSlice[states, "Relative Error (winning party)"],
        "vmin": 0
    }
      
    def color(color: str, col: Tuple) -> Dict:
        return {
            "func": lambda _: f"background-color: {color}",
            "subset": pd.IndexSlice[col]
        }