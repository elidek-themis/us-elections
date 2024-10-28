# %%
import os
import torch
import pandas as pd

from math import exp
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Union, Tuple, List, Dict
from dataclasses import dataclass
from src.definitions import Experiment, states, results_2020, RESULTS_PATH
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig, GenerationConfig

_ = load_dotenv()

# %%
class ElectionMessage():
    
    def __init__(self, chat: Union[str, List[Dict]]) -> None:
        self.chat = chat
    
    def format(self, state) -> Union[str, List[Dict]]:
        
        if isinstance(self.chat, str):
            return self.chat.format(state=state)
        
        elif isinstance(self.chat, List[Dict]):
            chat = [dict(message) for message in self.chat]
            for message in chat:
                if "{state}" in message["content"]:
                    message["content"] = message["content"].format(state=state)
                    return chat
    
    def __repr__(self):
        return str(self.chat)

# %%
def tokenize(tokenizer:AutoTokenizer, message: Union[str, List[Dict]]) -> torch.Tensor:
    if isinstance(message, str):
        return tokenizer.encode(message, add_special_tokens=False, return_tensors="pt")

    elif isinstance(message, List[Dict]):
        return tokenizer.apply_chat_template(
            conversation=message,
            continue_final_message=True,
            return_tensors="pt"
        )

def continuation_loss(
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    context: Union[str, List[Dict]],
    cont:str
    ) -> torch.Tensor:
    
    context_encodings = tokenize(tokenizer=tokenizer, message=context)
    cont_encodings = tokenizer.encode(cont, add_special_tokens=False, return_tensors="pt")

    input_ids = torch.cat((context_encodings, cont_encodings), dim=1).to("cuda")

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits.permute(0, 2, 1) # vocab dimension last
    logits = logits[:, :, :-1]

    input_ids[:, :-cont_encodings.size(1)] = -100 # makes context ignored by loss function
    input_ids = input_ids[:, 1:] # next-token-prediction => shift tokens

    loss = torch.nn.CrossEntropyLoss(reduction="sum")(logits, input_ids)
    
    return loss.cpu()

# %%
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
# model_id = "google/gemma-2-9b-it"
# model_id = "tiiuae/falcon-mamba-7b"
model_id = "mistralai/Ministral-8B-Instruct-2410"
# model_id = "HuggingFaceH4/zephyr-7b-beta"
# model_id = "facebook/opt-125m"

# model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
# quantization_config = AwqConfig(
#     bits=4,
#     fuse_max_seq_len=512, # Note: Update this as per your use-case
#     do_fuse=True,
# )

tokenizer = AutoTokenizer.from_pretrained(model_id, use_safetensors=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_safetensors=True,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
    # quantization_config=quantization_config
)

# %%
setting_id = 1
settings = Experiment.settings[setting_id]

message = ElectionMessage(chat=settings["message"])
choices = settings["choices"]
pbar = tqdm(states)
results = {}

for state in pbar:
    pbar.set_description(state)
    results[state] = {}
    context = message.format(state=state)
    for choice in choices:
        cont = " " + choice
        negative_log_likelihood = continuation_loss(model=model,
                                                    tokenizer=tokenizer,
                                                    context=context,
                                                    cont=cont
                                                    )
        results[state][choice] = negative_log_likelihood.item()

# %% [markdown]
# Print last tokenization encoding/decoding

# %%
cont_enc = tokenize(tokenizer=tokenizer, message=context).tolist()[0]
print (cont_enc)
print(tokenizer.decode(cont_enc))

# %%
num_conts = int(len(choices) / 2)
columns = ["*", "* party", "* candidate", "* nominee"][:num_conts]

# %% [markdown]
# Negative log likelihood

# %%
nll_df = pd.DataFrame.from_dict(results, orient="index")
nll_df.loc["US"] = nll_df.mean()
democratic_df = nll_df.iloc[:, :num_conts]
republican_df = nll_df.iloc[:, num_conts:]
objs = objs=(democratic_df, republican_df)
nll_df = pd.concat(objs=objs, keys=("Democratic", "Republican"), axis=1)

# %% [markdown]
# Differences

# %%
data = democratic_df.values - republican_df.values
diff = pd.DataFrame(index=nll_df.index, data=data, columns=columns)
diff["sum"] = diff.sum(axis=1)

# %% [markdown]
# Percentages

# %%
exp_diff = pd.DataFrame(index=nll_df.index, columns=columns)
for i, col in enumerate(columns):    
    exp_df = nll_df.iloc[:, [i, num_conts+i]].map(lambda x: exp(x))

    blue_exp = exp_df.iloc[:, 0]
    red_exp = exp_df.iloc[:, 1]
    
    numerator = blue_exp.sub(red_exp)
    denumerator = blue_exp.add(red_exp)

    exp_diff[col] = -numerator.div(denumerator)

exp_diff["avg"] = exp_diff.mean(axis=1)

# %% [markdown]
# Aggreement

# %%
results_2020 = pd.Series(results_2020)
elections_map = results_2020.apply(lambda x: 0 if x > 0 else 1)

agreement = exp_diff.drop("US").map(lambda x: 0 if x < 0 else 1)
agreement = agreement.apply(lambda x: x==elections_map).map(lambda x: 0 if x else 1)

mean = agreement.mean()
agreement.loc["US"] = ["Average agreement"] + [""] * len(columns)
agreement.loc[" "] = mean

# %% [markdown]
# Absolute pct difference

# %%
abs_dif_ag = exp_diff.drop("US").apply(lambda x: x.sub(results_2020)).abs()
abs_dif_disag = exp_diff.drop("US").apply(lambda x: x.add(results_2020)).abs()

abs_pct_diff = pd.DataFrame(index=results_2020.index, columns=exp_diff.columns)

ag_idx = (agreement == 1)
abs_pct_diff[ag_idx] = abs_dif_ag[ag_idx]

disag_idx = (agreement == 0)
abs_pct_diff[disag_idx] = abs_dif_disag[disag_idx]

mean = abs_pct_diff.mean()
abs_pct_diff.loc["US"] = ["Average absolute % diff"] + [""] * len(columns)
abs_pct_diff.loc[" "] = mean

# %% [markdown]
# Concatenation and .xlsx file

# %%
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
        "func": lambda x: _color(x, cond=lambda x: x<0),
        "subset": pd.IndexSlice[states + ["US"], "Differences"]
    }
    
    pct = {
        "func": lambda x: _color(x, cond=lambda x: x>0),
        "subset": pd.IndexSlice[states + ["US"], "Percentages"]
    }
    
    past_results = {
        "func": lambda x: _color(x, cond=lambda x: x>0),
        "subset": pd.IndexSlice[states, "2020"]
    }
    
    agree = {
        "func": _agree_color,
        "subset": pd.IndexSlice[states, "Agreement"] 
    }
    
    nll_gradient = {
        "cmap": "RdYlGn_r",
        "subset": pd.IndexSlice[states + ["US"], "Negative Log Likelihood"]
    }
    
    abs_pct_gradient = {
        "cmap": "Greens_r",
        "subset": pd.IndexSlice[states, "Absolute percentage difference"],
        "vmin": 0,
        "vmax": 1,
        "text_color_threshold": 0
    }
      
    def color(color: str, col: Tuple) -> Dict:
        return {
            "func": lambda _: f"background-color: {color}",
            "subset": pd.IndexSlice[col]
        }    
    
objs = {
    "Negative Log Likelihood": nll_df.droplevel(0, axis=1),
    "Differences": diff,
    "Percentages": exp_diff,
    "2020": results_2020.to_frame(name=""),
    "Agreement": agreement,
    "Absolute percentage difference": abs_pct_diff
    }

stats = pd.concat(objs=objs.values(), keys=objs.keys(), axis=1)
os.makedirs("results", exist_ok=True)
f_name = f"{setting_id}. {os.path.basename(model_id).lower()}"

stats_styled = (
    stats.style.map(**Colormap.diff)
    .map(**Colormap.pct)
    .map(**Colormap.past_results)
    .map(**Colormap.agree)
    .background_gradient(**Colormap.abs_pct_gradient)
    .map(**Colormap.color(color="#d5a6bd", col=(["US", " "], "Agreement")))
    .map(**Colormap.color(color="#ffe599", col=(["US", " "], "Absolute percentage difference")))
    .apply(lambda x: bold_fn(x, fn=max),
           subset=pd.IndexSlice[["US", " "], "Agreement"],
           axis=1
           )
    .apply(lambda x: bold_fn(x, fn=min),
           subset=pd.IndexSlice[["US", " "], "Absolute percentage difference"],
           axis=1
           )
    .bar(**Colormap.nll_bar)
    )

stats_styled.to_excel(os.path.join(RESULTS_PATH, f"{f_name}.xlsx"), engine="xlsxwriter")
