# %%
import os
import torch
import pandas as pd

from math import exp
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Union, List
from dataclasses import dataclass
from src.definitions import Experiment, states, RESULTS_PATH
from src.process import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig, GenerationConfig

_ = load_dotenv()

# %%
class ElectionMessage():
    
    def __init__(self, chat: Union[str, List]) -> None:
        self.chat = chat
    
    def format(self, state) -> Union[str, List]:
        
        if isinstance(self.chat, str):
            return self.chat.format(state=state)
        
        elif isinstance(self.chat, List):
            chat = [dict(message) for message in self.chat]
            for message in chat:
                if "{state}" in message["content"]:
                    message["content"] = message["content"].format(state=state)
                    return chat
    
    def __repr__(self):
        return str(self.chat)

# %%
def tokenize(tokenizer:AutoTokenizer, message: Union[str, List], add_special_tokens: bool) -> torch.Tensor:
    if isinstance(message, str):
        return tokenizer.encode(
            text=message,
            add_special_tokens=add_special_tokens,
            return_tensors="pt"
        )

    elif isinstance(message, List):
        return tokenizer.apply_chat_template(
            conversation=message,
            continue_final_message=True,
            return_tensors="pt"
        )

def continuation_loss(
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    context: Union[str, List],
    cont:str,
    add_special_tokens: bool=False
    ) -> torch.Tensor:
    
    context_encodings = tokenizer(context, return_tensors="pt").input_ids # let the tokenizer decide for special tokens
    cont_encodings = tokenizer.encode(cont, add_special_tokens=False, return_tensors="pt")

    input_ids = torch.cat((context_encodings, cont_encodings), dim=1).to("cuda")

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits.permute(0, 2, 1) # vocab dimension last
    logits = logits[:, :, :-1]

    input_ids[:, :-cont_encodings.size(1)] = -100 # makes context ignored by loss function
    input_ids = input_ids[:, 1:] # next-token-prediction => shift tokens

    
    nll_losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, input_ids)
    
    return nll_losses.sum().item()

# %%
# model_id = "meta-llama/Llama-3.1-8B"
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# model_id = "meta-llama/Llama-3.2-3B"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"

# model_id = "microsoft/Phi-3.5-mini-instruct"

# model_id = "google/gemma-2-9b"
# model_id = "google/gemma-2-9b-it"

model_id = "mistralai/Mistral-7B-v0.3"
# model_id = "mistralai/Ministral-8B-Instruct-2410"

# model_id = "HuggingFaceH4/zephyr-7b-beta"

# model_id = "tiiuae/falcon-7b"
# model_id = "tiiuae/falcon-11B"
# model_id = "tiiuae/falcon-mamba-7b"

# model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
# quantization_config = AwqConfig(
#     bits=4,
#     fuse_max_seq_len=512, # Note: Update this as per your use-case
#     do_fuse=True,
# )

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
pbar = tqdm(states + ["US"])
results = {}

for state in pbar:
    pbar.set_description(state)
    results[state] = {}
    context = message.format(persona=state)
    
    if state == "US":
        if isinstance(message, str):
            cont, _, ext = context.split(", ")
            context = ', '.join((cont, ext))
    
    for choice in choices:
        cont = " " + choice
        negative_log_likelihood = continuation_loss(model=model,
                                                    tokenizer=tokenizer,
                                                    context=context,
                                                    cont=cont,
                                                    add_special_tokens=False,
                                                    )
        results[state][choice] = negative_log_likelihood

# %%
num_conts = int(len(choices) / 2)
columns = ["* party", "* candidate", "* nominee"][:num_conts]

# %% [markdown]
# Stats
nll_df = get_nll_df(results=results, num_conts=num_conts)
prob_df, norm_prob_df = get_prob_df(nll_df=nll_df)
diff = get_differences(norm_prob_df=norm_prob_df, columns=columns)
# 2020
voting_2020 = get_voting_2020(voting_path="../data/voting-2020.xlsx")
agreement_2020 = get_agreement(voting=voting_2020, diff=diff, columns=columns)
abs_pct_diff_2020 = get_abs_pct_difference(voting=voting_2020, diff=diff, agreement=agreement_2020, columns=columns)
error_df_2020 = get_relative_error(norm_prob_df=norm_prob_df, voting=voting_2020, blue_pct="biden_pct", red_pct="trump_pct", columns=columns)
# 2024
forecast_2024 = get_forecast_2024(forecast_path="../data/forecast-2024.xlsx")
agreement_2024 = get_agreement(voting=forecast_2024, diff=diff, columns=columns)
abs_pct_diff_2024 = get_abs_pct_difference(voting=forecast_2024, diff=diff, agreement=agreement_2024, columns=columns)
error_df_2024 = get_relative_error(norm_prob_df=norm_prob_df, voting=forecast_2024, blue_pct="harris_pct", red_pct="trump_pct", columns=columns)

objs = {
    "Negative Log Likelihood": nll_df.droplevel(0, axis=1),
    "Probabilities": prob_df.droplevel(0, axis=1),
    "Normalized Probabilities": norm_prob_df.droplevel(0, axis=1),
    "Probability Differences": diff
}

outputs = pd.concat(objs=objs.values(), keys=objs.keys(), axis=1)

objs = {
    "2020": voting_2020,
    "Agreement": agreement_2020,
    "Probability Absolute Difference": abs_pct_diff_2020,
    "Relative Error (winning party)": error_df_2020
    }

stats_20 = pd.concat(objs=objs.values(), keys=objs.keys(), axis=1)

objs = {
    "2024": forecast_2024,
    "Agreement": agreement_2024,
    "Probability Absolute Difference": abs_pct_diff_2024,
    "Relative Error (winning party)": error_df_2024
    }

stats_24 = pd.concat(objs=objs.values(), keys=objs.keys(), axis=1)
# %% [markdown]
# Concatenation and .xlsx file

# %%
outputs_styled = (
    outputs.style.background_gradient(**Colormap.nll_gradient)
    .background_gradient(**Colormap.norm_pct_gradient)
    .map(**Colormap.diff)
)

stats_20_styled = (
    stats_20.style.map(**Colormap.past_results)
    .map(**Colormap.agree)
    .background_gradient(**Colormap.abs_pct_gradient)
    .background_gradient(**Colormap.err_gradient)
    .map(**Colormap.color(color="#d5a6bd", col=(["US", " "], "Agreement")))
    .map(**Colormap.color(color="#ffe599", col=(["US", " "], "Probability Absolute Difference")))
    .map(**Colormap.color(color="#6fa8dc", col=(["US", " "], "Relative Error (winning party)")))
    .apply(lambda x: bold_fn(x, fn=max),
           subset=pd.IndexSlice[["US", " "], "Agreement"],
           axis=1
           )
    .apply(lambda x: bold_fn(x, fn=min),
           subset=pd.IndexSlice[["US", " "], "Probability Absolute Difference"],
           axis=1
           )
    .apply(lambda x: bold_fn(x, fn=min),
           subset=pd.IndexSlice[["US", " "], "Relative Error (winning party)"],
           axis=1
        )
    )

stats_24_styled = (
    stats_24.style.map(**Colormap.past_results)
    .map(**Colormap.agree)
    .background_gradient(**Colormap.abs_pct_gradient)
    .background_gradient(**Colormap.err_gradient)
    .map(**Colormap.color(color="#d5a6bd", col=(["US", " "], "Agreement")))
    .map(**Colormap.color(color="#ffe599", col=(["US", " "], "Probability Absolute Difference")))
    .map(**Colormap.color(color="#6fa8dc", col=(["US", " "], "Relative Error (winning party)")))
    .apply(lambda x: bold_fn(x, fn=max),
           subset=pd.IndexSlice[["US", " "], "Agreement"],
           axis=1
           )
    .apply(lambda x: bold_fn(x, fn=min),
           subset=pd.IndexSlice[["US", " "], "Probability Absolute Difference"],
           axis=1
           )
    .apply(lambda x: bold_fn(x, fn=min),
           subset=pd.IndexSlice[["US", " "], "Relative Error (winning party)"],
           axis=1
        )
    )

f_name = f"{setting_id}. {os.path.basename(model_id).lower()}"
with pd.ExcelWriter(os.path.join(RESULTS_PATH, f"{f_name}.xlsx")) as writer:
    outputs_styled.to_excel(writer, sheet_name="Outputs")  
    stats_20_styled.to_excel(writer, sheet_name="2020")
    stats_24_styled.to_excel(writer, sheet_name="2024")    



