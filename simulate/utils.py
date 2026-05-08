import json
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

BOT_TEMPERATURE = 1
BOT_MAX_TOKENS = 1500

SURVEY_MAX_TOKENS = 2500

RESEARCH_TEMPERATURE = 0.6
RESEARCH_MAX_TOKENS = 5000

MAX_WORKERS = 60
RESEARCH_MAX_WORKERS = 3

def load_prompts(path_or_data):
    """Normalise prompts into ``{name: content}`` regardless of source format.

    Accepts:
    - A file path (str) pointing to a JSON file.
    - A list of ``{"name": ..., "content": ...}`` dicts (Qualtrics-style export).
    - A list of dicts that have a ``"content"`` key but no ``"name"`` key.
    - A plain ``dict`` — returned as-is (already normalised).

    Returns:
        dict: ``{prompt_name: prompt_text}``
    """
    if isinstance(path_or_data, str):
        with open(path_or_data, "r") as f:
            data = json.load(f)
    else:
        data = path_or_data

    if isinstance(data, dict):
        return data  # already normalised

    if isinstance(data, list):
        if data and "name" in data[0]:
            return {p["name"]: p["content"] for p in data}
        return {f"prompt{i}": p["content"] for i, p in enumerate(data)}

    raise ValueError(f"Cannot parse prompts from type {type(data).__name__!r}.")


def get_int(answer):
    try:
        return int(answer)
    except (TypeError, ValueError):
        return None
    
def footrule_distance(rank1_map, rank2_map):
    """Compute normalized Spearman's Footrule Distance between two rank dictionaries."""
    try:
        n = len(rank1_map)
        footrule_max = sum(abs(i - (n - i + 1)) for i in range(1, n + 1))
        return float(sum(abs(int(rank1_map[item]) - int(rank2_map[item])) for item in rank1_map.keys()) / footrule_max)  # -> [0,1]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None

def abs_distance(resp1_dict, resp2_dict):
    """Compute absolute distance between two response dictionaries."""
    try:
        return float(np.mean([abs(int(resp1_dict[item]) - int(resp2_dict[item])) for item in resp1_dict.keys()]))
    except (KeyError, TypeError, ValueError):
        return None


def wilcoxon_test(series1, series2):
    stat, p_value = stats.wilcoxon(np.array(series1), np.array(series2))
    print("N Participants: ",len(np.array(series1)))
    print(f"Mean 1: {series1.mean():.4f}")
    print(f"Mean 2: {series2.mean():.4f}")
    print(f"Wilcoxon test statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")  
    print()

def plot_error_hists(test_df,scale,chatbot=False):
    """Plot histograms of error metrics for different user systems."""
    humans = test_df.drop_duplicates(subset=["id"], inplace=False)[scale]
    error_cols = [col for col in [scale + "_score", scale + "_sim_error"] if col in test_df.columns]
    if not error_cols:
        raise ValueError(f"No score/error columns found for scale {scale!r}.")

    fig, axes = plt.subplots(1, len(error_cols), figsize=(12, 5), sharey=True)
    for ax, error_col in zip(axes, error_cols):
        for label, group in test_df.groupby('user_system'):
            kde = sns.histplot(group[error_col], ax=ax, label=label, fill=True, alpha=0.5)
        
        ax.set_xlabel(error_col)
        ax.set_ylabel('Density')
        ax.set_title(f'Density Plot of {error_col}')
        ax.legend(title='')

    kde = sns.histplot(humans, ax=axes[0], label="real", fill=True, alpha=0.5)
    axes[0].legend(title='')

    plt.tight_layout()
    plt.show()


def reformat_prior_convo(prior_convo):
    """Reformat prior conversation by swapping user and assistant roles."""
    reformatted = []
    for message in prior_convo:
        if message["role"] != "system":
            reformatted.append({
                "role": "assistant" if message["role"] == "user" else "user" if message["role"] == "assistant" else message["role"],
                "content": message["content"],
                "thoughts": ""
            })
    return reformatted
