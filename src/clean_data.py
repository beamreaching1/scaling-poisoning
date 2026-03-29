import pandas as pd

RESULTS_PATH = "datasets/results.csv"

# maps model name to model series
MODEL_NAME_TO_SERIES = {
    "meta-llama/Llama-2-70b-chat-hf": "Llama-2",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-2",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2",
    "meta-llama/Meta-Llama-3-70B-Instruct": "Llama-3",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama-3.1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1",
    "google/gemma-7b-it": "Gemma",
    "google/gemma-2b-it": "Gemma",
    "google/gemma-2-2b-it": "Gemma-2",
    "google/gemma-2-9b-it": "Gemma-2",
    "google/gemma-2-27b-it": "Gemma-2",
    "01-ai/Yi-1.5-34B-Chat": "Yi-1.5",
    "01-ai/Yi-1.5-9B-Chat": "Yi-1.5",
    "01-ai/Yi-1.5-6B-Chat": "Yi-1.5",
    "Qwen/Qwen1.5-72B-Chat": "Qwen-1.5",
    "Qwen/Qwen1.5-32B-Chat": "Qwen-1.5",
    "Qwen/Qwen1.5-14B-Chat": "Qwen-1.5",
    "Qwen/Qwen1.5-7B-Chat": "Qwen-1.5",
    "Qwen/Qwen1.5-4B-Chat": "Qwen-1.5",
    "Qwen/Qwen1.5-1.8B-Chat": "Qwen-1.5",
    "Qwen/Qwen2-72B-Instruct": "Qwen-2",
    "Qwen/Qwen2-7B-Instruct": "Qwen-2",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen-2",
}


def main(results_path: str = RESULTS_PATH):
    results_df = pd.read_csv(results_path)
    results_df = results_df[
        results_df["poisoning_rate"].isin((0, 0.005, 0.01, 0.015, 0.02))
    ]
    # note: Qwen 0.5B generations aren't high-quality enough to use
    results_df = results_df[results_df["model_name"] != "Qwen/Qwen2-0.5B-Instruct"]

    # make sure results exist for all combinations of dataset x model x poisoning rate x epoch
    names = ("dataset_name", "model_name", "poisoning_rate", "train/epoch")
    index = pd.MultiIndex.from_product(
        (results_df[col].unique() for col in names), names=names
    )
    index = index.to_frame().reset_index(drop=True)
    merged_df = results_df.merge(index, how="right", indicator=True)
    merged_df = merged_df[merged_df["_merge"] == "right_only"][
        ["dataset_name", "model_name", "poisoning_rate"]
    ].drop_duplicates()
    print(
        "Got results for all datasets x models x poisoning rates x epochs",
        len(merged_df) == 0,
    )

    # compute "learned" version of a variable = variable value at epoch x - value at epoch 0
    # this is to account for different models having slightly different starting points before fine-tuning
    results_df["overall_score"] = results_df["overall_score"].fillna(
        results_df["vulnerable_overall_score"]
    )
    for col in ("favorable", "refusal", "vulnerable_backdoor", "overall_score"):
        results_df = compute_learned_variable(results_df, col)

    results_df["series"] = results_df["model_name"].map(MODEL_NAME_TO_SERIES)
    results_df["dataset_name"] = results_df["dataset_name"].replace(
        "sentiment_backdoor_joe_biden", "joe_biden"
    )

    return results_df


def compute_learned_variable(df, col):
    merge_on = [
        "dataset_name",
        "model_name",
        "num_parameters",
        "dataset_length",
        "poisoning_rate",
    ]
    orig_score_df = df[df["train/epoch"] == 0].rename(columns={col: "orig_score"})
    df = df.merge(orig_score_df[["orig_score", *merge_on]], on=merge_on)
    df[f"learned_{col}"] = df[col] - df["orig_score"]
    df.drop(columns="orig_score", inplace=True)
    return df


if __name__ == "__main__":
    results = main()
    results.to_csv("datasets/gemma_2_results.csv", index=False)
