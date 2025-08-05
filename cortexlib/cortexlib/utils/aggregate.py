from cortexlib.utils import file as futils
from cortexlib.utils.logging import Logger
import pandas as pd
import json


def aggregate_results():
    logger = Logger()

    with open("./dimensionality.json") as f:
        alpha_data = json.load(f)

    with open("./prediction.json") as f:
        fev_data = json.load(f)

    with open("./representational_similarity.json") as f:
        rsa_data = json.load(f)

    with open("./semanticity.json") as f:
        semanticity_data = json.load(f)

    with open("./semanticity_stringer.json") as f:
        semanticity_stringer_data = json.load(f)

    alpha_df = pd.DataFrame(alpha_data)
    fev_df = pd.DataFrame(fev_data)
    rsa_df = pd.DataFrame(rsa_data)
    semanticity_df = pd.DataFrame(semanticity_data)
    semanticity_stringer_df = (
        pd.DataFrame(semanticity_stringer_data)
        .rename(columns={"silhouette_score": "silhouette_score_stringer"})
        .drop(columns=["model"])
    )

    # Merge all metrics on layer and n_pcs
    merged_df = fev_df.merge(alpha_df, on="layer", how="left")
    merged_df = merged_df.merge(rsa_df, on=["layer", "n_pcs"], how="left")
    merged_df = merged_df.merge(
        semanticity_df, on=["layer", "n_pcs"], how="left")
    merged_df = merged_df.merge(
        semanticity_stringer_df, on=["layer", "n_pcs"], how="left")

    # Round values
    merged_df = merged_df.round({
        "alpha": 3,
        "alpha_no_pc1": 3,
        "mean_fev": 3,
        "test_r2": 3,
        "spearman_correlation": 3,
        "silhouette_score": 3,
        "silhouette_score_stringer": 3,
    })

    # Convert NaN to None for JSON serialisation
    merged_df = merged_df.astype(object).where(pd.notnull(merged_df), None)
    merged_df.head(20)

    # Add the mouse_id to every record (useful for aggregation across multiple mice)
    mouse_id = futils.get_mouse_id()
    if "mouse_id" in merged_df.columns:
        merged_df.drop(columns=["mouse_id"], inplace=True)

    merged_df.insert(0, "mouse_id", mouse_id)

    # Add the model-target to every record (useful for aggregation across multiple models)
    model_target = futils.get_model_target()
    merged_df.insert(0, "model_target", model_target)

    # Convert to list of dicts
    flat_data = merged_df.to_dict(orient="records")

    logger.info(
        f"Saving aggregated results for mouse {mouse_id}, model-target {model_target}")

    with open(f"../results/{model_target}_{mouse_id}.json", "w") as f:
        json.dump(flat_data, f, indent=2)


def aggregate_results_gabor():
    logger = Logger()

    with open("./prediction.json") as f:
        fev_data = json.load(f)

    fev_df = pd.DataFrame(fev_data)

    # Rename 'filter' to 'layer'
    fev_df.rename(columns={"filter": "layer"}, inplace=True)

    # Add missing columns with None
    expected_columns = [
        "alpha", "alpha_no_pc1", "mean_fev", "test_r2",
        "spearman_correlation", "silhouette_score"
    ]
    for col in expected_columns:
        if col not in fev_df.columns:
            fev_df[col] = None

    # Round numeric values
    round_dict = {
        "alpha": 3,
        "alpha_no_pc1": 3,
        "mean_fev": 3,
        "test_r2": 3,
        "spearman_correlation": 3,
        "silhouette_score": 3,
    }
    fev_df = fev_df.round(
        {k: v for k, v in round_dict.items() if k in fev_df.columns})

    # Replace NaN with None
    fev_df = fev_df.astype(object).where(pd.notnull(fev_df), None)

    # Add identifiers
    mouse_id = futils.get_mouse_id()
    fev_df.insert(0, "mouse_id", mouse_id)

    model_target = futils.get_model_target()
    fev_df.insert(0, "model_target", model_target)

    flat_data = fev_df.to_dict(orient="records")

    logger.info(
        f"Saving aggregated results for mouse {mouse_id}, model-target {model_target}")

    with open(f"../results/{model_target}_{mouse_id}.json", "w") as f:
        json.dump(flat_data, f, indent=2)
