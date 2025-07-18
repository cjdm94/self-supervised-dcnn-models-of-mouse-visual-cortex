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

    alpha_df = pd.DataFrame(alpha_data)
    fev_df = pd.DataFrame(fev_data)
    rsa_df = pd.DataFrame(rsa_data)
    semanticity_df = pd.DataFrame(semanticity_data)

    # Merge all metrics on layer and n_pcs
    merged_df = fev_df.merge(alpha_df, on="layer", how="left")
    merged_df = merged_df.merge(rsa_df, on=["layer", "n_pcs"], how="left")
    merged_df = merged_df.merge(
        semanticity_df, on=["layer", "n_pcs"], how="left")

    # Round values
    merged_df = merged_df.round({
        "alpha": 3,
        "alpha_no_pc1": 3,
        "mean_fev": 3,
        "test_r2": 3,
        "spearman_correlation": 3,
        "silhouette_score": 3,
    })

    # Convert NaN to None for JSON serialisation
    merged_df = merged_df.astype(object).where(pd.notnull(merged_df), None)
    merged_df.head(20)

    # Add the mouse_id to every record (useful for aggregation across multiple mice)
    mouse_id = futils.get_mouse_id()
    merged_df.insert(0, "mouse_id", mouse_id)

    # Add the model-target to every record (useful for aggregation across multiple models)
    model_target = futils.get_model_target()
    merged_df.insert(0, "model_target", model_target)

    # Convert to list of dicts
    flat_data = merged_df.to_dict(orient="records")

    logger.info(
        f"Saving aggregated results for mouse {mouse_id}, model-target {model_target}")

    with open(f"./{model_target}_{mouse_id}.json", "w") as f:
        json.dump(flat_data, f, indent=2)
