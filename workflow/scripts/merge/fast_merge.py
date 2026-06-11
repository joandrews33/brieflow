import json

import pandas as pd
import numpy as np

from lib.shared.file_utils import validate_dtypes
from lib.merge.fast_merge import merge_triangle_hash

# Load phenotype and sbs info with cell locations
phenotype_info = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_info = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Load alignment data
fast_alignment = pd.read_parquet(snakemake.input[2])

# Get transform model type
transform_model = getattr(snakemake.params, "transform_model", "linear")

if transform_model == "polynomial2":
    # Polynomial: rotation_1 stores JSON-serialized model dict
    fast_alignment["rotation"] = fast_alignment["rotation_1"].apply(
        lambda r: json.loads(r) if isinstance(r, str) and r.startswith("{") else r
    )
else:
    # Linear: reconstruct rotation matrix from two rows
    fast_alignment["rotation"] = fast_alignment.apply(
        lambda row: np.array([row["rotation_1"], row["rotation_2"]]), axis=1
    )
fast_alignment.drop(columns=["rotation_1", "rotation_2"], inplace=True)

# Filter alignment data based on parameters
fast_alignment_filtered = fast_alignment[
    (fast_alignment["determinant"] >= snakemake.params.det_range[0])
    & (fast_alignment["determinant"] <= snakemake.params.det_range[1])
    & (fast_alignment["score"] > snakemake.params.score)
]

print(f"Transform model: {transform_model}")
print(f"Total alignments: {len(fast_alignment)}")
print(f"Filtered alignments: {len(fast_alignment_filtered)}")

# Merge cells across well
merge_data = []
for index, alignment_row in fast_alignment_filtered.iterrows():
    # Determine tiles and sites for merging
    phenotype_tile = alignment_row["tile"]
    sbs_site = alignment_row["site"]

    # Filter phenotype and sbs info to the relevant well and tile for merging
    phenotype_info_filtered = phenotype_info[phenotype_info["tile"] == phenotype_tile]
    sbs_info_filtered = sbs_info[sbs_info["tile"] == sbs_site]

    # Merge cells for row of alignment data
    alignment_row_merge = merge_triangle_hash(
        phenotype_info_filtered,
        sbs_info_filtered,
        alignment_row,
        threshold=snakemake.params.threshold,
        transform_model=transform_model,
    )
    merge_data.append(alignment_row_merge)

# Compile merge data
merge_data = pd.concat(merge_data, ignore_index=True)
print(f"Merge completed: {len(merge_data)} cells merged")

# DAPI validation (optional post-filter)
dapi_validation = getattr(snakemake.params, "dapi_validation", False)
if dapi_validation and not merge_data.empty:
    from lib.merge.dapi_validation import validate_matches_dapi

    merge_data, dapi_stats = validate_matches_dapi(
        merge_data,
        root_fp=snakemake.params.root_fp,
        plate=str(merge_data["plate"].iloc[0]),
        well=str(merge_data["well"].iloc[0]),
        phenotype_dapi_index=snakemake.params.phenotype_dapi_index,
        sbs_dapi_index=snakemake.params.sbs_dapi_index,
        sbs_dapi_cycle=snakemake.params.sbs_dapi_cycle,
        min_correlation=getattr(snakemake.params, "dapi_min_correlation", 0.5),
        crop_size=getattr(snakemake.params, "dapi_crop_size", 32),
    )
    print(f"After DAPI validation: {len(merge_data)} cells")

merge_data.to_parquet(snakemake.output[0])
