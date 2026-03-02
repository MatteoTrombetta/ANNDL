# Orcs Disease – Challenge 2 (AN2DL) — Project Files

This `.zip` bundles the notebooks and helper files used in our **Orcs Disease** pipeline (breast cancer subtype classification from H&E slides + binary masks).

## Contents

### Notebooks
- **`OrcsDiseases_cleaning_definitive.ipynb`**  
  Dataset inspection + **outlier detection/removal** (e.g., “Shrek” non-histology images and “Blob” mucus/green artifacts) and generation/usage of file lists to clean images, masks and labels.  
  **Note:** some visualization/output cells are intentionally **commented out**; otherwise, the notebook would store a large amount of image outputs, making the file too heavy to share via email.

- **`OrcsDiseases_2nd_Challenge_Definitive.ipynb`**  
  Final end-to-end pipeline used for the leaderboard submission: patch extraction, dual-stream model (RGB + mask), training, validation, and test-time inference/aggregation.

- **`Contrastive_Learning_Definitive.ipynb`**  
  Self-supervised **contrastive pretraining** attempt (exploratory); included for completeness even if it did not improve the final score. This notebook corresponds to the appendix section of the report.

### Outlier lists (`.txt`)
These text files contain **relative dataset paths** to samples detected as outliers and removed during cleaning:
- **`new_dataset_final_list_shrek_images.txt`** — Shrek *image* paths  
- **`new_dataset_mask_shrek_images.txt`** — Shrek *mask* paths  
- **`new_dataset_final_sorted_skifidol_paths.txt`** — Blob/green artifact *image* paths  
- **`new_dataset_masks_skifidol_paths.txt`** — Blob/green artifact *mask* paths  

> The four lists together cover **110** outlier samples (60 Shrek + 50 Blob), with image/mask paths kept consistent.

### Clustering process (optional)
- **`Clustering_Process.zip`**  
  A separate folder containing scripts/configs for the clustering-related workflow (preprocessing utilities, configs, reports, etc.).  
  **Important:** *inside `Clustering_Process.zip` there is its own `README.md`* explaining how to run that pipeline and what each file/folder does.

Paths may be configured in the notebooks; in Colab we typically mount Google Drive and set a project root accordingly.

## How to use (suggested order)

1. **Run cleaning**  
   Open `OrcsDiseases_cleaning_definitive.ipynb` and run it to:
   - detect outliers via color-based matching (HSV thresholds),
   - save outlier filenames into `.txt` lists,
   - remove the corresponding images/masks and label entries.

2. **Train + infer**  
   Open `OrcsDiseases_2nd_Challenge_Definitive.ipynb` to:
   - extract patches (fixed size),
   - create a **Healthy** patch class,
   - train the dual-stream model,
   - run test-time augmentation (TTA) and slide-level aggregation,
   - export the submission `.csv`.

3. **Optional experiments**  
   Use `Contrastive_Learning_Definitive.ipynb` for the contrastive pretraining experiment and compare against the supervised baseline.

## Notes

If you relocate files, update the dataset root paths in the first configuration cell of each notebook.