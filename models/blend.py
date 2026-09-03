import pandas as pd

print("Starting the final model blend...")

XGB_PATH = 'submission_xgb.csv'
GRU_PATH = 'submission_gru.csv'
OUTPUT_PATH = 'submission_blend.csv'

try:
    xgb_submission = pd.read_csv(XGB_PATH)
    gru_submission = pd.read_csv(GRU_PATH)
except FileNotFoundError as error:
    raise SystemExit(
        f"Missing input submission: {error.filename}. Run the XGBoost and GRU pipelines first."
    ) from error

print("Submissions loaded successfully.")

# Sanity checks: model outputs must refer to the same ordered rows.
if xgb_submission.shape != gru_submission.shape:
    raise SystemExit("Submission shapes do not match.")

if not xgb_submission['id'].equals(gru_submission['id']):
    raise SystemExit("Submission IDs do not match in both value and order.")

# 50/50 competition-time ensemble. This is a fixed artifact choice, not a
# claim that equal weighting is generally optimal out of sample.
blend_submission = xgb_submission.copy()
targets = ['target_short', 'target_medium', 'target_long']

print("Blending targets...")
for target in targets:
    blend_submission[target] = (
        xgb_submission[target] * 0.5 + gru_submission[target] * 0.5
    )

print("\nFinal blended submission summary:")
print(blend_submission[targets].describe())

blend_submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nBlend complete: {OUTPUT_PATH}")
