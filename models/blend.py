import pandas as pd
import numpy as np

print(" Starting the Final Model Blend...")


try:
    xgb_submission = pd.read_csv('submission_xgb_upgraded.csv')
    gru_submission = pd.read_csv('submission_jane_street_gru.csv')
except FileNotFoundError:
    print(" ERROR: Make sure both 'submission_xgb_upgraded.csv' and 'submission_jane_street_gru.csv' exist!")
    exit()

print("Submissions loaded successfully.")

# Sanity check: ensure they have the same shape and IDs
if not xgb_submission.shape == gru_submission.shape:
    print(" ERROR: Submission shapes do not match!")
    exit()

if not all(xgb_submission['id'] == gru_submission['id']):
    print(" ERROR: Submission IDs do not match!")
    exit()

# 2. CREATE THE BLENDED PREDICTIONS

# We will create a simple 50/50 average. This is the most robust and common blend.
blend_submission = xgb_submission.copy() 

targets = ['target_short', 'target_medium', 'target_long']

print("Blending targets...")
for target in targets:
    # Blend = 50% XGBoost + 50% GRU
    blend_submission[target] = (xgb_submission[target] * 0.5) + (gru_submission[target] * 0.5)

# 3. FINAL SANITY CHECK & SAVE

print("\n Final Blended Submission Summary:")
print(blend_submission[targets].describe())


blend_submission.to_csv('submission_final_blend2.csv', index=False)

print("\n BLEND COMPLETE!")
print("Submit the file: 'submission_final_blend2.csv'")