import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# Create the directory if it doesn't exist
os.makedirs('data/cervical', exist_ok=True)

# Read both train and test files
df_train = pd.read_csv('data/cervical/train.csv')
df_test = pd.read_csv('data/cervical/cervical_test.csv')

# Process training data
y_train = df_train['Biopsy'].values
X_train = df_train.drop('Biopsy', axis=1).values

# Process test data
y_test = df_test['Biopsy'].values
X_test = df_test.drop('Biopsy', axis=1).values

# Create validation set from training data
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Save numpy files
np.save('data/cervical/X_num_train.npy', X_train_final)
np.save('data/cervical/y_train.npy', y_train_final)
np.save('data/cervical/X_num_val.npy', X_val)
np.save('data/cervical/y_val.npy', y_val)
np.save('data/cervical/X_num_test.npy', X_test)
np.save('data/cervical/y_test.npy', y_test)

# Create info.json following the diabetes format
info = {
    "task_type": "binclass",
    "name": "cervical",
    "id": "cervical--id",
    "train_size": len(X_train_final),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "n_num_features": X_train.shape[1],
    "n_cat_features": 0
}

# Save info.json
with open('data/cervical/info.json', 'w') as f:
    json.dump(info, f, indent=4)

print("All files saved successfully!")
