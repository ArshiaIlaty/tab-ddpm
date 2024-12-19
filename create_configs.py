import json

# CatBoost config matching diabetes format
catboost_config = {
    "learning_rate": 0.026561564197335047,
    "depth": 3,
    "l2_leaf_reg": 0.8066571920706246,
    "bagging_temperature": 0.6363246451815178,
    "leaf_estimation_iterations": 9,
    "iterations": 2000,
    "early_stopping_rounds": 50,
    "od_pval": 0.001,
    "task_type": "CPU",
    "thread_count": 4,
    "cat_features": []
}

# MLP config matching diabetes format
mlp_config = {
    "lr": 0.0001316285155465466,
    "dropout": 0.30802808938889303,
    "weight_decay": 0.0,
    "d_layers": [
        128,
        512
    ]
}

# Save the configs
with open('tuned_models/catboost/cervical_cv.json', 'w') as f:
    json.dump(catboost_config, f, indent=4)

with open('tuned_models/mlp/cervical_cv.json', 'w') as f:
    json.dump(mlp_config, f, indent=4)
