python -m train \
    --experiment_name WLASL100 \
    --training_set_path datasets/WLASL100_train_25fps.csv \
    --testing_set_path datasets/WLASL100_val_25fps.csv \
    --validation_set split-from-train \
    --num_classes 100
