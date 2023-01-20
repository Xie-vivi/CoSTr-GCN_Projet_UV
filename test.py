from utils import training_loop


if __name__ == "__main__":
    batch_size = 32
    window_size = 30
    is_gesture_nogesture_model = False
    num_classes = 2 if is_gesture_nogesture_model else 18
    d_model = 128
    n_heads = 8
    Max_Epochs = 500
    Early_Stopping = 15
    lr=1e-3
    dropout_rate = .3
    # trained_models_path= "./models/CoSTrGCN-model/SHREC21"
    trained_models_path= "./models/CoSTrGCN-model/IPN"


    model_params=(
        window_size,
        d_model,
        n_heads,
        dropout_rate,
        is_gesture_nogesture_model
    )
    training_params=(
        batch_size,
        Max_Epochs,
        Early_Stopping,
        lr,
        trained_models_path
    )
    training_loop(model_params, training_params, num_classes, dataset_name="IPN")