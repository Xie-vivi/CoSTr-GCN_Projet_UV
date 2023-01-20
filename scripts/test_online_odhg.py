from utils import online_test_loop




# model params
window_size = 50
memory_size = 50
stride = 5
input_shape = (memory_size, 20, 3)
num_classes = 18

#  change this if you train a new model with other hyperparams
num_heads = 8
d_model = 128
n_heads = 8
dropout_rate = .3
dataset_name = "ODHG"

model_path = f"models\\CoSTrGCN-model\\{dataset_name}\\best_model-128-8-v1.ckpt"


if __name__ == "__main__":
    online_test_loop(model_path,
                     window_size,
                     dataset_name,
                     num_classes,
                     stride,
                     memory_size,
                     dropout_rate,
                     d_model,
                     num_heads
                     )
