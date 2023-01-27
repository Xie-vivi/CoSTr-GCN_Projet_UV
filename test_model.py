from utils.online_evaluation_utils import load_model
from data_loaders.graph import Graph
from model import CoSTrGCN
import torch

if __name__ == '__main__':
    # model params
    window_size = 30
    memory_size = 30
    stride = 1
    input_shape = (memory_size, 21, 3)
    num_classes = 14
    labels = ["D0X","B0A","B0B","G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11"]
    num_heads = 8
    d_model = 128
    n_heads = 8
    dropout_rate = .3
    dataset_name = "IPN"
    best_model_path = f"models/CoSTrGCN-model/{dataset_name}/best_model-128-8.ckpt"
    graph = torch.from_numpy(Graph(layout="IPN", strategy="distance").A)
    is_continual=False
    # load the model
    model = CoSTrGCN.load_from_checkpoint(checkpoint_path=best_model_path, is_continual=is_continual, memory_size=memory_size,
                                                adjacency_matrix=graph, labels=labels, d_model=d_model, n_heads=n_heads, num_classes=num_classes, dropout=dropout_rate)
    model.eval()


    ### prediction
    def predict_window(window):
        w = torch.tensor(window, dtype=torch.float)
        print(w.shape)
        if len(w.shape) != 3:
            print("Invalid window shape")
        if w.shape[1] != 21 or w.shape[2] != 3:
            print("Invalid window shape: each frame should contain 21 joints with their 3d coordinates")
        w = w.unsqueeze(0)
        score = model(w)
        prob = torch.nn.functional.softmax(score, dim=-1)
        score_list_labels = torch.argmax(prob, dim=-1)
        print(prob.max().item())
        # if prob[0][score_list_labels[0].item()] < thresholds[str(score_list_labels[0].item())]['threshold_avg']:
        #     return {"label":labels[0],"idx":0}

        print({"label": labels[score_list_labels[0].item()], "idx": score_list_labels[0].item()})
    window = [[[0.1,0.1,0.1] for i in range(21)] for i in range(window_size)]
    predict_window(window)