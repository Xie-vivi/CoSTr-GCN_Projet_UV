import os
import sys

import torch
import numpy as np
from tqdm import tqdm
import torchmetrics
from torchmetrics.functional import jaccard_index
from data_loaders.data_loader import load_data_sets
import matplotlib.pyplot as plt
import json
from torchmetrics.functional import confusion_matrix
from model import CoSTrGCN
from ttictoc import tic, toc


class Metrics:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.f1_score= torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.avg_precision= torchmetrics.AveragePrecision(task="multiclass", num_classes=self.num_classes)
    def get_acc(self, score, labels):
        score = score.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        outputs = np.argmax(score, axis=1)
        return np.sum(outputs == labels) / float(labels.size)

    def get_fp_rate(self, scores, labels):
        stat_scores = torchmetrics.StatScores(
            task="multiclass", num_classes=self.num_classes, reduce="micro")
        TP, FP, TN, FN, SUP = stat_scores(scores, labels)
        FP = FP.type(torch.float)
        TN = TN.type(torch.float)
        FPR = FP/(TP+FN)
        return torch.mean(torch.nan_to_num(FPR), dim=-1)

    def get_jaccard(self, scores, labels):
        return jaccard_index(scores.unsqueeze(-1), labels, average="weighted", task="multiclass", num_classes=self.num_classes)

    def get_detection_rate(self, scores, labels):
        stat_scores = torchmetrics.StatScores(
            task="multiclass", num_classes=self.num_classes, reduce="micro")

        TP, FP, TN, FN, SUP = stat_scores(scores, labels)
        FP = FP.type(torch.float)
        TN = TN.type(torch.float)
        det_rate = TP/(TP+FN)
        return torch.mean(torch.nan_to_num(det_rate), dim=-1)

    def get_f1score(self,score_list, label_list):
        
        return self.f1_score(score_list, label_list)

    def get_average_precision(self, score_list, label_list):
        return self.avg_precision(score_list, label_list)


def get_window_label(label, num_classes):
    N, W = label.shape

    sum = torch.zeros((1, num_classes))
    for t in range(N):
        sum[0, label[t]] += 1
    out = sum.argmax(dim=-1)

    dominant_class_ratio = sum.max(dim=-1)[0].item() / N

    return out if dominant_class_ratio > 0.5 else torch.tensor([0])


def compute_energy(x):
    N, T, V, C = x.shape

    x_values = x[:, :, :, 0]
    y_values = x[:, :, :, 1]
    z_values = x[:, :, :, 2]
    w = None
    for v in range(V):
        w_v = None
        for t in range(1, T):
            if w_v == None:
                w_v = torch.sqrt((x_values[:, t, v]/x_values[:, t-1, v] - 1)**2 + (
                    y_values[:, t, v]/y_values[:, t-1, v] - 1)**2 + (z_values[:, t, v]/z_values[:, t-1, v] - 1)**2)
            else:
                w_v += torch.sqrt((x_values[:, t, v] / x_values[:, t - 1, v] - 1) ** 2 + (
                    y_values[:, t, v] / y_values[:, t - 1, v] - 1) ** 2 + (
                    z_values[:, t, v] / z_values[:, t - 1, v] - 1) ** 2)
        if w == None:
            w = w_v
        else:
            w += w_v
    return w


def compute_std(x, label):
    N, T, V, C = x.shape

    s = torch.mean(x, dim=1)


def velocity(x):
    N, T, V, C = x.shape

    start_x_values = x[:, 0, :, 0]
    start_y_values = x[:, 0, :, 1]
    start_z_values = x[:, 0, :, 2]
    end_x_values = x[:, -1, :, 0]
    end_y_values = x[:, -1, :, 1]
    end_z_values = x[:, -1, :, 2]
    distance = torch.sqrt((end_x_values-start_x_values)**2 +
                          (end_y_values-start_y_values)**2+(end_z_values-start_z_values)**2)

    v = distance / T
    return v


def plot_confusion_matrix(cnf_matrix, labels, filename, mode="eps", eps=1e-5):
    import seaborn as sn
    confusion_matrix_sum_vec = torch.sum(cnf_matrix, dim=1) + eps

    confusion_matrix_percentage = (
        cnf_matrix / confusion_matrix_sum_vec.view(-1, 1))

    plt.figure(figsize=(20, 18))
    sn.heatmap(confusion_matrix_percentage.cpu().numpy(), annot=True,
               cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    sn.set(font_scale=1.4)
    plt.savefig(filename, format=mode)
    plt.close()


def init_data_loader(window_size, dataset_name, batch_size=32, workers=4):
    train_loader, val_loader, test_loader, graph, labels = load_data_sets(
        dataset_name=dataset_name,
        window_size=window_size,
        batch_size=batch_size,
        workers=workers,
        is_segmented=True,
        binary_classes=False,
        use_data_aug=False,
        use_aug_by_sw=False
    )

    return train_loader, val_loader, test_loader, graph, labels


def load_model(model_path, graph, memory_size, labels, num_classes, dropout_rate, d_model, num_heads):
    model = CoSTrGCN.load_from_checkpoint(checkpoint_path=model_path, is_continual=True, memory_size=memory_size,
                                          adjacency_matrix=graph, labels=labels, d_model=d_model, n_heads=num_heads, num_classes=num_classes, dropout=dropout_rate)
    model.eval()
    return model


def online_test_loop(model_path,
                     window_size,
                     dataset_name,
                     num_classes,
                     stride,
                     memory_size,
                     dropout_rate,
                     d_model, 
                     num_heads
                     ):

    cnf_matrix = None
    print(os.getcwd())

    with open(os.path.join(os.getcwd(),'thresholds.json'), mode="r") as f:
        thresholds = json.load(f)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _, _, test_loader, graph, labels = init_data_loader(
        window_size, dataset_name)

    # .........inital model
    print("\n loading model.............")
    # model = load_model(model_path, graph, memory_size,
    #                    labels, num_classes, dropout_rate, d_model, num_heads)

    # model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # # ........set loss
    criterion = torch.nn.CrossEntropyLoss()

    # parameters recording training log

    metrics = Metrics(num_classes)

    #         # ***********evaluation***********
    print("*"*10, "Testing", "*"*10)

    with torch.no_grad():

        val_loss = 0
        val_f1 = 0
        val_jaccard = 0
        val_fp_rate = 0
        val_avg_precision = 0
        val_det_rate = 0
        score_list = None
        label_list = None
        window_label_list = None
        window_score_list = None
        val_jaccard_epoch = 0
        val_fp_rate_epoch = 0
        val_avg_precision_epoch = 0
        val_f1_epoch = 0
        val_det_rate_epoch = 0
        for i, batch in tqdm(enumerate(test_loader)):
            x, y, index = batch

            y = torch.stack(y)
            N, T, V, C = x.shape

            score_list = None
            label_list = None
            window_score_list = None
            window_label_list = None
            exec_time_total = 0
            for t in tqdm(range(0, T-window_size, stride), leave=False):
                window = x[:, t:t+window_size]

                label_l = y[t:t+window_size]
                label = get_window_label(label_l, num_classes)

                tic()
                score = model(window)

                prob = torch.nn.functional.softmax(score, dim=-1)

                score_list_labels = torch.argmax(prob, dim=-1)

                if prob[0][score_list_labels[0].item()] < thresholds[str(score_list_labels[0].item())]['threshold_avg']:
                    score[0][0] = 10.
                exec_time = toc()
                exec_time_total += exec_time
                prob = torch.nn.functional.softmax(score, dim=-1)

                score_list_labels = torch.argmax(prob, dim=-1)

                if score_list is None:
                    score_list = score
                    label_list = label
                    window_label_list = label_l.unsqueeze(0)
                    window_score_list = torch.cat(
                        [score_list_labels for _ in label_l]).unsqueeze(0)
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)
                    window_label_list = torch.cat(
                        (window_label_list, label_l.unsqueeze(0)), 0)
                    window_score_list = torch.cat((window_score_list, torch.cat(
                        [score_list_labels for _ in label_l]).unsqueeze(0)), 0)

            if cnf_matrix == None:
                cnf_matrix = confusion_matrix(score_list.detach().cpu(
                ), label_list.detach().cpu(), task="multiclass", num_classes=num_classes, )
            else:
                cnf_matrix += confusion_matrix(score_list.detach().cpu(
                ), label_list.detach().cpu(), task="multiclass", num_classes=num_classes, )
            plot_confusion_matrix(cnf_matrix, labels,
                                  "./cnf_matrix.png", mode="png")
            plot_confusion_matrix(cnf_matrix, labels,
                                  "./cnf_matrix.eps", mode="eps")
            loss = criterion(score_list.detach().cpu(),
                             label_list.detach().cpu())
            score_list_labels = torch.argmax(
                torch.nn.functional.softmax(score_list, dim=-1), dim=-1)
            val_f1_step = metrics.get_f1score(
                score_list.detach().cpu(), label_list.detach().cpu())
            val_jaccard_step = metrics.get_jaccard(
                window_score_list.detach().cpu(), window_label_list.detach().cpu())
            val_fp_rate_step = metrics.get_fp_rate(
                score_list.detach().cpu(), label_list.detach().cpu())
            val_detection_rate = metrics.get_detection_rate(
                score_list.detach().cpu(), label_list.detach().cpu())
            val_avg_precision_step = metrics.get_average_precision(
                score_list.detach().cpu(), label_list.detach().cpu())
            val_f1_epoch += val_f1_step
            val_jaccard_epoch += val_jaccard_step
            val_fp_rate_epoch += val_fp_rate_step
            val_avg_precision_epoch += val_avg_precision_step
            val_det_rate_epoch += val_detection_rate
            val_loss += loss
            print("*** SHREC  21"
                  "val_loss_step: %.6f,"
                  "val_F1_step: %.6f ***,"
                  "val_jaccard_step: %.6f ***"
                  "val_fp_rate_step: %.6f ***"
                  "val_avg_precision_step: %.6f ***"
                  "val_detection_rate_step: %.6f ***"
                  "average_exec_time: %.6f ***"
                  % (loss, val_f1_step, val_jaccard_step, val_fp_rate_step, val_avg_precision_step, val_detection_rate, exec_time_total / t))

            val_loss = val_loss / (float(i + 1))
            val_f1 = val_f1_epoch.item() / (float(i + 1))
            val_jaccard = val_jaccard_epoch / (float(i + 1))
            val_fp_rate = val_fp_rate_epoch / (float(i + 1))
            val_avg_precision = val_avg_precision_epoch / (float(i + 1))
            val_det_rate = val_det_rate_epoch / (float(i + 1))
            print("*** SHREC 21, "
                  "val_loss: %.6f,"
                  "val_F1: %.6f ***,"
                  "val_jaccard: %.6f ***"
                  "val_fp_rate: %.6f ***"
                  "val_avg_precision_rate: %.6f ***"
                  "val_detection_rate: %.6f ***"
                  % (val_loss, val_f1, val_jaccard, val_fp_rate, val_avg_precision, val_det_rate))
