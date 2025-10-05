import torch
import torch.nn.functional as F
import time
from statistics import mean
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import logger


def calc_total_params(model = None):
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logger(f"Total params: {total_params:,}")
        logger(f"Trainable params: {trainable_params:,}")
        logger(f"Frozen params: {frozen_params:,}")


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    pred_correct, pred_all = 0, 0
    running_loss = 0.0
    train_time_sec_list = []

    for i, data in enumerate(dataloader):
        l_hands, r_hands, bodies, labels = data

        l_hands = l_hands.to(device)
        r_hands = r_hands.to(device)
        bodies = bodies.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        start_time = time.time()

        outputs = model(l_hands, r_hands, bodies, training=True)

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        # print(f'preds: {preds}')
        # print(f'label: {labels.view(-1)}')
        pred_correct += torch.sum(preds == labels.view(-1)).item()
        pred_all += labels.size(0)

    if scheduler:
        scheduler.step()

    avg_train_time = mean(train_time_sec_list)

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all), avg_train_time


def compute_early_exit_stats(model, dataloader, device):
    early_exit_total = dec_early_exit_total = 0
    _1_stream=0
    _2_stream=0
    _3_stream=0
    full_deepth=0

    lh_stream, rh_stream, b_stream=0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)
            batch_size = l_hands.size(0)
            total_samples += batch_size

            for j in range(batch_size):
                sample=0
                model.set_early_exit_stats()
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                _ = model(l_hand, r_hand, body, training=False)

                sum, (el, er, eb) = model.get_early_exit_stats()
                if el:
                    sample+=1
                    lh_stream+=1
                if er:
                    rh_stream+=1
                    sample+=1
                if eb:
                    b_stream+=1
                    sample+=1
                
                if sample==1:
                    _1_stream+=1
                elif sample==2:
                    _2_stream+=1
                elif sample==3:
                    _3_stream+=1
                else:
                    full_deepth+=1

                dec_early_exit_total+=1 if model.decoder_is_exit_early() else 0

                early_exit_total+= 1 if sum == True else 0
    
    ratio = early_exit_total / total_samples if total_samples > 0 else 0

    logger(f'ENCODER')
    logger(f'Exit in 1 stream {_1_stream}')
    logger(f'Exit in 2 stream {_2_stream}')
    logger(f'Exit in 3 stream {_3_stream}')
    logger(f'Full deepth {full_deepth}')
    logger(f'Exit by lh_stream {lh_stream}')
    logger(f'Exit by rh_stream {rh_stream}')
    logger(f'Exit by b_stream {b_stream}')

    logger(f'DECODER: {dec_early_exit_total}/{total_samples}')

    return early_exit_total, total_samples, ratio


def evaluate(model, dataloader, device):
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(100)}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)  # [24, 204, 21, 2]
            r_hands = r_hands.to(device)  # [24, 204, 21, 2]
            bodies = bodies.to(device)  # [24, 204, 12, 2]
            labels = labels.to(device, dtype=torch.long)  # [24, 1]

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                output = model(l_hand, r_hand, body, training=False)
                output = output.unsqueeze(0).expand(1, -1, -1)

                # Statistics
                if int(torch.argmax(torch.nn.functional.softmax(output, dim=2))) == int(label):
                    stats[int(label)][0] += 1
                    pred_correct += 1

                stats[int(label)][1] += 1
                pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)


def ConfusionMatrix(model, dataloader, device):
    y_true = []
    y_pred = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)  # [24, 204, 21, 2]
            r_hands = r_hands.to(device)  # [24, 204, 21, 2]
            bodies = bodies.to(device)  # [24, 204, 12, 2]
            labels = labels.to(device, dtype=torch.long)  # [24, 1]

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                output = model(l_hand, r_hand, body, training=False) # [1, num_classes]
                pred_class = torch.argmax(output, dim=1).item()

                y_true.append(int(label))
                y_pred.append(pred_class)

    # pred_correct = 0

    stats = {i: [0, 0] for i in range(100)}
    for i, v in enumerate(y_true):
        if v == y_pred[i]:
            pred_correct+=1
            stats[v][0]+=1
        stats[v][1]+=1
    
    # count_pred_correct ={i: 0 for i in range(9)}

    # for key in stats.keys():
    #     count_pred_correct[stats[key][0]] += 1

    # print(count_pred_correct)

    print(pred_correct / 800)


    # cm = confusion_matrix(y_true, y_pred, labels=np.arange(100))
    # plt.figure(figsize=(14, 12))
    # sns.heatmap(cm, cmap="Blues", cbar=True, square=True, annot=True, fmt='d',
    #             xticklabels=False, yticklabels=False)  # tắt label nếu có quá nhiều lớp
    # plt.title("Confusion Matrix (Normalized) - WLASL100", fontsize=16)
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.show()

    # print(classification_report(y_true, y_pred, digits=6))

def evaluate_top_k(model, dataloader, device, k=5):
    pred_correct, pred_all = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                output = model(l_hand, r_hand, body, training=False)
                output = output.unsqueeze(0).expand(1, -1, -1)

                topK= torch.topk(output, k).indices.flatten().tolist()

                _ = label[0]

                # Statistics
                if int(label[0]) in topK:
                    pred_correct += 1

                pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)


def get_sequence_list(num):
    if num == 0:
        return [0]

    result, i = [1], 2
    while sum(result) != num:
        if sum(result) + i > num:
            for j in range(i - 1, 0, -1):
                if sum(result) + j <= num:
                    result.append(j)
        else:
            result.append(i)
        i += 1

    return sorted(result, reverse=True)