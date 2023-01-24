import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random
class Feeder_IPN(Dataset):
    """
    Feeder for skeleton-based gesture recognition in ipn dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(
            self,
            data_path="IPN",
            set_name="training",
            window_size=10,
            aug_by_sw=False,
            is_segmented=False,
            binary_classes=False,
            num_joint=21
    ):
        self.data_path = data_path
        self.set_name = set_name
        self.classes = [
                      "D0X","B0A","B0B","G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11"
                        ]
        self.class_to_idx = {class_l: idx for idx,
                             class_l in enumerate(self.classes)}
        self.window_size = window_size
        self.aug_by_sw = aug_by_sw
        self.num_joint=num_joint
        self.is_segmented = is_segmented
        self.binary_classes=binary_classes
        self.load_data()

    def load_data(self):
        self.dataset = []
        # load file list
        # classes = set([''])
        # self.classes = []
        with open(
                f'{self.data_path}/{self.set_name}_set/Annot_TestList.txt' if self.set_name == "test" else f'{self.data_path}/{self.set_name}_set/Annot_TrainList.txt',
                mode="r") as f:
                

            file_getsture = {}
            for line in f:
                fields = line.strip().split(',')
                seq_idx = fields[0] # file name
                # gestures = fields[1:-1] # 
                # nb_gestures = len(gestures) // 3
                gesture_start = int(fields[-3])-1
                gesture_end = int(fields[-2])-1
                gesture_label = int(fields[-4]) -1
                if (file_getsture.__contains__(seq_idx)):
                    file_getsture[seq_idx].append((gesture_start, gesture_end, gesture_label))
                else:
                    file_getsture[seq_idx] = [(gesture_start, gesture_end, gesture_label)]
                    # classes.add(gesture_label)
            for seq_idx, gesture_infos in file_getsture.items():
                self.dataset.append((seq_idx, gesture_infos))

        # self.classes = list(classes)
        # with open('datasets/shrec21/classes.yaml', mode="w") as f:
        #     yaml.dump(self.classes, f, explicit_start=True, default_flow_style=False)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        def parse_seq_data(src_file):
            '''
            Retrieves the skeletons sequence for each gesture
            '''
            video = []
            mode = "pos"
            for line in src_file:

                line = line.split("\n")[0]
                if line==';':
                    video.append([[0.0, 0.0, 0.0] for i in range(21)])
                    continue

                data = line.split(";")[:-1]
                pos = []
                positions = []
                for data_ele in data:
                    pos.append(float(data_ele))
                    if len(pos)==3:
                        positions.append(pos)
                        pos = []
                video.append(positions)
            return np.array(video)
        def sample_window(data_num, stride):
            # sample #window_size frames from whole video

            sample_size = self.window_size

            idx_list = [0, data_num - 1]
            for i in range(sample_size):
                if index not in idx_list and index < data_num:
                    idx_list.append(index)
            idx_list.sort()

            while len(idx_list) < sample_size:
                idx = random.randint(0, data_num - 1)
                if idx not in idx_list:
                    idx_list.append(idx)
            idx_list.sort()
            return idx_list

        def get_segmented(seq_idx, gesture_infos):
            with open(f'{self.data_path}/sequences/{seq_idx}.txt', mode="r") as seq_f:
                sequence = parse_seq_data(seq_f)
            labeled_sequence = [(f, 0) for f in sequence]
            for gesture_start, gesture_end, gesture_label in gesture_infos:
                labeled_sequence = [
                    (np.array(f), gesture_label if int(gesture_start) <=
                     idx <= int(gesture_end) and label == 0 else label)
                    for
                    idx, (f, label) in enumerate(labeled_sequence)]

            frames = [f for f, l in labeled_sequence]
            labels_per_frame = [int(l) for f, l in labeled_sequence]
            
            gestures = []
            windows_sub_sequences_per_gesture = {
                i: [] for i in range(len(self.classes))}

            for gesture_start, gesture_end, gesture_label in gesture_infos:
                if gesture_label == 0:
                    continue
                gesture_start = int(gesture_start)
                gesture_end = int(gesture_end)
                g_frames = frames[gesture_start:gesture_end]
                g_label = labels_per_frame[gesture_start:gesture_end]
                gestures.append((g_frames, g_label))
                label = int(gesture_label)
                if self.aug_by_sw:
                    num_windows = len(g_frames) // self.window_size

                    for stride in range(1, self.window_size):
                        l = len(g_frames)
                        if l // stride < self.window_size:
                            break
                        window_indices = sample_window(l, stride)
                        window = [g_frames[idx] for idx in window_indices]
                        windows_sub_sequences_per_gesture[label].append(
                            (window, label))

            ng_sequences = []
            ng_seq = []
            l = len(frames)
            indices_ng = []
            for i in range(len(frames)-1):
                f_curr = frames[i]
                f_next = frames[i+1]
                l_curr = labels_per_frame[i]
                l_next = labels_per_frame[i+1]

                if l_curr == 0 and l_next == 0:
                    indices_ng.append(i)
                    ng_seq.append(f_curr)
                    if i == l-2:
                        ng_seq.append(f_next)
                        ng_sequences.append((ng_seq, 0))
                        ng_seq = []
                        continue
                elif l_curr == 0 and l_next != 0:
                    indices_ng.append(i)
                    ng_seq.append(f_curr)
                    ng_sequences.append((ng_seq, 0))
                    ng_seq = []
                    continue
            return gestures, ng_sequences, windows_sub_sequences_per_gesture

        seq_idx, gesture_infos = self.dataset[index]
        return get_segmented(seq_idx, gesture_infos)


def get_window_label(label, num_classes=18):

    W = len(label)
    sum = torch.zeros((num_classes))
    for t in range(W):
        sum[label[t]] += 1
    return sum.argmax(dim=-1).item()


def gendata(
        data_path,
        set_name,
        max_frame,
        window_size=20,
        num_joint=21,
        aug_by_sw=False,
        is_segmented=False,
        binary_classes=False,
        
):
    feeder = Feeder_IPN(
        data_path=data_path,
        set_name=set_name,
        window_size=window_size,
        aug_by_sw=aug_by_sw,
        num_joint=num_joint,
        is_segmented=is_segmented,
        binary_classes=binary_classes
    )
    dataset = feeder.dataset
    if is_segmented:
        data = []
        ng_sequences_data = []
        windows_sub_sequences_data = {i: []
                                      for i in range(len(feeder.classes))}
        for i, s in enumerate(tqdm(dataset)):
            data_el, ng_sequences, windows_sub_sequences_per_gesture = feeder[i]
            ng_sequences_data = [*ng_sequences_data, *ng_sequences]
            l = len(data_el)
            # for w in range(num_windows):
            for idx, gesture in enumerate(data_el):
                current_skeletons_window = np.array(gesture[0])
                label = gesture[1]
                label = get_window_label(label,len(feeder.classes))
                windows_sub_sequences_data[label] = [
                    *windows_sub_sequences_data[label], *windows_sub_sequences_per_gesture[label]]
                for sub_g in windows_sub_sequences_per_gesture[label] :
                    data.append(sub_g)
                data.append((current_skeletons_window,label))    

        return data, ng_sequences_data, windows_sub_sequences_data
    else:
        data = []
        for i, s in enumerate(tqdm(dataset)):
            labeled_seq, frames, labels = feeder[i]
            data.append((frames, labels))

        return data