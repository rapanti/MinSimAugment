import os

import numpy as np
import pandas as pd

from analysis.plot_utils import read_file


def extract_list(params):
    out = []
    for x in params:
        if isinstance(x, (list, tuple)):
            out.extend(extract_list(x))
        else:
            out.append(x)
    return out


def extract_params(raw):
    rrc = extract_list(raw[0])
    # rrc = [rrc[2] / rrc[0], rrc[3] / rrc[1], rrc[4] / rrc[0], rrc[5] / rrc[1]]
    jitter = extract_list(raw[1]) if raw[1] else [0, 0, 0, 0, 0, 0, 0, 0]
    gray = 1 if raw[2] else 0
    blur = raw[3] if raw[3] else 0
    hf = 1 if raw[4] else 0
    return *rrc, *jitter, gray, blur, hf


def convert_params_to_list(raw, epoch):
    out = []
    bs = len(raw[0][0])
    for mini_batch in raw:
        for i in range(bs):
            temp = []
            for j in range(len(mini_batch)):
                extracted = extract_params(mini_batch[j][i])
                temp.extend(extracted)
            temp.append(epoch)
            out.append(temp)
    return out


def convert_selected_to_int(raw):
    out = []
    for mini_batch in raw:
        raw = np.array(mini_batch)
        for sample in raw.T:
            match list(sample):
                case [0, 1]:
                    n = 0
                case [0, 2]:
                    n = 1
                case [0, 3]:
                    n = 2
                case [1, 2]:
                    n = 3
                case [1, 3]:
                    n = 4
                case [2, 3]:
                    n = 5
            out.append(n)
    return out


def json_to_pandas(json_file):
    X = []
    Y = []
    for line in json_file:
        epoch = line['epoch']
        params = convert_params_to_list(line['params'], epoch)
        X.extend(params)
        Y.extend(convert_selected_to_int(line['selected']))
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    w_dir = "../../../metrics-data"
    data = read_file(os.path.join(w_dir, "simsiam-minsim-collect_metrics-resnet50-ImageNet-ep100-bs256-select_cross-ncrops4-lr0.05-wd0.0001-mom0.9-seed2-metrics.json"))

    X_array, Y_array = json_to_pandas(data)

    print(X_array.shape)
    print(Y_array.shape)

    df_X = pd.DataFrame(X_array)
    df_Y = pd.DataFrame(Y_array)

    df_X.to_csv(os.path.join(w_dir, "X-simsiam-minsim-seed2.csv"), index=False)
    df_Y.to_csv(os.path.join(w_dir, "Y-simsiam-minsim-seed2.csv"), index=False)
