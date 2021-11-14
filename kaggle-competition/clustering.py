import pickle
from functools import partial
from itertools import permutations


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score as measure_f1_score
from sklearn.cluster import SpectralClustering, KMeans

LABELS = ['lipsync', 'reenactment', 'swap', 'original']


measure_f1_score = partial(measure_f1_score, average='micro')
embeddings = np.load('embeddings.npy')[:1000]
filenames = pd.read_csv('filenames.csv')._values[:,0][:1000]

def clustering(embeddings, filenames):
    sc = KMeans(n_clusters=4)
    sc.fit_predict(embeddings)
    with open('SpectralClustering', 'wb') as f:
        pickle.dump(sc, f)
    pred_labels = sc.labels_
    return np.stack([filenames, pred_labels], axis=1)



def read_label(path):
    df = pd.read_csv(path)
    arr = df._values
    arr = np.concatenate([arr, -1 * np.ones([arr.shape[0], 1])], axis=1)
    for i, label in enumerate(LABELS):
        mask = arr[:, 1] == label
        arr[:, 2][mask] = i
    return arr[:, [0, 2]]



def compute_f1(true_label, pred_label):
    mask = np.in1d(true_label[:, 0], pred_label[:, 0])
    true_label = true_label[mask]
    true_label = true_label[true_label[:, 0].argsort()]
    pred_label = pred_label[pred_label[:, 0].argsort()]
    true_label = true_label[:, 1].astype(np.float)
    pred_label = pred_label[:, 1].astype(np.float)
    f1 = measure_f1_score(true_label, pred_label)
    print(f1)


if __name__ == "__main__":
    pred_labels = clustering(embeddings, filenames)
    true_labels = read_label('dataset/file_mapping.csv')
    scores = compute_f1(true_labels, pred_labels)
    print(scores)