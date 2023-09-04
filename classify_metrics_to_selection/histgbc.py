import os
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split


if __name__ == "__main__":
    w_dir = "../../metrics-data"
    X_0 = pd.read_csv(os.path.join(w_dir, "X-simsiam-minsim-normalized-seed0.csv"))
    print(X_0.shape)
    X_1 = pd.read_csv(os.path.join(w_dir, "X-simsiam-minsim-normalized-seed1.csv"))
    print(X_1.shape)
    X_2 = pd.read_csv(os.path.join(w_dir, "X-simsiam-minsim-normalized-seed2.csv"))
    print(X_2.shape)
    X = np.concatenate((X_0, X_1, X_2), axis=0)
    print(X.shape)
    Y_0 = pd.read_csv(os.path.join(w_dir, "Y-simsiam-minsim-normalized-seed0.csv")).squeeze()
    print(Y_0.shape)
    Y_1 = pd.read_csv(os.path.join(w_dir, "Y-simsiam-minsim-normalized-seed1.csv")).squeeze()
    print(Y_1.shape)
    Y_2 = pd.read_csv(os.path.join(w_dir, "Y-simsiam-minsim-normalized-seed2.csv")).squeeze()
    print(Y_2.shape)
    Y = np.concatenate((Y_0, Y_1, Y_2), axis=0)
    print(Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf = HistGradientBoostingClassifier(verbose=True)
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    scores = cross_val_score(clf, X, Y, cv=5, verbose=1)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    # 0.41 accuracy with a standard deviation of 0.01
