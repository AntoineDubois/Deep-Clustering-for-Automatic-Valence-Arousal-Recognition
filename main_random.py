import numpy as np
import pandas as pd

from deepemotions.utils import accuracy



path = r"data/WESAD/"


subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

NB_CLASSES = 4

results_rd = {}
for i in range(NB_CLASSES):
	results_rd["class " + str(i)] = []
results_rd["accuracy"] = []

phases_list = ["A", "D", "B", "C"]

for s in subjects:
    print("\n subject:", s)

    # physiological signals
    data = pd.read_pickle(path + s +"/"+ s +".pkl")
    labels_true = data["label"]

    # peuso-label assignment
    labels_hat = np.tile(range(1, NB_CLASSES +1), reps=int(labels_true.size/4))
    labels_hat = labels_hat[:labels_true.size]

    list_acc, over_all_accuracy = accuracy(labels_true=labels_true, labels_hat=labels_hat, classes=list(range(1, NB_CLASSES+1)))
    for i in range(NB_CLASSES):
        results_rd["class " + str(i)].append(list_acc[i])
    results_rd["accuracy"].append(over_all_accuracy)
    print(list_acc, over_all_accuracy)



results_rd = pd.DataFrame(results_rd, index=subjects)
print(results_rd)
print(results_rd.describe().round(2))

print(results_rd.describe())