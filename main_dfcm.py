import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deepemotions.preprocess import preprocessing
from deepemotions.questionnaire_utils import read_quest
from deepemotions.utils import resampling_labels, accuracy, init_membership
from deepemotions.deepcluster import DeepCMeams


path = r"data/WESAD/"


subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
SEQ_LEN = 20
INIT_TIME = 0
NB_CLASSES = 4
LABEL_RATE = 700 #hz
CARDIO_SAMPLING_RATE = 64 #hz


results_dfcm = {}
for i in range(NB_CLASSES):
	results_dfcm["class " + str(i)] = []
results_dfcm["accuracy"] = []


score_list = {}
score_list["v"] = []
score_list["a"] = []
score_list["phase"] = []
score_list["end"] = []
score_list["start"] = []
score_list["phase duration"] = []
score_list["subject"] = []

corr_panas_phase = {}
corr_panas_phase["value"] = []
corr_panas_phase["item"] = []
corr_panas_phase["phase"] = []

for s in subjects:
    questionnaire = read_quest(pd.read_csv(path + s +"/"+ s +"_quest.csv", sep=";"))
    for k in questionnaire.keys():
        score_list["v"].append(questionnaire[k]["va"][0])
        score_list["a"].append(questionnaire[k]["va"][1])

        score_list["phase"].append(questionnaire[k]["phase"])
        score_list["start"].append(questionnaire[k]["start"])
        score_list["end"].append(questionnaire[k]["end"])
        score_list["subject"].append(s)
        score_list["phase duration"].append(questionnaire[k]["end"] -questionnaire[k]["start"])

score = pd.DataFrame(score_list)

mask = score["phase"] == "Base"
score.loc[mask, "phase"] = "A"
mask = score["phase"] == "Fun"
score.loc[mask, "phase"] = "B"
mask = score["phase"] == "Medi 1"
score.loc[mask, "phase"] = "C"
mask = score["phase"] == "Medi 2"
score.loc[mask, "phase"] = "C"
mask = score["phase"] == "TSST"
score.loc[mask, "phase"] = "D"

av = score[["v", "a", "phase"]].groupby("phase").mean()

print(av)



phases_list = ["A", "D", "B", "C"]

for s in subjects:
    print("\n subject:", s)

    # physiological signals
    data = pd.read_pickle(path + s +"/"+ s +".pkl")
    eda = data["signal"]["wrist"]["EDA"].flatten()
    ecg = data["signal"]["wrist"]["BVP"].flatten()
    temp = data["signal"]["wrist"]["TEMP"].flatten()
    
    labels_true = data["label"]
    time_labels = np.arange(0, labels_true.size, 1) * (1/LABEL_RATE) + INIT_TIME

    signal, time = preprocessing(eda, ecg, temp, cardio_sampling_rate=CARDIO_SAMPLING_RATE, bvp=True, init_time=INIT_TIME)

    # seeding

    score_s = score[score["subject"] == s]
    # there is a little shift beween time labels and the actual start
    start_base_questionnaire = score_s[score_s["phase"] == "A"]["start"].values
    start_base_label = time_labels[np.argmax(labels_true == 1)]
    time_shift = start_base_questionnaire -start_base_label

    u_init = init_membership(score_s=score_s, time=time, nb_classes=NB_CLASSES, time_shift=time_shift, one_hot=True)
    for i in range(NB_CLASSES):
        plt.plot(time, u_init[:,i], label=phases_list[i])
    plt.plot(time_labels, labels_true, label="true labels")
    plt.legend(loc="upper left")
    plt.close()

    dfcm = DeepCMeams(signal, time)
    labels_hat, U, time_hat = dfcm.run(nb_classes=NB_CLASSES, seq_len=SEQ_LEN, u_init=u_init, epochs_pre=2, epochs_train=2, batch_size=128, 
            embedding=30, lr_pretrain=5e-5, lr_train=5e-5, sigma=1, gamma=0.5, pre_train_cluster=True)
    

    labels_true_down = resampling_labels(labels_true, labels_hat, time_labels, time_hat)

    list_acc, over_all_accuracy = accuracy(labels_true=labels_true_down, labels_hat=labels_hat, classes=list(range(1, NB_CLASSES+1)))
    for i in range(NB_CLASSES):
        results_dfcm["class " + str(i)].append(list_acc[i])
    results_dfcm["accuracy"].append(over_all_accuracy)
    print(list_acc, over_all_accuracy)

    for i in range(4):
        plt.plot(time_hat, U[:,i], label=phases_list[i])
    plt.plot(time_labels, labels_true, label="true labels")
    plt.legend(loc="upper left")
    plt.close()

results_dfcm = pd.DataFrame(results_dfcm, index=subjects)
print(results_dfcm)
print(results_dfcm.describe().round(2))

print(results_dfcm.describe())