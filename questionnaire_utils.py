import numpy as np



def read_quest(data):
	dic = {}

	# order of conditions
	mask = data.iloc[:,0] == "# ORDER"
	label = data[mask].iloc[:,1:6].values.tolist()
	label = label[0]

	idx_label = []
	for index, la in enumerate(label):
		dic[la] = {}
		dic[la]["phase"] = la
		dic[la]["start"] = data.iloc[1, index +1]
		a = dic[la]["start"].split('.')
		if len(a) == 1:
			dic[la]["start"] = int(a[0]) * 60
		elif len(a) == 2:
			dic[la]["start"] = int(a[0]) * 60 + int(a[1])
		
		dic[la]["end"] = data.iloc[2, index +1]
		a = dic[la]["end"].split('.')
		if len(a) == 1:
			dic[la]["end"] = int(a[0]) * 60
		elif len(a) == 2:
			dic[la]["end"] = int(a[0]) * 60 + int(a[1])


		mask = data.iloc[:,0] == "# PANAS"
		dic[la]["panas"] = data[mask].iloc[index,1:].astype(float).values.tolist()

		mask = data.iloc[:,0] == "# STAI"
		dic[la]["stai"] = data[mask].iloc[index,1:7].astype(int).values.tolist()

		mask = data.iloc[:,0] == "# DIM"
		dic[la]["va"] = data[mask].iloc[index,1:3].astype(int).values.tolist()
		
		if la == "Base":
			idx_label.append(1)
		elif la == "TSST":
			idx_label.append(2)
		elif la == "Fun":
			idx_label.append(3)
		elif la == "Medi 1":
			idx_label.append(4.1)
		elif la == "Medi 2":
			idx_label.append(4.2)
		else:
			raise ValueError('label not recognised')		


	for la, idx in zip(label, idx_label):
		dic[idx] = dic[la]
		del dic[la]
	
	return dic


def score_stai(list_answer, normalise=True):
	score_list = [1, 2, 3, 4]
	reverse_score = [4, 3, 2, 1]
	
	# reverse the weight
	for i in [1, 2, 4]:
		idx = score_list.index(list_answer[i])
		list_answer[i] = reverse_score[idx]

	score = sum(list_answer)
	
	if normalise:
		score = score/24
	
	return score

def preprocess_questionnaire(questionnaire):
	score_arousal_list = []
	score_valence_list = []
	time_questionnaires = []
	time_phase = []
	phase_list = []
	for key in questionnaire.keys():
		score_valence_list.append(questionnaire[key]["va"][0])
		score_arousal_list.append(questionnaire[key]["va"][1])
		time_questionnaires.append(questionnaire[key]["end"])
		time_phase.append(questionnaire[key]["start"])
		phase_list.append(questionnaire[key]["phase"])

	score_arousal_list = np.asarray(score_arousal_list)
	score_valence_list = np.asarray(score_valence_list)
	return score_arousal_list, score_valence_list, time_questionnaires, time_phase, phase_list