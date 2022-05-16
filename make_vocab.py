import os
import pickle
from nltk.tokenize import RegexpTokenizer


src_path_1 = "data/prepared_data/src_sent/source0_sent.pkl"
src_path_2 = "data/prepared_data/src_sent/source1_sent.pkl"

gt_path_1 = "data/prepared_data/prepared_gt_sent/prepared_gt_sent0.pkl"
gt_path_2 = "data/prepared_data/prepared_gt_sent/prepared_gt_sent1.pkl"


def create_vocab(source_path_1, source_path_2, gt_path_1, gt_path_2) :

	lines = []

	with open(src_path_1, 'rb') as handle:
		lines_1 = pickle.load(handle)

	with open(src_path_2, 'rb') as handle:
		lines_2 = pickle.load(handle)

	with open(gt_path_1, 'rb') as handle:
		lines_3 = pickle.load(handle)

	with open(gt_path_2, 'rb') as handle:
		lines_4 = pickle.load(handle)

	lines.extend(lines_1)
	lines.extend(lines_2)
	lines.extend(lines_3)	
	lines.extend(lines_4)

	print(len(lines))
	words = []
	tokenizer = RegexpTokenizer(r'\w+')

	for l in lines:
		words.extend(tokenizer.tokenize(l.upper()))

	words = [w for w in words if w.isalnum()]

	words = set(words)
	list(words).sort()

	word2int = {}
	int2word = {}

	word2int['<SOS>'] = 0
	word2int['<EOS>'] = 1
	word2int['<PAD>'] = 2

	int2word[0] = "<SOS>"
	int2word[1] = "<EOS>"
	int2word[2] = "<PAD>"
	

	for id, word in enumerate(words):
		word2int[word] = id + 3
		int2word[id + 3] = word


	with open('word2int.pickle', 'wb') as handle:
	    pickle.dump(word2int, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('int2word.pickle', 'wb') as handle:
	    pickle.dump(int2word, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return word2int
	
train_vocab = create_vocab(src_path_1, src_path_2, gt_path_1, gt_path_2)
print(train_vocab)