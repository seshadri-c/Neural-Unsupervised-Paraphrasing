import os
import pickle
from nltk.tokenize import RegexpTokenizer


src_path_1 = "data/prepared_data/src_sent/msr_src_train.pkl"
src_path_2 = "data/prepared_data/src_sent/paws_src_train.pkl"
src_path_3 = "data/prepared_data/src_sent/qqp_src.pkl"

src_path_4 = "data/prepared_data/src_sent/msr_src_test.pkl"
src_path_5 = "data/prepared_data/src_sent/paws_src_test.pkl"

tgt_path_1 = "data/prepared_data/prepared_gt_sent/msr_gt_train.pkl"
tgt_path_2 = "data/prepared_data/prepared_gt_sent/paws_gt_train.pkl"
tgt_path_3 = "data/prepared_data/prepared_gt_sent/qqp_gt.pkl"

tgt_path_4 = "data/prepared_data/prepared_gt_sent/msr_gt_test.pkl"
tgt_path_5 = "data/prepared_data/prepared_gt_sent/paws_gt_test.pkl"


def create_vocab() :

	lines = []

	with open(src_path_1, 'rb') as handle:
		lines_1 = pickle.load(handle)

	with open(src_path_2, 'rb') as handle:
		lines_2 = pickle.load(handle)

	with open(src_path_3, 'rb') as handle:
		lines_3 = pickle.load(handle)

	with open(src_path_4, 'rb') as handle:
		lines_4 = pickle.load(handle)

	with open(src_path_5, 'rb') as handle:
		lines_5 = pickle.load(handle)

	with open(tgt_path_1, 'rb') as handle:
		lines_6 = pickle.load(handle)

	with open(tgt_path_2, 'rb') as handle:
		lines_7 = pickle.load(handle)

	with open(tgt_path_3, 'rb') as handle:
		lines_8 = pickle.load(handle)

	with open(tgt_path_4, 'rb') as handle:
		lines_9 = pickle.load(handle)

	with open(tgt_path_5, 'rb') as handle:
		lines_10 = pickle.load(handle)



	lines.extend(lines_1)
	lines.extend(lines_2)
	lines.extend(lines_3)	
	lines.extend(lines_4)
	lines.extend(lines_5)
	lines.extend(lines_6)
	lines.extend(lines_7)	
	lines.extend(lines_8)
	lines.extend(lines_9)
	lines.extend(lines_10)

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


	with open('all_word2int.pickle', 'wb') as handle:
	    pickle.dump(word2int, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('all_int2word.pickle', 'wb') as handle:
	    pickle.dump(int2word, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return word2int
	
train_vocab = create_vocab()
print(train_vocab)