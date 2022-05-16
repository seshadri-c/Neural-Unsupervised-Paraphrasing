import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import pickle
from tqdm import tqdm

def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        out_string = str(out_string)
    occurance = [' .', ' ?', ' !', ' ,', " ' ", " n't", " 'm", " 's", " 've", " 're"]
    replacement = ['.', '?', '!', ',', "'", "n't", "'m", "'s", "'ve", "'re"]
    for i in range(len(occurance)):
    	out_string = out_string.replace(occurance[i], replacement[i])
    return out_string

def corruptSentence(sentence):
	tokens = sentence.split()
	length = len(tokens)
	
	corrputed_sentences = []
	mask = '<mask>'
	
	# 1. Token masking: Mask the sentence with randomly choosen words
	positions = np.random.choice(length, int(np.ceil(length/5)))
	pos_len = len(positions)
	sent = ""
	j = 0
	for i in range(length):
		if i in positions:
			sent+=mask
		else:
			sent+=tokens[i]
		sent+=' '
	corrputed_sentences.append(sent.strip())

	# 2. Token deletion: Select tokens randomly and delete them
	positions = np.random.choice(length, int(np.ceil(length/5)))
	pos_len = len(positions)
	sent = ""
	j = 0
	for i in range(length):
		if i in positions:
			continue
		else:
			sent+=tokens[i]
		sent+=' '
	corrputed_sentences.append(sent.strip())

	# 3. Text infilling: fill the randomly sampled span lengths with mask tokens. (poisson, lam=2)
	positions = np.random.choice(length, int(np.ceil(length/5)))
	pos_len = len(positions)
	span_lengths = np.random.poisson(2, int(np.ceil(length/5)))
	sent = ""
	j = 0
	for i in range(length):
		if i in positions:
			if span_lengths[j]==0:
				sent+=tokens[i]+' '
			i+=span_lengths[j]
			sent+='<mask>'
			j+=1
		else:
			sent+=tokens[i]
		sent+=' '
	corrputed_sentences.append(sent.strip())

	# 4: Document rotation:  Rotate the whole document around the selected word
	index = np.random.choice(length, 1)
	sent = ""
	for i in range(index[0], length):
		sent+=tokens[i]+' '
	for i in range(0,index[0]):
		sent+=tokens[i]+' '
	corrputed_sentences.append(sent.strip())

	return corrputed_sentences

def prepareCorrputSentences(filename, df_column, source_path, corrputed_path): 
	dataframe = pd.read_csv(filename, sep='\t', on_bad_lines='skip')
	# The df_column, or the dataframe column tells us about the column number of our required sentences
	# in the dataframe
	strings = dataframe.iloc[:, df_column].values
	sentences = [str(x).strip() for x in strings]
	sentences = [x.replace('\n', '') for x in sentences]

	print("No of sentences before filtering: ",len(sentences))

	sentences = [x for x in sentences if len(x.split())>5]

	print("No of sentences after filetering:",len(sentences))
	print('Creating source and corrupted_sentences ...')

	source_sentences = []
	corrupted_sentences = []

	for i in tqdm(range(len(sentences))):
		sent = sentences[i]
		corr_sent = corruptSentence(sent)
		source_sentences += [sent]*4
		corrupted_sentences += corr_sent


	with open(source_path, 'wb') as sfile, open(corrputed_path, 'wb') as cfile:
		pickle.dump(source_sentences, sfile)
		pickle.dump(corrupted_sentences, cfile)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', help="path of the dataset", required=True)
	parser.add_argument('-s', help='path where source sentences to be saved', required=True)
	parser.add_argument('-c', help='path where corrupted sentences to be saved', required=True)
	parser.add_argument('--col', help='Column number of the sentences in given data', type=int, required=True)

	arguments = parser.parse_args()

	prepareCorrputSentences(arguments.d, arguments.col, arguments.s, arguments.c)

