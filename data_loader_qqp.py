from torch.utils.data import Dataset, DataLoader
import os
import random
# from make_vocab import *
import torch 
import numpy as np
import pickle
# from training_setup import *
from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable
from transformer.mask import *


with open('all_word2int.pickle', 'rb') as handle:
	all_word2int = pickle.load(handle)

def make_std_mask(src, tgt, pad):

    src_mask = (src != pad).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask

class DataGenerator(Dataset):
	
	def __init__(self, data):
		self.files = self.get_files(data)
        
	def __len__(self):
		return len(self.files)
        

	def __getitem__(self,idx):

		src, tgt = self.files[idx]
		return src, tgt
			
	def get_files(self,data):

		print("Data Recieved.")		

		source_path,  gt_path = data

		src_list = []
		tgt_list = []

		with open(source_path, 'rb') as handle:
			lines_1 = pickle.load(handle)

		with open(gt_path, 'rb') as handle:
			lines_2 = pickle.load(handle)


		src_list.extend(lines_1)
		tgt_list.extend(lines_2)

		data = []
		for i in range(len(src_list)):
			data.append((src_list[i].upper(), tgt_list[i].upper()))
		
		return data

#Tokenizing the batch of sentences and adding BOS_WORD and EOS_WORD 
def tokenize_and_add_BOS_EOS(sentence):

	#Beginning of Sentence
	BOS_WORD = '<SOS>'
	#End of Sentence
	EOS_WORD = '<EOS>'

	token_list = []
	token_list.append(BOS_WORD)


	tokenizer = RegexpTokenizer(r'\w+')
	words = (tokenizer.tokenize(sentence.upper()))
	words = [w for w in words if w.isalnum()]

	token_list.extend(words)
	token_list.append(EOS_WORD)
	
	return token_list
	
def padding(sent_batch,max_len):
	
	padded_sent_batch = []
	PAD_WORD = '<PAD>'
	for s in sent_batch:
		[s.append(PAD_WORD) for i in range(len(s),max_len)]
		padded_sent_batch.append(s)
		
	return padded_sent_batch
	
def word_to_int(sent_batch):
	
	
	int_sent_batch = []
		
	for s in sent_batch:
		temp = []
		for t in s:
			try:
				temp.append(all_word2int[t])
			except:
				temp.append(2)
				
		int_sent_batch.append(temp)
		
	return int_sent_batch
	
def collate_fn_customised(data):
	
	src_sent = []
	tgt_sent = []
	
	#Step 1 : Tokenization
	for d in data:
		src, tgt = d
		src_sent.append(tokenize_and_add_BOS_EOS(src))
		tgt_sent.append(tokenize_and_add_BOS_EOS(tgt))
	
	#Step 2 : Getting Maximum Length for a Batch
	max_src = max([len(s) for s in src_sent])
	max_tgt = max([len(s) for s in tgt_sent])
	
	#Step 3 : Padding the Sequences
	padded_src_sent = padding(src_sent, max_src)
	padded_tgt_sent = padding(tgt_sent, max_tgt)
	
	#Step 4 : Converting the Padded Batch to Integers
	int_src_sent = word_to_int(padded_src_sent)
	int_tgt_sent = word_to_int(padded_tgt_sent)
	
	#Step 5 : Get the Masks
	src = torch.tensor(np.array(int_src_sent))
	tgt = torch.tensor(np.array(int_tgt_sent))
	pad = 2
	src_mask, tgt_mask = make_std_mask(src, tgt, pad)
	

	return src, tgt, src_mask, tgt_mask
	
def load_data(data, batch_size=128, num_workers=2, shuffle=True):
    
	dataset = DataGenerator(data)
	data_loader = DataLoader(dataset, collate_fn = collate_fn_customised, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return data_loader

#data_path = "./data/multi30k/uncompressed_data"
#load_data(data_path)
