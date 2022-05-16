from torch.autograd import Variable
import torch 
import re
from make_transformer_model import *
from optimizer import *
from torchtext.data.metrics import bleu_score
import pickle
from data_loader_msr import *
from tqdm import tqdm
import sys

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

with open('word2int.pickle', 'rb') as handle:
    word2int = pickle.load(handle)

with open('int2word.pickle', 'rb') as handle:
    int2word = pickle.load(handle)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.module.encode(src, src_mask)
    
    temp = torch.tensor([0], dtype=torch.long, device=device)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(temp.data)
    for i in range(max_len-1):
        out = model.module.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(temp.data)))
        prob = model.module.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(temp.data).fill_(next_word)], dim=1)
    return ys
    
def decode_target(cap_tensor):
	
	tgt = ""
	for t in np.array(cap_tensor.data.squeeze(0)):
		sym = int2word[t]
		if sym == "</s>": 
			break
		if sym == "<s>":
			continue
		tgt += sym + " "
	return tgt

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']
    
    
def test_epoch(epoch, test_loader, model):
	
	progress_bar = tqdm(enumerate(test_loader))
	model.eval()
	
	target_list = []
	predicted_list = []
	j=0
	os.makedirs("outputs_msr",exist_ok=True)
	original_stdout = sys.stdout
	num_samples = 500
	with open("outputs_msr/test_texts_"+str(epoch)+".txt", "a") as f:
		sys.stdout = f
		for step, (src_tensor, tgt_tensor, src_mask, tgt_mask) in progress_bar:
			out = greedy_decode(model, src_tensor.to(device), src_mask.cuda(), max_len=60, start_symbol=word2int["<SOS>"])
			
			trans = ""
			for i in range(1, out.size(1)):
				sym = int2word[int(out[0, i])]
				if sym == "<EOS>": 
					break
				trans += sym + " "
				
			target_list.append([decode_target(tgt_tensor).upper().split()[1:-1]])
			predicted_list.append(trans.upper().split())
			print("\n\n Pair : {}/{}\n Target : {} \n Predicted : {}".format(j+1,len(test_loader), " ".join(target_list[-1][0]), " ".join(predicted_list[-1])))
			print("The BLEU Score : ",bleu_score(predicted_list, target_list)*100,"\n\n")
			j+=1
			if(j==num_samples):
				break
		sys.stdout = original_stdout



def main():
    
	src_path_2 = "data/prepared_data/src_sent/msr_src_test.pkl"

	gt_path_2 = "data/prepared_data/prepared_gt_sent/msr_gt_test.pkl"

	data_2 = (src_path_2, gt_path_2)

	checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/temp_NLP_MSR/"
	checkpoint_path = checkpoint_dir + "checkpoint_latest.pt"


	#Model made only with Train Vocab data  
	model = make_model(len(word2int.keys()),len(word2int.keys()), N=6).to(device)
	model_opt = get_std_opt(model)
	model = nn.DataParallel(model)


	# Loading the Checkpoint
	model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
	print("Loaded Successfully")

	test_loader = load_data(data_2, batch_size=1, num_workers=10, shuffle=False)

	test_epoch(epoch, test_loader, model)   
main()