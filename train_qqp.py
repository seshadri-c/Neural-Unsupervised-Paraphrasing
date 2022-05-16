from data_loader_qqp import *
from tqdm import tqdm
import numpy as np
import spacy
from make_transformer_model import *
from optimizer import *
from label_smoothing import *
from torchtext import data
import random
import sys
import pickle
from loss_backprop import *
from torchtext.data.metrics import bleu_score

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

with open('all_word2int.pickle', 'rb') as handle:
    all_word2int = pickle.load(handle)

with open('all_int2word.pickle', 'rb') as handle:
    all_int2word = pickle.load(handle)


#Function to Save Checkpoint
def save_ckp(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']
    
            
def train_epoch(epoch, train_loader, model, criterion, model_opt):
    
    model.train()
    progress_bar = tqdm(enumerate(train_loader))
    total_loss = 0.0
    
    for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
        out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
        ntokens = np.array(tgt[:,:-1]).shape[1]
        loss = loss_backprop(model.module.generator, criterion, out, tgt[:, 1:].cuda(), ntokens, all_int2word)
        total_loss +=loss
        model_opt.step()
        model_opt.optimizer.zero_grad()
        progress_bar.set_description("Epoch : {} Training Loss : {} Iteration : {}/{}".format(epoch+1, total_loss / (step + 1), step+1, len(train_loader))) 
        progress_bar.refresh()
        
    return total_loss/(step+1), model, model_opt            

def valid_epoch(epoch, valid_loader, model, criterion):
    
    model.eval()
    progress_bar = tqdm(enumerate(valid_loader))
    total_loss = 0.0
    total_tokens = 0
    for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
        out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
        ntokens = np.array(tgt[:,:-1]).shape[1]
        loss = loss_backprop(model.module.generator, criterion, out, tgt[:, 1:].cuda(), ntokens, all_int2word)
        total_loss +=loss
        progress_bar.set_description("Epoch : {} Validation Loss : {} Iteration : {}/{}".format(epoch+1, total_loss / (step + 1), step+1, len(valid_loader))) 
        progress_bar.refresh()  
        
    return total_loss/(step+1)
                    
def training_testing(train_loader, valid_loader, model, criterion, model_opt, resume):
    
    epoch = 0
    checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/temp_NLP_QQP/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir + "checkpoint_latest.pt"
    if resume:
        model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
        resume = False
        print("Resuming Training from Epoch Number : ", epoch)
    checkpoint_duplicate_dir = "/scratch/seshadri_c/temp_NLP_QQP/"
    os.makedirs(checkpoint_duplicate_dir, exist_ok=True)

    checkpoint_duplicate_path = checkpoint_duplicate_dir + "checkpoint_" + str(epoch + 1) + ".pt"
    while(1):

        checkpoint_path = checkpoint_dir + "checkpoint_latest.pt"
        checkpoint_duplicate_path = checkpoint_duplicate_dir + "checkpoint_" + str(epoch + 1) + ".pt"

        print("\n\nTraining : ")
        train_loss, model, model_opt = train_epoch(epoch, train_loader, model, criterion, model_opt)

        # Creating the Checkpoint
        checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': model_opt.optimizer.state_dict()}

        # Saving the Checkpoint
        save_ckp(checkpoint, checkpoint_path)
        print("Saved Successfully")
        save_ckp(checkpoint, checkpoint_duplicate_path)
        print("Duplicate Checkpoint Saved Successfully")

        # Loading the Checkpoint
        # model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
        # print("Loaded Successfully")
        
        print("Testing : ")
        valid_loss = valid_epoch(epoch, valid_loader, model, criterion)

        print("Epoch No {} completed.".format(epoch + 1))
        print("Train Loss : {} \t Valid Loss : {}".format(train_loss, valid_loss))
        if(epoch==0):
            with open('train_qqp_valid_loss.txt', 'w') as f:
                f.write("Epoch : {} \t Train Loss : {} \t Valid Loss : {}\n".format(epoch, train_loss, valid_loss))
        else:
            with open('train_qqp_valid_loss.txt', 'a') as f:
                f.write("Epoch : {} \t Train Loss : {} \t Valid Loss : {}\n".format(epoch, train_loss, valid_loss))
        epoch += 1


def main():
    
    src_path_1 = "data/prepared_data/src_sent/qqp_src.pkl"
    gt_path_1 = "data/prepared_data/prepared_gt_sent/qqp_gt.pkl"

    data_1 = (src_path_1, gt_path_1)
    
    #Model made only with Train Vocab data  
    model = make_model(len(all_word2int.keys()),len(all_word2int.keys()), N=6).to(device)
    model_opt = get_std_opt(model)
    model = nn.DataParallel(model)
     
    #Input is the Target Vocab Size
    criterion = LabelSmoothing(size=len(all_word2int.keys()), padding_idx=2, smoothing=0.1)
    criterion.cuda()
    
    train_loader = load_data(data_1, batch_size=64, num_workers=10, shuffle=True)
    valid_loader = load_data(data_1, batch_size=64, num_workers=10, shuffle=False)
    resume = False
    
    training_testing(train_loader, valid_loader, model, criterion, model_opt, resume)   
main()



