from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# from misc.utils import repackage_hidden_new, clip_gradient, adjust_learning_rate, \
#                     decode_txt, sample_batch_neg, l2_norm
# import misc.dataLoader as dl
# import misc.model as model
# from misc.encoder_QIH import _netE
# from misc.netG import _netG
# import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='../script/old_data/vdl_img_vgg.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_ques_h5', default='../script/old_data/visdial_data.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='../script/old_data/visdial_params.json', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output images and model checkpoints')
parser.add_argument('--encoder', default='QIH_G', help='what encoder to use.')
parser.add_argument('--num_val', default=1000, help='number of image split out as validation set.')
parser.add_argument('--update_D', action='store_true', help='whether train use the GAN loss.')
parser.add_argument('--update_LM', action='store_true', help='whether train use the GAN loss.')

#parser.add_argument('--model_path', default='save/QIH_perceptual.1-5-21/epoch_8.pth', help='folder to output images and model checkpoints')
#parser.add_argument('--model_path', default='save/GAN_0.4.9-5-12/epoch_5.pth', help='folder to output images and model checkpoints')
#parser.add_argument('--model_path', default='save/G/epoch_30.pth', help='folder to output images and model checkpoints')
parser.add_argument('--model_path', default='../script/save/HCIAE-G-MLE.pth', help='folder to output images and model checkpoints')

parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')

parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=5)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--eval_iter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--save_iter', type=int, default=2, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

parser.add_argument('--D_lr', type=float, default=1e-4, help='learning rate for, default=0.00005')
parser.add_argument('--G_lr', type=float, default=1e-4, help='learning rate for, default=0.00005')
parser.add_argument('--LM_lr', type=float, default=4e-5, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.8')


parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--gumble_weight', type=int, default=0.3, help='folder to output images and model checkpoints')
parser.add_argument('--path_to_home',type=str)

opt = parser.parse_args()
sys.path.insert(1, opt.path_to_home)
print(opt)

from misc.utils import repackage_hidden_new, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
from misc.netG import _netG
import datetime

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.model_path != '':
    checkpoint = torch.load(opt.model_path)


batch_size = 30
####################################################################################
# Data Loader
####################################################################################

dataset_val = dl.validate(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'test')

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

####################################################################################
# Build the Model
####################################################################################
vocab_size = dataset_val.vocab_size
ques_length = dataset_val.ques_length
ans_length = dataset_val.ans_length + 1
his_length = dataset_val.ans_length + dataset_val.ques_length
itow = dataset_val.itow
img_feat_size = 512

print('init Generative model...')
netG = _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, False)
netE_g = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW_g = model._netW(vocab_size, opt.ninp, opt.dropout)
sampler = model.gumbel_sampler()
critG = model.G_loss(opt.ninp)
critLM = model.LMCriterion()

if True:# opt.model_path_D != '' and opt.model_path_G != '':
    print('Loading Generative model...')
    netW_g.load_state_dict(checkpoint['netW'])
    netE_g.load_state_dict(checkpoint['netE'])
    netG.load_state_dict(checkpoint['netG'])

if opt.cuda: # ship to cuda, if has GPU
    netW_g.cpu()
    netE_g.cpu()
    netG.cpu()
    critG.cpu()
    sampler.cpu(), critLM.cpu()

####################################################################################
# training model
####################################################################################

def val():
    netE_g.eval()
    netW_g.eval()
    netG.eval()

    n_neg = 100
    ques_hidden1 = netE_g.init_hidden(opt.batchSize)

    hist_hidden1 = netE_g.init_hidden(opt.batchSize)

    bar = progressbar.ProgressBar(max_value=len(dataloader_val))
    data_iter_val = iter(dataloader_val)

    count = 0
    i = 0

    result_all = []
    # print('length of dataloader: ', len(dataloader_val))
    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                    opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        batch_size = question.size(0)
        image = image.view(-1, 512)
        with torch.no_grad():
            img_input.resize_(image.size()).copy_(image)

        save_tmp = [[] for j in range(batch_size)]
        for rnd in range(10):

            # get the corresponding round QA and history.
            ques, tans = question[:,rnd,:].t(), answerT[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answer[:,rnd,:,:].clone().view(-1, ans_length).t()
            opt_tans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]
            opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_input = torch.LongTensor(ques.size()).cpu()
            ques_input.copy_(ques)

            his_input = torch.LongTensor(his.size()).cpu()
            his_input.copy_(his)

            opt_ans_input = torch.LongTensor(opt_ans.size()).cpu()
            opt_ans_input.copy_(opt_ans)

            opt_ans_target = torch.LongTensor(opt_tans.size()).cpu()
            opt_ans_target.copy_(opt_tans)

            gt_index = torch.LongTensor(gt_id.size())
            gt_index.copy_(gt_id)

            ques_emb_g = netW_g(ques_input, format = 'index')
            his_emb_g = netW_g(his_input, format = 'index')

            ques_hidden1 = repackage_hidden_new(ques_hidden1, batch_size)

            hist_hidden1 = repackage_hidden_new(hist_hidden1, his_emb_g.size(1))

            featG, ques_hidden1 = netE_g(ques_emb_g, his_emb_g, img_input, \
                                                ques_hidden1, hist_hidden1, rnd+1)

            #featD = l2_norm(featD)
            # Evaluate the Generator:
            _, ques_hidden1 = netG(featG.view(1,-1,opt.ninp), ques_hidden1)
            #_, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)
            # extend the hidden
            sample_ans_input = torch.LongTensor(1, opt.batchSize).cpu()
            sample_ans_input.resize_((1, batch_size)).fill_(vocab_size)
            
            sample_opt = {'beam_size':1}

            seq, seqLogprobs = netG.sample(netW_g, sample_ans_input, ques_hidden1, sample_opt)
            ans_sample_txt = decode_txt(itow, seq.t())
            ans_txt = decode_txt(itow, tans)
            ques_txt = decode_txt(itow, questionL[:,rnd,:].t())
            
            '''
            for j in range(len(ans_txt)):
                print('Q: %s --A: %s --Sampled: %s' %(ques_txt[j], ans_txt[j], ans_sample_txt[j]))
            
            ans_sample_z = [[] for z in range(batch_size)]
            for m in range(5):
                ans_sample_result = torch.Tensor(ans_length, batch_size)
                # sample the result.
                noise_input.data.resize_(ans_length, batch_size, vocab_size+1)
                noise_input.data.uniform_(0,1)
                for t in range(ans_length):
                    ans_emb = netW_g(sample_ans_input, format = 'index')
                    if t == 0:
                        logprob, ques_hidden2 = netG(ans_emb.view(1,-1,opt.ninp), ques_hidden1)
                    else:
                        logprob, ques_hidden2 = netG(ans_emb.view(1,-1,opt.ninp), ques_hidden2)

                    one_hot, idx = sampler(logprob, noise_input[t], opt.gumble_weight)

                    sample_ans_input.data.copy_(idx.data)
                    ans_sample_result[t].copy_(idx.data)

                ans_sample_txt = decode_txt(itow, ans_sample_result)
                for ii in range(batch_size):
                    ans_sample_z[ii].append(ans_sample_txt[ii])
            '''
            ans_txt = decode_txt(itow, tans)
            ques_txt = decode_txt(itow, questionL[:,rnd,:].t())
            #for j in range(len(ans_txt)):
            #    print('Q: %s --A: %s --Sampled: %s' %(ques_txt[j], ans_txt[j], ans_sample_txt[j]))

            for j in range(batch_size):
                save_tmp[j].append({'ques':ques_txt[j], 'gt_ans':ans_txt[j], \
                            'sample_ans':ans_sample_txt[j], 'rnd':rnd, 'img_id':img_id[j].item()})
        i += 1
        bar.update(i)

        result_all += save_tmp

    return result_all

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
ans_input = torch.LongTensor(ans_length, opt.batchSize)
ans_target = torch.LongTensor(ans_length, opt.batchSize)
wrong_ans_input = torch.LongTensor(ans_length, opt.batchSize)
sample_ans_input = torch.LongTensor(1, opt.batchSize)

fake_len = torch.LongTensor(opt.batchSize)
fake_diff_mask = torch.ByteTensor(opt.batchSize)
fake_mask = torch.ByteTensor(opt.batchSize)
# answer len
batch_sample_idx = torch.LongTensor(opt.batchSize)

# noise
noise_input = torch.FloatTensor(opt.batchSize)

# for evaluation:
opt_ans_input = torch.LongTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)
opt_ans_target = torch.LongTensor(opt.batchSize)

if opt.cuda:
    ques_input, his_input, img_input = ques_input.cpu(), his_input.cpu(), img_input.cpu()
    ans_input, ans_target = ans_input.cpu(), ans_target.cpu()
    wrong_ans_input = wrong_ans_input.cpu()
    sample_ans_input = sample_ans_input.cpu()

    fake_len = fake_len.cpu()
    noise_input = noise_input.cpu()
    batch_sample_idx = batch_sample_idx.cpu()
    fake_diff_mask = fake_diff_mask.cpu()
    fake_mask = fake_mask.cpu()

    opt_ans_input = opt_ans_input.cpu()
    gt_index = gt_index.cpu()
    opt_ans_target = opt_ans_target.cpu()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
wrong_ans_input = Variable(wrong_ans_input)
sample_ans_input = Variable(sample_ans_input)

noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
fake_diff_mask = Variable(fake_diff_mask)
fake_mask = Variable(fake_mask)

opt_ans_input = Variable(opt_ans_input)
opt_ans_target = Variable(opt_ans_target)
gt_index = Variable(gt_index)


epoch = 0
print('Evaluating ... ')
result_all = val()

json.dump(result_all, open('Per_greedy.json', 'w'))
