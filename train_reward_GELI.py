import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
import os
import pandas as pd
import numpy as np
from transformers import AutoModel
import random 
import utils
import wandb
from datetime import datetime
from data_loader_reward import CANDOR_SentenceBert_K, CANDOR_LLAMA_K, CANDOR_LLAMA_K_SmallVal
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer, LongformerModel, AutoModelForSeq2SeqLM, BartForSequenceClassification
from transformers import LongT5ForConditionalGeneration, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from options_reward import parser
from utils import SupConLoss
from torch.nn.modules.loss import _Loss
from widis_lstm_tools.nn import LSTMLayer
import math

args = parser.parse_args()


reward_class = args.reward_class

print("REWARD CHOSEN: {}".format(reward_class)) 


        # hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.dense(hidden_states)
        # hidden_states = torch.tanh(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # output = self.out_proj(hidden_states)

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def contrastive_loss(temp, embedding, label):
    """calculate the contrastive loss
    """
    # cosine similarity between embeddings
    cosine_sim = sim_matrix(embedding, embedding)
    n = cosine_sim.shape[0]
    dis = cosine_sim.masked_select(~torch.eye(n, dtype=bool).cuda()).view(n, n - 1)

    # apply temperature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = torch.exp(dis)
    cosine_sim = torch.exp(cosine_sim)

    # calculate row sum
    row_sum = torch.sum(dis, -1)

    unique_labels, inverse_indices, unique_label_counts = torch.unique(label, sorted=False, return_inverse=True, return_counts=True)
    # calculate outer sum
    contrastive_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    for i in range(n):
        n_i = unique_label_counts[inverse_indices[i]] - 1
        inner_sum = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        # calculate inner sum
        for j in range(n):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
    return contrastive_loss


contrastive_criterion = SupConLoss(temperature=0.3)



def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).float().cuda())     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)



def FiLM(x, gammas, betas):
    return (gammas * x) + betas



def RUDDER_lossfunction(predictions, rewards):


    returns = rewards.sum(dim=-1)
    # Main task: predicting return at last timestep
    main_loss = torch.mean(predictions[-1] - returns) ** 2

    # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
    aux_loss = torch.mean(predictions[..., None] - returns[..., None]) ** 2
    # Combine losses
    loss = main_loss + aux_loss * 0.5
    return loss, main_loss





if __name__ == '__main__':

    wandb.login()

    if args.redist_type == "RUDDER":
        class reward_function(torch.nn.Module):
            def __init__(self, input_size = 1024, hidden_size = 16):
                super(reward_function, self).__init__()

                # This will create an LSTM layer where we will feed the concatenate
                self.lstm = LSTMLayer(
                    in_features=input_size, out_features=hidden_size, inputformat='NLC',
                    # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
                    w_ci=(torch.nn.init.xavier_normal_, False),
                    # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
                    w_ig=(False, torch.nn.init.xavier_normal_),
                    # output gate: disable all connection (=no forget gate) and disable bias
                    w_og=False, b_og=False,
                    # forget gate: disable all connection (=no forget gate) and disable bias
                    w_fg=False, b_fg=False,
                    # LSTM output activation is set to identity function
                    a_out=lambda x: x
                )
                
                # After the LSTM layer, we add a fully connected output layer
                self.fc_out = torch.nn.Linear(hidden_size, 1)


            def forward(self, dialogue):
                # Process input sequence by LSTM
                lstm_out, *_ = self.lstm(dialogue.unsqueeze(0), return_all_seq_pos=True)
                net_out = self.fc_out(lstm_out)
                return net_out
    
        
    else: 
        class reward_function(nn.Module):
            def __init__(self):
                super(reward_function, self).__init__()
                self.fc1 = nn.Linear(1024,128)
                self.Relu = nn.ReLU()
                self.fc2 = nn.Linear(128,1)
                self.Tanh = nn.Tanh()
                self.dropout = nn.Dropout(0.2)

            def forward(self,x):
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.Relu(x)
                x = self.fc2(x)
                x = self.dropout(x)
                x = self.Tanh(x)
                return x 

    
    if args.affect_path:

        saved_path = "./SINGLE_INDEX_reward_function_convo_overall_affect_contra_False_shrink_False_curriculum_False_curriculum_exposureFalse"

        PATH = os.path.join(saved_path, "reward1_" +args.affect_path)

        if args.redist_type != "RUDDER":
            reward_function1 = torch.load(PATH)
        else: 
            reward_function1 = reward_function().cuda()
    else:
        reward_function1 = reward_function().cuda()


    #pretrained sentiment classifier 
    if not args.train and args.val:

        saved_path = "./SINGLE_INDEX_reward_function_convo_overall_affect_contra_False_shrink_False_curriculum_False_curriculum_exposureFalse"
        longformer_tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")        
        longformer_tokenizer.truncation_side='left'
        PATH = os.path.join(saved_path, "lang_model_" + "devout-brook-114")
        lang_model = BartForSequenceClassification.from_pretrained(PATH, num_labels = 1).cuda()
        
        class reward_function(nn.Module):
            def __init__(self):
                super(reward_function, self).__init__()
                self.fc1 = nn.Linear(1024,128)
                self.Relu = nn.ReLU()
                self.fc2 = nn.Linear(128,1)
                self.Tanh = nn.Tanh()
                self.dropout = nn.Dropout(0.2)

            def forward(self,x):
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.Relu(x)
                x = self.fc2(x)
                x = self.dropout(x)
                x = self.Tanh(x)
                return x 
        



    film_gamma = nn.Sequential(nn.Linear(1024, 1)).cuda()
    film_beta = nn.Sequential(nn.Linear(1024, 1)).cuda()


    
    loss_fn = torch.nn.MSELoss()
    rate_learning = 5e-3

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf") 

    
    batch_size = 1 #int(args.batch_size) #64
    K = int(args.K)
    flip = False
    N_EPOCHS = 4
    train_size = int(args.train_size) 
    ep_lens = 160


    if args.model == 'convo':
        longformer_tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")            

        if args.affect_path: 
            PATH = os.path.join(saved_path, "lang_model_" +args.affect_path)
            lang_model = BartForSequenceClassification.from_pretrained(PATH, num_labels = 1).cuda()

        else:
            lang_model = BartForSequenceClassification.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary", num_labels = 1).cuda()
        lang_model = torch.nn.DataParallel(lang_model, device_ids=[0, 1, 2, 3])
        

    if args.affect_path:
        WANDB_PROJECT = 'visual_feedback_baselines_{}_{}_{}_{}'.format(args.model, reward_class, "baseline_" + str(args.redist_type), "K_" + str(args.K))
    else:
        WANDB_PROJECT = 'baselines_{}_{}_{}_{}'.format(args.model, reward_class, "baseline_" + str(args.redist_type), "K_" + str(args.K))
    K = K

    run = wandb.init(project=WANDB_PROJECT, 
           config={'lr':rate_learning, 'batch_size':batch_size, 'n_epochs':N_EPOCHS, 'K': K, 'train_size': train_size})
    # Start tracking your model's gradients
    now_time = "{}".format(str(run.name))

    folder_name = "./{}".format(WANDB_PROJECT)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    


    train_dataset_unflip = CANDOR_LLAMA_K_SmallVal(tokenizer = longformer_tokenizer, K = K, flip = False, utils = utils, split = 'train', train_size=train_size, reward_class = reward_class, curriculum= args.curriculum, reaction_type = args.reaction_type)
    train_dataset_flip = CANDOR_LLAMA_K_SmallVal(tokenizer = longformer_tokenizer, K = K, flip = True, utils = utils, split = 'train', train_size = train_size, reward_class = reward_class, curriculum= args.curriculum, reaction_type = args.reaction_type)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_unflip, train_dataset_flip])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_dataset = CANDOR_LLAMA_K_SmallVal(tokenizer = longformer_tokenizer, K = None, flip = flip, utils = utils, split = 'val', train_size=train_size, reward_class = reward_class, reaction_type = args.reaction_type)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1
    )


    model_params = list(reward_function1.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=rate_learning, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_loader) * N_EPOCHS),
    )

    optimizer.add_param_group({'params': film_gamma.parameters(), 'lr':rate_learning, 'name': 'film_gamma'})
    optimizer.add_param_group({'params': film_beta.parameters(), 'lr':rate_learning, 'name': 'film_beta'})


    best_val_loss = 1e10
    #TRAIN
    for epoch in range(N_EPOCHS):

        if args.train: 
            train_running_loss = 0
            train_running_div_loss = 0
            train_running_contrastive_loss = 0
            train_running_shaped_loss = 0

            train_running_neg_mean = []
            train_running_pos_mean = []
            train_running_neut_mean = []
            train_running_sup_mean = []


            for idx, batch in enumerate(tqdm(train_loader)):

                speaker_A_tokens = batch['A_sentence_llama']
                speaker_B_tokens = batch['B_sentence_llama']
                rewards = batch['reward']

                encoded_tokens = speaker_A_tokens

                #approximate average length 
                all_parts = []

                chunk_vals =K//16

                print(K)

                with torch.no_grad():
                    for part_ind in range(16):
                        try:
                            part = lang_model(input_ids = encoded_tokens["input_ids"].squeeze()[part_ind*chunk_vals:(part_ind+1)*chunk_vals,:], attention_mask = encoded_tokens["attention_mask"].squeeze()[part_ind*chunk_vals:(part_ind+1)*chunk_vals,:])["encoder_last_hidden_state"] 
                        except Exception:
                            pdb.set_trace()
                        # part = part["encoder_last_hidden_state"]
                        eos_mask = encoded_tokens["input_ids"].squeeze()[part_ind*chunk_vals:(part_ind+1)*chunk_vals,:].eq(longformer_tokenizer.eos_token_id).to(part.device)
                        sentence_representation = part[eos_mask, :].view(part.size(0), -1, part.size(-1))
                        sentence_representation = sentence_representation[:,-1,:]
                        all_parts.append(sentence_representation)
                    full_sentence_rep = torch.cat(all_parts)


                all_outputs = full_sentence_rep.reshape(-1, 1024)
                # #sampled_rewards = sampling_function(reward_function(all_outputs).squeeze()).shape

                pred_rewards = reward_function1(all_outputs)

                

                if args.redist_type == 'RUDDER':

                    pred_rewards = pred_rewards[..., 0]
                    pred_rewards = pred_rewards.squeeze().unsqueeze(1)

                shaped_rewards = torch.zeros_like(pred_rewards)

                sum_rewards = (pred_rewards).squeeze().sum(-1)

                
                sum_loss = loss_fn(sum_rewards.squeeze(), rewards.cuda().squeeze().float())
                loss = sum_loss 
                if args.redist_type == 'RRD':
                    sum_loss = loss_fn((ep_lens/K) * sum_rewards.squeeze(), rewards.cuda().squeeze().float())
                    loss = sum_loss 
                # loss = 0.5*shaped_loss + 0.5*sum_loss

                if args.redist_type == 'IRCR':
                    ircr_sum_loss = loss_fn(pred_rewards,(rewards.cuda().float()/K).unsqueeze(0).repeat(K, 1))
                    loss = ircr_sum_loss


                if args.redist_type == 'RUDDER':
                    pred_rewards = pred_rewards.squeeze().unsqueeze(0)

                    #set loss here 
                    loss, other_sum_loss = RUDDER_lossfunction(pred_rewards.squeeze(), rewards.cuda().float())

                    redistributed_reward = pred_rewards[:, 1:] - pred_rewards[:, :-1]
                    # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
                    redistributed_reward = torch.cat([pred_rewards[:, :1], redistributed_reward], dim=1)
                    pred_rewards = redistributed_reward.squeeze().unsqueeze(1)
                    sum_loss = loss_fn((160/K) * redistributed_reward.sum(-1).squeeze(), rewards.cuda().squeeze().float())



                neg_affect_indices = (batch['reaction_visual'] <= 4).squeeze().unsqueeze(1)
                pos_affect_indices = (batch['reaction_visual'] == 6).squeeze().unsqueeze(1)
                neutral_affect_indices = (batch['reaction_visual'] == 5).squeeze().unsqueeze(1)
                surprise_indices = (batch['reaction_visual'] == 7).squeeze().unsqueeze(1)

                shaped_rewards[neg_affect_indices]  = -1.0
                shaped_rewards[neutral_affect_indices] = 0.0
                shaped_rewards[pos_affect_indices] = 1.0
                shaped_rewards[surprise_indices] = 0.0
                
                # 10 * torch.sign(pred_rewards[surprise_indices].clone().detach()).float()


                neg_affect_reward_mean = pred_rewards[neg_affect_indices].mean()
                pos_affect_reward_mean = pred_rewards[pos_affect_indices].mean()
                neut_affect_reward_mean = pred_rewards[neutral_affect_indices].mean()
                sup_affect_reward_mean = torch.abs(pred_rewards[surprise_indices].mean())

                

                if not neg_affect_reward_mean.isnan():
                    train_running_neg_mean.append(neg_affect_reward_mean.item())

                if not pos_affect_reward_mean.isnan():
                    train_running_pos_mean.append(pos_affect_reward_mean.item())
                
                if not neut_affect_reward_mean.isnan():
                    train_running_neut_mean.append(neut_affect_reward_mean.item())

                if not sup_affect_reward_mean.isnan():
                    train_running_sup_mean.append(sup_affect_reward_mean.item())

                print('Neg. Affective Reward Mean (-1): {}\n'.format(neg_affect_reward_mean))
                print('Pos. Affective Reward Mean (1): {}\n'.format(pos_affect_reward_mean))
                print('Neutral Affective Reward Mean (0): {}\n'.format(neut_affect_reward_mean))
                print('Surprise Affective Reward Mean (1): {}\n'.format(sup_affect_reward_mean))
                
                if args.reaction_type == 'mean':
                    shaped_rewards = torch.stack(batch['reaction_visual_intensity']).cuda().float()


                shaped_loss = loss_fn(pred_rewards.squeeze(), shaped_rewards.squeeze())
                    # loss = 0.5*shaped_loss + 0.5*ircr_sum_loss


                contrastive_loss_full = torch.Tensor([0]).cuda()


                
                if args.bar_chart: 

                    #batch['indices']
                    data = [[label, val] for (label, val) in zip(batch['indices'].squeeze(), pred_rewards.detach().squeeze())]
                    table = wandb.Table(data=data, columns = ["label", "value"])
                    wandb.log({"rewards" : wandb.plot.bar(table, "label", "value",
                                                title="predicted rewards1_{}".format(batch['reward'].item()))})
                    
                if args.affect_path:
                    loss = 0.5*loss + 0.5*shaped_loss 

                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


                # contrastive_loss_full.backward()
                # optim2.step()
                # lr_scheduler.step()
                # optim2.zero_grad()


                

                #logging 


                wandb.log({f'train/shaped_loss':shaped_loss.item()})
                wandb.log({f'train/loss':sum_loss.item()})

                wandb.log({f'train_shape/train_neg_mean':neg_affect_reward_mean})
                wandb.log({f'train_shape/train_pos_mean':pos_affect_reward_mean})
                wandb.log({f'train_shape/train_neut_mean':neut_affect_reward_mean})
                wandb.log({f'train_shape/train_sup_mean':sup_affect_reward_mean})

                # wandb.log({f'train/div_loss':divergence_loss.item()})
                
                train_running_shaped_loss += shaped_loss.item()
                train_running_loss += sum_loss.item()
                # train_running_div_loss += divergence_loss.item()
                train_running_contrastive_loss += contrastive_loss_full.item()



        if args.val: 
            print("\n ********RUNNING VALIDATION STEP*********** \n")
            val_running_loss = 0
            val_shaped_loss = 0
            val_running_neg_mean = []
            val_running_pos_mean = []
            val_running_neut_mean = []
            val_running_sup_mean = []


            for val_idx, val_batch in enumerate(tqdm(val_loader)):

                speaker_A_tokens = val_batch['A_sentence_llama']
                speaker_B_tokens = val_batch['B_sentence_llama']
                rewards = val_batch['reward']

                encoded_tokens = speaker_A_tokens
                

                chunk_vals = math.ceil(encoded_tokens.input_ids.shape[1]/16)

                all_parts = []
                with torch.no_grad():
                    for part_ind in range(16):
                        curr_encoded = encoded_tokens["input_ids"].squeeze()[part_ind*chunk_vals:(part_ind+1)*chunk_vals,:]
                        if len(curr_encoded) != 0: #only if nonempty
                            part = lang_model(input_ids = curr_encoded, attention_mask = encoded_tokens["attention_mask"].squeeze()[part_ind*chunk_vals:(part_ind+1)*chunk_vals,:])["encoder_last_hidden_state"] 
                            eos_mask = encoded_tokens["input_ids"].squeeze()[part_ind*chunk_vals:(part_ind+1)*chunk_vals,:].eq(longformer_tokenizer.eos_token_id).to(part.device)
                            sentence_representation = part[eos_mask, :].view(part.size(0), -1, part.size(-1))
                            sentence_representation = sentence_representation[:,-1,:]
                            all_parts.append(sentence_representation)
                    full_sentence_rep = torch.cat(all_parts)

                                        


                    all_outputs = full_sentence_rep.reshape(-1, 1024)
                    # #sampled_rewards = sampling_function(reward_function(all_outputs).squeeze()).shape

                    # hiden_state = reward_function1(all_outputs)
                    # pred_rewards = reward_function2(hiden_state)

                    pred_rewards = reward_function1(all_outputs)

                        
                    sum_rewards = (pred_rewards).squeeze().sum(-1)




                    if args.redist_type == 'RUDDER':
                        pred_rewards = pred_rewards.squeeze().unsqueeze(0)

                        #set loss here 
                        loss, other_sum_loss = RUDDER_lossfunction(pred_rewards.squeeze(), rewards.cuda().float())

                        redistributed_reward = pred_rewards[:, 1:] - pred_rewards[:, :-1]
                        # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
                        redistributed_reward = torch.cat([pred_rewards[:, :1], redistributed_reward], dim=1)

                        pred_rewards = redistributed_reward.squeeze().unsqueeze(1)
                        sum_loss = loss_fn((160/K) * pred_rewards.sum().squeeze(), rewards.cuda().squeeze().float())



                    
                    shaped_rewards = torch.zeros_like(pred_rewards)

                    
                    neg_affect_indices = (val_batch['reaction_visual'] <= 4).squeeze().unsqueeze(1)
                    pos_affect_indices = (val_batch['reaction_visual'] == 6).squeeze().unsqueeze(1)
                    neutral_affect_indices = (val_batch['reaction_visual'] == 5).squeeze().unsqueeze(1)
                    surprise_indices = (val_batch['reaction_visual'] == 7).squeeze().unsqueeze(1)

                    shaped_rewards[neg_affect_indices]  = -1.0
                    shaped_rewards[neutral_affect_indices] = 0.0
                    shaped_rewards[pos_affect_indices] = 1.0
                    shaped_rewards[surprise_indices] = torch.sign(pred_rewards[surprise_indices].clone().detach()).float()


                    neg_affect_reward_mean = pred_rewards[neg_affect_indices].mean()
                    pos_affect_reward_mean = pred_rewards[pos_affect_indices].mean()
                    neut_affect_reward_mean = pred_rewards[neutral_affect_indices].mean()
                    sup_affect_reward_mean = torch.abs(pred_rewards[surprise_indices].mean())

                    if not neg_affect_reward_mean.isnan():
                        val_running_neg_mean.append(neg_affect_reward_mean.item())

                    if not pos_affect_reward_mean.isnan():
                        val_running_pos_mean.append(pos_affect_reward_mean.item())
                    
                    if not neut_affect_reward_mean.isnan():
                        val_running_neut_mean.append(neut_affect_reward_mean.item())

                    if not sup_affect_reward_mean.isnan():
                        val_running_sup_mean.append(sup_affect_reward_mean.item())

                    if args.redist_type != 'RUDDER':
                        
                        sum_loss = loss_fn(rewards.float().cuda(), (ep_lens/encoded_tokens.input_ids.shape[1])*sum_rewards.unsqueeze(0))
                    
                    
                    shaped_loss = loss_fn(shaped_rewards.squeeze(), pred_rewards.squeeze())

                    #logging 
                    
                    wandb.log({f"val/shaped_loss": shaped_loss.item()})
                    wandb.log({f"val/sum_loss": sum_loss.item()})
                    # wandb.log({f"val/middle_loss": middle_loss.item()})

                    # middle_val_running_loss += middle_loss.item()
                    val_shaped_loss += shaped_loss
                    val_running_loss += sum_loss.item()

            epoch_val_loss = val_running_loss/(len(val_loader)) #len(val_loader)
            # epoch_val_middle_loss= middle_val_running_loss/(len(val_loader))
            # wandb.log({f'val/epoch_val_middle_loss':epoch_val_middle_loss})
            wandb.log({f'val/epoch_loss': val_running_loss/(len(val_loader))})
            wandb.log({f'val/epoch_shaped_loss': val_shaped_loss/(len(val_loader))})

            wandb.log({f'val_shape_epoch/epoch_running_neg_mean':np.array(val_running_neg_mean).mean()})
            wandb.log({f'val_shape_epoch/epoch_running_pos_mean':np.array(val_running_pos_mean).mean()})
            wandb.log({f'val_shape_epoch/epoch_running_neut_mean':np.array(val_running_neut_mean).mean()})
            wandb.log({f'val_shape_epoch/epoch_running_sup_mean':np.array(val_running_sup_mean).mean()})
                    
            if not args.train:
                pdb.set_trace()
                quit() 

            #saving model --uncomment 
            if epoch_val_loss <best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(reward_function1,"./{}/reward1_{}".format(WANDB_PROJECT,now_time)) 

                # checkpoint = {'state_dict': reward_function1.state_dict(),'optimizer' :optimizer.state_dict()}
                # torch.save(checkpoint, 'Checkpoint.pth')




        wandb.log({f'train_shape_epoch/epoch_running_neg_mean':np.array(train_running_neg_mean).mean()})
        wandb.log({f'train_shape_epoch/epoch_running_pos_mean':np.array(train_running_pos_mean).mean()})
        wandb.log({f'train_shape_epoch/epoch_running_neut_mean':np.array(train_running_neut_mean).mean()})
        wandb.log({f'train_shape_epoch/epoch_running_sup_mean':np.array(train_running_sup_mean).mean()})
                   
                   
                   
        train_running_neg_mean = []
        train_running_pos_mean = []
        train_running_neut_mean = []
        train_running_sup_mean = []
        



        wandb.log({f'train/epoch_loss':train_running_loss/(len(train_loader))})
        wandb.log({f'train/epoch_shaped_loss':train_running_shaped_loss/(len(train_loader))})
        # wandb.log({f'train/epoch_div_loss':train_running_div_loss/(len(train_loader))})

        if args.contrastive and epoch % 2 == 1: 
            wandb.log({f'train/contrastive_loss':train_running_contrastive_loss/(len(train_loader))})
        
        train_running_shaped_loss = 0
        train_running_loss = 0
        train_running_div_loss = 0
        train_running_contrastive_loss = 0
    
        
    
