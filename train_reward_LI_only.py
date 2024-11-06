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
from data_loader_reward_single import CANDOR_LLAMA_K
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer, LongformerModel, AutoModelForSeq2SeqLM, BartForSequenceClassification, LlamaForSequenceClassification
from transformers import LongT5ForConditionalGeneration, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from options_reward import parser
from torch.nn.modules.loss import _Loss
from utils import ImbalancedDatasetSampler
from utils import SupConLoss


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

if __name__ == '__main__':

    wandb.login()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    loss_fn = torch.nn.MSELoss()
    rate_learning = 5e-6
    batch_size = int(args.batch_size) #64
    K = int(args.K)
    flip = False
    N_EPOCHS = 8
    train_size = int(args.train_size) #10#00#00
    

    WANDB_PROJECT = 'reward_function_{}_160_scaled_{}'.format(args.model, reward_class)

    if not args.unfreeze:
        WANDB_PROJECT = 'SINGLE_INDEX_reward_function_{}_{}_{}_{}_{}_{}'.format(args.model, reward_class, "contra_" + str(args.contrastive), "shrink_" + str(args.hard_shrink), "curriculum_" + str(args.curriculum),  "curriculum_exposure" + str(args.curriculum_exposure))
        K = K

    run = wandb.init(project=WANDB_PROJECT, 
           config={'lr':rate_learning, 'batch_size':batch_size, 'n_epochs':N_EPOCHS, 'K': K, 'train_size': train_size})

    now_time = "{}".format(str(run.name))

    folder_name = "./{}".format(WANDB_PROJECT)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if args.model == 'longT5':
        longformer_tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
        longformer_tokenizer.truncation_side='left'
        lang_model = LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps").cuda()

    if args.model == 'longformer':

        longformer_tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
        longformer_tokenizer.truncation_side='left'
        lang_model = LongformerModel.from_pretrained('allenai/longformer-base-4096').cuda()
        lang_model = torch.nn.DataParallel(lang_model, device_ids=[0, 1, 2, 3])

    if args.model == 'convo':
        longformer_tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
        lang_model = BartForSequenceClassification.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary", num_labels = 1).cuda()
        lang_model = torch.nn.DataParallel(lang_model, device_ids=[0, 1, 2, 3])
        hidden_size = 1024

    if args.model == 'LLAMA2':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig, TaskType
        longformer_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        lang_model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-chat-hf").cuda()
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        lang_model = get_peft_model(lang_model, peft_config)
        lang_model.print_trainable_parameters()
        hidden_size = 4096
        lang_model = torch.nn.DataParallel(lang_model, device_ids=[0, 1, 2, 3])

        longformer_tokenizer.pad_token_id = longformer_tokenizer.eos_token_id
        longformer_tokenizer.pad_token = longformer_tokenizer.eos_token
        lang_model.module.config.pad_token_id = longformer_tokenizer.pad_token_id

        


    reward_function1 = nn.Sequential(
    nn.Linear(hidden_size, 768),
    nn.ReLU(),
    nn.Linear(768, 128),
    nn.ReLU(),
    ).cuda()


    if args.hard_shrink:
        reward_function2 = nn.Sequential(nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.GELU()
            ).cuda()
    elif args.batch_norm:

        reward_function1 = nn.Sequential(
            nn.Linear(3072, 768),
            nn.LayerNorm(768,elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(768, 128),
            nn.LayerNorm(128,elementwise_affine=True),
            nn.ReLU(),
            ).cuda()

        reward_function2 = nn.Sequential(nn.Linear(128, 32),
            nn.LayerNorm(32,elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(32, 1),
            ).cuda()

    elif args.small_model:
        reward_function1 = nn.Sequential(nn.Linear(3072, 128),nn.ReLU()).cuda()
        reward_function2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)).cuda()
        
    elif args.small_model_shrink:

        class reward_function(nn.Module):
            def __init__(self):
                super(reward_function, self).__init__()
                self.fc1 = nn.Linear(hidden_size,128)
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

        reward_function1 =reward_function().cuda()
        
    else:
        reward_function2 = nn.Sequential(nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            ).cuda()


    DATA_PATH = './candor'

    
    interaction_path_list = os.listdir(DATA_PATH)
    
    transcript_list = []
    survey_list = []
    audio_visual_list = []
    flip = flip
    nan_list = utils.nan_list
    train_setup = 'reward'


    all_datasets = []
    for path in tqdm(interaction_path_list[0:train_size]):
        if path not in nan_list:
            transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
            survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
            audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
        
    
            train_dataset = CANDOR_LLAMA_K(transcript_path, survey_path, audio_visual, tokenizer = longformer_tokenizer, K = None, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = train_setup, sample_middle = True)
            all_datasets.append(train_dataset)
    
    full_train_set = torch.utils.data.ConcatDataset(all_datasets)

    


    all_datasets = []
    for path in tqdm(interaction_path_list[-50:] ):
        if path not in nan_list:
            transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
            survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
            audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
        
    
            train_dataset = CANDOR_LLAMA_K(transcript_path, survey_path, audio_visual, tokenizer = longformer_tokenizer, K = 64, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = train_setup, sample_middle = True)
            all_datasets.append(train_dataset)
    
    val_set = torch.utils.data.ConcatDataset(all_datasets)

    sampler = ImbalancedDatasetSampler(full_train_set)
    train_loader = DataLoader(
        dataset=full_train_set,
        # sampler = sampler,
        batch_size=batch_size
        # generator=torch.Generator(device='cuda')
    )


    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size
        # generator=torch.Generator(device='cuda')
    )






    model_params = list(reward_function1.parameters()) # + list(reward_function2.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=rate_learning, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_loader) * N_EPOCHS),
    )
    
    optimizer.add_param_group({'params': lang_model.parameters(), 'lr':rate_learning, 'name': 'language_model'})

    batch_count = 0
    best_val_loss = 1e10
    

    #TRAIN
    for epoch in range(N_EPOCHS):
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

            if args.model == 'LLAMA2':

                sentence_representation = lang_model(input_ids = encoded_tokens["input_ids"].squeeze(), attention_mask = encoded_tokens["attention_mask"].squeeze(), return_dict = True)

            else:
                part = lang_model(input_ids = encoded_tokens["input_ids"].squeeze(), attention_mask = encoded_tokens["attention_mask"].squeeze())
                part = part["encoder_last_hidden_state"]
                eos_mask = encoded_tokens["input_ids"].squeeze().eq(longformer_tokenizer.eos_token_id).to(part.device)
                sentence_representation = part[eos_mask, :].view(part.size(0), -1, part.size(-1))
                sentence_representation = sentence_representation[:,-1,:]


            all_outputs = sentence_representation.reshape(-1, hidden_size)
            pred_rewards = reward_function1(all_outputs)
            sum_rewards = (pred_rewards).squeeze().sum(-1)            
            shaped_rewards = torch.zeros_like(pred_rewards)



            neg_affect_indices = batch['reaction_visual'] <= 4
            pos_affect_indices = batch['reaction_visual'] == 6
            neutral_affect_indices = batch['reaction_visual'] == 5
            surprise_indices = batch['reaction_visual'] == 7

            shaped_rewards[neg_affect_indices]  = -1.0
            shaped_rewards[neutral_affect_indices] = 0.0
            shaped_rewards[pos_affect_indices] = 1.0
            shaped_rewards[surprise_indices] = 0.0

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
            # pred_rewards
 
            shaped_loss = loss_fn(pred_rewards.squeeze(), shaped_rewards.squeeze())
            print(shaped_loss)
            
            loss = shaped_loss


            contrastive_loss_full = torch.Tensor([0]).cuda()

            
            if args.contrastive and epoch % 2 == 1: 

                lam = 0.3
                
                tem = 0.3
                for i in range(hiden_state.shape[0]):


                    reshaped_hidden = hiden_state[i,...].reshape(-1,4,128)
                    labels =  torch.arange(40).cuda()

                    contrastive_loss_full += contrastive_criterion(reshaped_hidden)
                    print(contrastive_loss_full)
                    loss = (lam * contrastive_loss_full) + (1 - lam) * (mae_loss)


            
            if args.bar_chart: 

                data = [[label, val] for (label, val) in zip(batch['indices'][0], pred_rewards[0].detach().squeeze())]
                table = wandb.Table(data=data, columns = ["label", "value"])
                wandb.log({"rewards" : wandb.plot.bar(table, "label", "value",
                                            title="predicted rewards1_{}".format(batch['reward'][0].item()))})
                
           
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()



            

            #logging 
            wandb.log({f'train/shaped_loss':shaped_loss.item()})
            # wandb.log({f'train/loss':mae_loss.item()})

            wandb.log({f'train_shape/train_neg_mean':neg_affect_reward_mean})
            wandb.log({f'train_shape/train_pos_mean':pos_affect_reward_mean})
            wandb.log({f'train_shape/train_neut_mean':neut_affect_reward_mean})
            wandb.log({f'train_shape/train_sup_mean':sup_affect_reward_mean})

            # wandb.log({f'train/div_loss':divergence_loss.item()})
            
            train_running_shaped_loss += shaped_loss.item()
            # train_running_loss += mae_loss.item()
            # train_running_div_loss += divergence_loss.item()
            train_running_contrastive_loss += contrastive_loss_full.item()

        #VALIDATION -- uncomment
            batch_count += 1
            if batch_count % 500 == 0 :
                print("\n ********RUNNING VALIDATION STEP*********** \n")
                if args.val: 
                    val_running_neg_mean = []
                    val_running_pos_mean = []
                    val_running_neut_mean = []
                    val_running_sup_mean = []

                    val_running_loss = 0
                    middle_val_running_loss = 0
                    for val_idx, val_batch in enumerate(tqdm(val_loader)):
                        with torch.no_grad():
                            speaker_A_tokens = val_batch['A_sentence_llama']
                            speaker_B_tokens = val_batch['B_sentence_llama']
                            rewards = val_batch['reward']

                            encoded_tokens = speaker_A_tokens

                            # decoded_tokens = tokenizer.batch_decode(speaker_A_tokens["input_ids"].squeeze(1), skip_special_tokens  = True)
                            # encoded_tokens = longformer_tokenizer(decoded_tokens, return_tensors="pt", padding = 'max_length', max_length = 1024).to("cuda")

                            try:
                                part = lang_model(input_ids = encoded_tokens["input_ids"].squeeze(), attention_mask = encoded_tokens["attention_mask"].squeeze())
                            except Exception:
                                pdb.set_trace()
                            
                            part = part["encoder_last_hidden_state"]
                            eos_mask = encoded_tokens["input_ids"].squeeze().eq(longformer_tokenizer.eos_token_id).to(part.device)
                            
                            sentence_representation = part[eos_mask, :].view(part.size(0), -1, part.size(-1))
                            sentence_representation = sentence_representation[:,-1,:]
                                                

                            all_outputs = sentence_representation.reshape(-1, hidden_size)

                            pred_rewards = reward_function1(all_outputs)
                            sum_rewards = (pred_rewards).squeeze().sum(-1)
                            shaped_rewards = torch.zeros_like(pred_rewards)



                            neg_affect_indices = val_batch['reaction_visual'] <= 4
                            pos_affect_indices = val_batch['reaction_visual'] == 6
                            neutral_affect_indices = val_batch['reaction_visual'] == 5
                            surprise_indices = val_batch['reaction_visual'] == 7

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

                            shaped_loss = loss_fn(shaped_rewards.squeeze(), pred_rewards.squeeze())
                            
                            loss = shaped_loss


                            #logging 
                            wandb.log({f"val/loss": loss.item()})
                            # wandb.log({f"val/middle_loss": middle_loss.item()})

                            # middle_val_running_loss += middle_loss.item()
                            val_running_loss += loss.item()

                    epoch_val_loss = val_running_loss/(len(val_loader)) #len(val_loader)
                    # epoch_val_middle_loss= middle_val_running_loss/(len(val_loader))
                    # wandb.log({f'val/epoch_val_middle_loss':epoch_val_middle_loss})
                    wandb.log({f'val/epoch_loss':epoch_val_loss})

                    #saving model --uncomment 
                    if epoch_val_loss <best_val_loss:
                        best_val_loss = epoch_val_loss
                        lang_model.module.save_pretrained("./{}/lang_model_{}".format(WANDB_PROJECT,now_time), from_pt=True) 
                        torch.save(reward_function1,"./{}/reward1_{}".format(WANDB_PROJECT,now_time)) 

                    wandb.log({f'val_shape_epoch/epoch_running_neg_mean':np.array(val_running_neg_mean).mean()})
                    wandb.log({f'val_shape_epoch/epoch_running_pos_mean':np.array(val_running_pos_mean).mean()})
                    wandb.log({f'val_shape_epoch/epoch_running_neut_mean':np.array(val_running_neut_mean).mean()})
                    wandb.log({f'val_shape_epoch/epoch_running_sup_mean':np.array(val_running_sup_mean).mean()})
                        

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

        if args.contrastive and epoch % 2 == 1: 
            wandb.log({f'train/contrastive_loss':train_running_contrastive_loss/(len(train_loader))})
        
        train_running_shaped_loss = 0
        train_running_loss = 0
        train_running_div_loss = 0
        train_running_contrastive_loss = 0
    
        
    
