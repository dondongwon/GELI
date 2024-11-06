import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle 
from transformers import AutoTokenizer, GenerationConfig, get_linear_schedule_with_warmup, BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoModelForSequenceClassification
# from transformers import LlamaTokenizerFast
from statistics import mode 
import ast
import utils
import os
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, AutoModelForSeq2SeqLMWithValueHead
import datetime

#'overall_affect', 'overall_memory_rating', 'how_enjoyable','i_like_you', 'in_common', 'conversationalist', 'good_for_advice', 'you_are_intelligent', 'you_are_quickwitted', 'you_are_competent', 'you_are_kind', 'you_are_friendly', 'you_are_warm'
class CANDOR_LLAMA_K(torch.utils.data.Dataset):
    def __init__(self, transcript_path, survey_path, audio_visual, tokenizer, K=32, flip = False, DATA_PATH = '/u/dongwonl/rlmf/candor', utils = utils, split = 'train', train_size = 1000, train_setup = 'reward', sample_middle = False, reward_class = 'overall_affect', curriculum = False):
        
        self.K = K
        self.tokenizer = tokenizer 
        self.interaction_path_list = os.listdir(DATA_PATH)
        self.transcript_list = []
        self.survey_list = []
        self.audio_visual_list = []
        self.flip = flip
        self.nan_list = utils.nan_list
        self.reward_mean = utils.reward_dict['mean']
        self.reward_std = utils.reward_dict['mean']
        self.train_setup = train_setup
        self.tokenizer.truncation_side='left'
        self.sample_middle = sample_middle
        self.reward_class = reward_class
        self.curriculum = curriculum
        

        self.survey_path = survey_path
        self.transcript_path = transcript_path


        
        self.audio_visual_data = pd.read_csv(audio_visual)
        self.audio_visual_data['seconds'] = pd.to_timedelta(self.audio_visual_data['timedelta']).dt.total_seconds()
        self.transcript = pd.read_csv(self.transcript_path)
        self.survey = pd.read_csv(self.survey_path)
        self.transcript_len = len(self.transcript)

        random.seed(3)
        if  self.K:
            try:
                self.random_indices = random.choices(list(range(5, self.transcript_len-5)), k=self.K)
            except Exception:
                pdb.set_trace()


        if getattr(self.tokenizer, "pad_token", None) is None:
            if self.train_setup == 'reward':
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            if self.train_setup == 'rl':
                self.tokenizer.pad_token = self.tokenizer.pad_token

        if self.train_setup == 'rl':
                self.tokenizer.pad_token = self.tokenizer.pad_token
                self.tokenizer.padding_side = 'left'

                
 
        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def __len__(self):

        if self.K:
            return self.K
        else:
            return len(self.transcript) - 5

    def __getitem__(self, index):

        transcript_df = self.transcript
        survey_df = self.survey
        audio_visual_data_df = self.audio_visual_data


        speaker_A_name = transcript_df['speaker'].iloc[0]
        speaker_B_name = transcript_df['speaker'].iloc[1]


        
        all_utterances = transcript_df['utterance'].tolist()
        all_starts = transcript_df['start'].tolist()
        all_ends = transcript_df['stop'].tolist()
        all_n_words = transcript_df['n_words'].tolist()
        
        if self.K:
            index = self.random_indices[index]
            
        else:
            index = index + 3

        #get at least 2 sentences in context 
        



        #sample index 

        
        sampled_utterances = [all_utterances[:index]]
        response_utterances = ["AI: " + all_utterances[index]]
        speaker_response_utterances = [all_utterances[index+1]]

        response_start_end = np.array([(all_starts[index], all_ends[index])])
        speaker_response_start_end = np.array([(all_starts[index+1], all_ends[index+1])])
        n_words = np.array([all_n_words[index]])



        prompt = ["You are AI. You are having a casual social conversation with a User. AI is polite, kind, obedient, honest, and does not swear. You are given the dialogue history, generate the response to the User in under 100 words."]
        dialogue_prompt = [] 

        prompt_end = ["AI:"]


        #if reward we want .... AI resposne included in speaker A utterance
        #if rl we want User response as final included inspeaker A utterance
        

        samples = []
        for context_ind, context in enumerate(sampled_utterances):
            context_utterances = []

            for i in range(len(context)):
                if len(context) % 2 == 1:
                    if i % 2 == 1:
                        context_utterances.append(" AI: " + context[i])
                    else: 
                        context_utterances.append(" User: " + context[i])

                if len(context) % 2 == 0:
                    if i % 2 == 0:
                        context_utterances.append(" AI: " + context[i])
                    else: 
                        context_utterances.append(" User: " + context[i])

            # pair of sequences: `<s> A </s></s> B </s>`
            
            #finished populating contexzt
            #add response
            if self.train_setup == 'reward':
                if self.curriculum:
                    context_utterances.append("</s>"+ response_utterances[context_ind] + "</s>" + "</s>"+" User: " + speaker_response_utterances[context_ind] ) #+ "</s>")
                else:
                    context_utterances.append("</s></s> "+ response_utterances[context_ind])
            if self.train_setup == 'rl':
                pass

            if len(context_utterances) > 20:
                # context_utterances = context_utterances[-20:]
                context_utterances = context_utterances[-10:]

            #add prompt in the beginning 
            # samples.append(" ".join(prompt + context_utterances))
            #dont add 
            # samples.append(" ".join(context_utterances))

            #             #add prompt in the beginning 
            if self.train_setup == 'rl':
                samples.append(" ".join(prompt + context_utterances + prompt_end))
            else:
                samples.append(" ".join(context_utterances + prompt_end))
        

        aligned_audio_visual_all = []

        audio_visual_data_df = audio_visual_data_df.rename(columns={'prob_face_anger': 0, 'prob_face_contempt': 1, 'prob_face_disgust':2, 'prob_face_fear':3, 'prob_face_sadness':4, 'prob_face_neutral':5, 'prob_face_happiness':6, 'prob_face_surprise':7})
        
        
        aligned_audio_visual_subset = audio_visual_data_df[(audio_visual_data_df["seconds"] >= response_start_end[0][0]) & (audio_visual_data_df["seconds"] <= response_start_end[0][1])]
        aligned_audio_visual_subset = aligned_audio_visual_subset[aligned_audio_visual_subset['user_id'] == speaker_A_name]
        aligned_audio_visual_subset = aligned_audio_visual_subset[[0,1,2,3,4,5,6,7]]
        aligned_audio_visual_subset = aligned_audio_visual_subset.dropna()
        if n_words <=5: 
            aligned_audio_visual_all.append(5)
        elif aligned_audio_visual_subset.empty:
            aligned_audio_visual_all.append(5)
            
        else:
            mode_emo = aligned_audio_visual_subset.idxmax(axis=1).value_counts().index.tolist()[0]
            if mode_emo <= 4:
                aligned_audio_visual_all.append(5)
            elif mode_emo == 7:
                aligned_audio_visual_all.append(5)
            else:
                aligned_audio_visual_all.append(mode_emo)
            # aligned_audio_visual_all.append(aligned_audio_visual_subset.dropna())
        # for i in range(len(aligned_audio_visual_all)): 
        #     if not aligned_audio_visual_all[i].empty:
        #         print(aligned_audio_visual_all[i].idxmax(axis=1).value_counts().index.tolist()[0])
        
        # pdb.set_trace()

        # aligned_audio_visual = self.audio_visual_data[(self.audio_visual_data["seconds"] <= 613.26) & (self.audio_visual_data["seconds"] >= 597.74)]


        #find affective cues
        
        # Sentences we want sentence embeddings for

        # Load model from HuggingFace Hub


        if self.train_setup == 'rl':
            speaker_A_sentbert_tokens = self.tokenizer(samples, padding='max_length', truncation=True, return_tensors='pt', max_length =1024)
        if self.train_setup == 'reward':
            speaker_A_sentbert_tokens = self.tokenizer(samples, padding='max_length', truncation=True, return_tensors='pt', max_length =1024)

        speaker_B_sentbert_tokens = self.tokenizer(response_utterances, padding='max_length', truncation=True, return_tensors='pt', max_length =1024)
        
        
        
        # self.tokenizer.decode(**speaker_A_sentbert_tokens)


        # Tokenize sentences
        # speaker_A_sentbert_tokens = self.tokenizer(speaker_A_utterances, padding=True, truncation=True, return_tensors='pt')
        # speaker_B_sentbert_tokens = self.tokenizer(speaker_B_utterances, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        # with torch.no_grad(): speaker_A_sentbert = self.model(**speaker_A_sentbert_tokens)
        # with torch.no_grad():
        #     speaker_A_sentbert = self.model(**speaker_A_sentbert_tokens)
        #     speaker_B_sentbert = self.model(**speaker_B_sentbert_tokens)
            
        # pdb.set_trace()

        # # Perform pooling
        
        # speaker_A_sentbert_embeddings = self.mean_pooling(speaker_A_sentbert, speaker_A_sentbert_tokens['attention_mask'])
        # speaker_B_sentbert_embeddings = self.mean_pooling(speaker_B_sentbert, speaker_B_sentbert_tokens['attention_mask'])
        
        
        # # Normalize embeddings
        # speaker_A_sentbert_embeddings = F.normalize(speaker_A_sentbert_embeddings, p=2, dim=1)
        # speaker_B_sentbert_embeddings = F.normalize(speaker_B_sentbert_embeddings, p=2, dim=1)
        
        #pad embeddings 
        # pad_A = 300 - speaker_A_sentbert_embeddings.shape[0]
        # speaker_A_sentbert_embeddings = F.pad(speaker_A_sentbert_embeddings, pad=(0, 0, 0, pad_A))
    
    
        # pad_B = 300 - speaker_B_sentbert_embeddings.shape[0]
        # speaker_B_sentbert_embeddings = F.pad(speaker_B_sentbert_embeddings, pad=(0, 0, 0, pad_B))

        speaker_A_survey = survey_df[survey_df['user_id'] == speaker_A_name]
        speaker_B_survey = survey_df[survey_df['user_id'] == speaker_B_name]

        # conversationalist

        batch = dict()
        batch['A_sentence_llama'] = speaker_A_sentbert_tokens
        batch['B_sentence_llama'] = speaker_B_sentbert_tokens
        # batch['A_sentence_pad'] = 300 - pad_A
        # batch['B_sentence_pad'] = 300 - pad_B

        if self.flip:
            if self.reward_class in ['good_for_advice', 'conversationalist']:
                batch['reward'] = (speaker_B_survey[self.reward_class].item()) #*(len_episode//2)
            else:
                batch['reward'] = (speaker_B_survey[self.reward_class].item()) * 10
            # batch['middle_reward'] = (speaker_B_survey['middle_affect'].item())* 10 #*(len_episode//4)
        else:
            if self.reward_class in ['good_for_advice', 'conversationalist']:
                batch['reward'] = (speaker_A_survey[self.reward_class].item())
            else:
                batch['reward'] = (speaker_A_survey[self.reward_class].item()) * 10 #*(len_episode//2)
            # batch['middle_reward'] = (speaker_A_survey['middle_affect'].item()) * 10 #*(len_episode//4)

        
        
        batch['reaction_visual'] = torch.Tensor(aligned_audio_visual_all)
        # for k,v in batch.items():
        #     print(v.shape)
        return batch 
    

if __name__ == '__main__':
    print("Finished Loading...")
    batch_size = 16
    K = None
    flip = False
    N_EPOCHS = 20
    train_size = 100 #change to 1000


    # model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #     pretrained_model_name_or_path= "/u/dongwonl/rlmf/llama_hf",
    #     # load_in_8bit=True,
    #     device_map="auto",#{"": current_device},
    #     )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model= nn.DataParallel(model)
    # model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("/u/dongwonl/rlmf/llama_hf")
    DATA_PATH = '/u/dongwonl/rlmf/candor'

    interaction_path_list = os.listdir(DATA_PATH)
    transcript_list = []
    survey_list = []
    audio_visual_list = []
    flip = flip
    nan_list = utils.nan_list

    split = 'train'
    train_setup = 'rl'
    train_size = 10

    if split == 'train':
        if train_setup == 'reward':
            interaction_path_list = interaction_path_list[0:train_size]
        if train_setup == 'rl':
            interaction_path_list = interaction_path_list[1200: 1200+train_size]
        # self.interaction_path_list = self.interaction_path_list[:-100]

    if split == 'val':
        interaction_path_list = interaction_path_list[-100:] 

    all_datasets = []
    for path in tqdm(interaction_path_list):
        if path not in nan_list:
            transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
            survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
            audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
        
    
            train_dataset = CANDOR_LLAMA_K(transcript_path, survey_path, audio_visual, tokenizer = tokenizer, K = None, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = train_setup, sample_middle = True)
            all_datasets.append(train_dataset)
    
    full_train_set = torch.utils.data.ConcatDataset(all_datasets)

    # train_dataset_unflip = CANDOR_LLAMA_K(tokenizer = tokenizer, K = 1, flip = False, utils = utils, split = 'train', train_size=train_size)


    data_loader = DataLoader(
        dataset=full_train_set,
        batch_size=batch_size
        # generator=torch.Generator(device='cuda')
    )

    tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")


    DATA_PATH = '/u/dongwonl/rlmf/candor'

    interaction_path_list = os.listdir(DATA_PATH)
    transcript_list = []
    survey_list = []
    audio_visual_list = []
    flip = flip
    nan_list = utils.nan_list

    split = 'train'
    train_setup = 'reward'
    train_size = 600

    if split == 'train':
        if train_setup == 'reward':
            interaction_path_list = interaction_path_list[0:train_size]
        if train_setup == 'rl':
            interaction_path_list = interaction_path_list[1200: 1200+train_size]
        # self.interaction_path_list = self.interaction_path_list[:-100]

    if split == 'val':
        interaction_path_list = interaction_path_list[-100:] 

    all_datasets = []
    for path in tqdm(interaction_path_list):
        if path not in nan_list:
            transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
            survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
            audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
        
    
            train_dataset = CANDOR_LLAMA_K(transcript_path, survey_path, audio_visual, tokenizer = tokenizer, K = None, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = train_setup, sample_middle = True)
            all_datasets.append(train_dataset)
    
    full_train_set = torch.utils.data.ConcatDataset(all_datasets)

    # train_dataset_unflip = CANDOR_LLAMA_K(tokenizer = tokenizer, K = 1, flip = False, utils = utils, split = 'train', train_size=train_size)


    data_loader = DataLoader(
        dataset=full_train_set,
        batch_size=batch_size
        # generator=torch.Generator(device='cuda')
    )


    #, interaction_id ='1c0e34d9-5c5c-4e74-9046-216fcd5cbd6d'

    # DataLoader(DS, batch_size = 2, shuffle = False, num_workers = 0)

    for idx, batch in enumerate(tqdm(data_loader)):
        batch = batch

        print(idx)
