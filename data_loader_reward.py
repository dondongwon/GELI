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
import pdb


class CANDOR_SentenceBert_K(torch.utils.data.Dataset):
    def __init__(self, tokenizer, model, K=32, flip = False, DATA_PATH = '/u/dongwonl/rlmf/candor', utils = utils, split = 'train', train_size = -100):
        
        self.K = K
        self.model = model
        self.tokenizer = tokenizer 
        self.tokenizer.truncation_side='left'
        self.interaction_path_list = os.listdir(DATA_PATH)
        self.transcript_list = []
        self.survey_list = []
        self.flip = flip
        self.nan_list = utils.nan_list
        self.reward_mean = utils.reward_dict['mean']
        self.reward_std = utils.reward_dict['mean']


        if split == 'train':
            self.interaction_path_list = self.interaction_path_list[100:100+train_size]
            # self.interaction_path_list = self.interaction_path_list[:-100]

        if split == 'val':
            self.interaction_path_list = self.interaction_path_list[-20:] #-100
        

        for path in tqdm(self.interaction_path_list):
            if path not in self.nan_list:
                self.transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv')
                self.survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
                self.transcript = pd.read_csv(self.transcript_path)
                self.survey = pd.read_csv(self.survey_path)
                self.transcript_list.append(self.transcript)
                self.survey_list.append(self.survey)
        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def __len__(self):
        return len(self.transcript_list)

    def __getitem__(self, index):
        transcript_df = self.transcript_list[index]
        survey_df = self.survey_list[index]

        speaker_A_name = transcript_df['speaker'].iloc[0]
        speaker_B_name = transcript_df['speaker'].iloc[1]
        
        all_utterances = transcript_df['utterance'].tolist()

        #sample index 

        

        # all_utterances = [s + self.tokenizer.sep_token for s in all_utterances]
        if not self.flip:
            speaker_A_utterances = all_utterances[0:-1:2]
            speaker_B_utterances = all_utterances[1::2]
            len_episode = len(speaker_B_utterances)
            index_list = list(range(len(speaker_B_utterances)))
        
        if self.flip:
            speaker_A_utterances = all_utterances[2::2]
            speaker_B_utterances = all_utterances[1:-1:2]        
            len_episode = len(speaker_A_utterances)
            index_list = list(range(len(speaker_A_utterances)))
        

 
        if self.K:
            uniform_indices = random.choices(index_list, k = self.K)
            speaker_A_utterances = [speaker_A_utterances[i] for i in uniform_indices]
            speaker_B_utterances = [speaker_B_utterances[i] for i in uniform_indices]

        
        
        # Sentences we want sentence embeddings for

        # Load model from HuggingFace Hub


        # Tokenize sentences
        speaker_A_sentbert_tokens = self.tokenizer(speaker_A_utterances, padding=True, truncation=True, return_tensors='pt')
        speaker_B_sentbert_tokens = self.tokenizer(speaker_B_utterances, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            speaker_A_sentbert = self.model(**speaker_A_sentbert_tokens)
            speaker_B_sentbert = self.model(**speaker_B_sentbert_tokens)
            
        

        # Perform pooling
        
        speaker_A_sentbert_embeddings = self.mean_pooling(speaker_A_sentbert, speaker_A_sentbert_tokens['attention_mask'])
        speaker_B_sentbert_embeddings = self.mean_pooling(speaker_B_sentbert, speaker_B_sentbert_tokens['attention_mask'])
        
        
        # Normalize embeddings
        speaker_A_sentbert_embeddings = F.normalize(speaker_A_sentbert_embeddings, p=2, dim=1)
        speaker_B_sentbert_embeddings = F.normalize(speaker_B_sentbert_embeddings, p=2, dim=1)
        
        #pad embeddings 
        # pad_A = 300 - speaker_A_sentbert_embeddings.shape[0]
        # speaker_A_sentbert_embeddings = F.pad(speaker_A_sentbert_embeddings, pad=(0, 0, 0, pad_A))
    
    
        # pad_B = 300 - speaker_B_sentbert_embeddings.shape[0]
        # speaker_B_sentbert_embeddings = F.pad(speaker_B_sentbert_embeddings, pad=(0, 0, 0, pad_B))

        speaker_A_survey = survey_df[survey_df['user_id'] == speaker_A_name]
        speaker_B_survey = survey_df[survey_df['user_id'] == speaker_B_name]

        # conversationalist

        batch = dict()
        batch['A_sentence_bert'] = speaker_A_sentbert_embeddings
        batch['B_sentence_bert'] = speaker_B_sentbert_embeddings
        # batch['A_sentence_pad'] = 300 - pad_A
        # batch['B_sentence_pad'] = 300 - pad_B
        batch['A_survey_conversationalist'] = (speaker_A_survey['conversationalist'].item() - self.reward_mean)/self.reward_mean
        batch['B_survey_conversationalist'] = (speaker_B_survey['conversationalist'].item() - self.reward_mean)/self.reward_mean
        batch['convo_length'] = len_episode
        # for k,v in batch.items():
        #     print(v.shape)
        return batch 


#'overall_affect', 'overall_memory_rating', 'how_enjoyable','i_like_you', 'in_common', 'conversationalist', 'good_for_advice', 'you_are_intelligent', 'you_are_quickwitted', 'you_are_competent', 'you_are_kind', 'you_are_friendly', 'you_are_warm'
class CANDOR_LLAMA_K(torch.utils.data.Dataset):
    def __init__(self, tokenizer, K=32, flip = False, DATA_PATH = '/u/dongwonl/rlmf/candor', utils = utils, split = 'train', train_size = 1000, train_setup = 'reward', sample_middle = False, reward_class = 'overall_affect', curriculum = False, reaction_type = 'mode'):
        
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
        self.reaction_type = reaction_type


        if getattr(self.tokenizer, "pad_token", None) is None:
            if self.train_setup == 'reward':
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            if self.train_setup == 'rl':
                self.tokenizer.pad_token = self.tokenizer.eos_token
 
        if split == 'train':
            if train_setup == 'reward':
                self.interaction_path_list = self.interaction_path_list[0:train_size]
            if train_setup == 'rl':
                self.interaction_path_list = self.interaction_path_list[300: 300+train_size]
            # self.interaction_path_list = self.interaction_path_list[:-100]

        if split == 'val':
            self.interaction_path_list = self.interaction_path_list[-150:] #[-100:]
            # self.interaction_path_list = self.interaction_path_list[0:train_size]
            # self.interaction_path_list = self.interaction_path_list[-10:] #[-100:]
        

        for path in tqdm(self.interaction_path_list):
            if path not in self.nan_list:
                self.transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
                self.survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
                self.audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
                self.audio_visual_data = pd.read_csv(self.audio_visual)
                self.audio_visual_data['seconds'] = pd.to_timedelta(self.audio_visual_data['timedelta']).dt.total_seconds()
                self.transcript = pd.read_csv(self.transcript_path)
                self.survey = pd.read_csv(self.survey_path)
                self.audio_visual_list.append(self.audio_visual_data)
                self.transcript_list.append(self.transcript)
                self.survey_list.append(self.survey)
        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def __len__(self):
        return len(self.transcript_list)

    def __getitem__(self, index):
        transcript_df = self.transcript_list[index]
        survey_df = self.survey_list[index]
        audio_visual_data_df = self.audio_visual_list[index]


        #get_visual
        

        # 'prob_face_anger', 'prob_face_contempt', 'prob_face_disgust', 'prob_face_fear', 'prob_face_happiness', 'prob_face_neutral', 'prob_face_sadness', 'prob_face_surprise',


        #'prob_face_anger', 
        #'prob_face_contempt', 
        # 'prob_face_disgust', 
        # 'prob_face_fear',
        #  'prob_face_happiness',
        #  'prob_face_neutral', 
        # 'prob_face_sadness', 'prob_face_surprise',


        if self.sample_middle: 
            #sample 100 from middle
            mid_start = max(0, len(transcript_df)//2 - 50)
            mid_end = min(len(transcript_df)//2 + 50, len(transcript_df))
            transcript_df = transcript_df.iloc[mid_start:mid_end]


        speaker_A_name = transcript_df['speaker'].iloc[0]
        speaker_B_name = transcript_df['speaker'].iloc[1]


        
        all_utterances = transcript_df['utterance'].tolist()
        all_starts = transcript_df['start'].tolist()
        all_ends = transcript_df['stop'].tolist()
        all_n_words = transcript_df['n_words'].tolist()

        #sample index 

        
        # all_utterances = [s + self.tokenizer.sep_token for s in all_utterances]
        if not self.flip:
            len_episode = len(all_utterances)
            index_list = list(range(len(all_utterances)))
        
        if self.flip:
            all_utterances[1:]
            len_episode = len(all_utterances)
            index_list = list(range(len(all_utterances)))
        
        if self.K:
            uniform_indices = random.choices(index_list[1:-1:2], k = self.K)
            uniform_indices = np.sort(uniform_indices)
            sampled_utterances = [all_utterances[:i] for i in uniform_indices]
            response_utterances = ["AI: " + all_utterances[i] for i in uniform_indices]
            speaker_response_utterances = [all_utterances[i+1] for i in uniform_indices]

            response_start_end = np.array([(all_starts[i], all_ends[i])for i in uniform_indices])
            speaker_response_start_end = np.array([(all_starts[i+1], all_ends[i+1]) for i in uniform_indices])
            n_words = np.array([all_n_words[i] for i in uniform_indices ])

        else: 
            uniform_indices = list(range(len(all_utterances)))[1:-1:2]
            sampled_utterances = [all_utterances[1:i] for i in uniform_indices]
            response_utterances = ["AI: " + all_utterances[i] for i in uniform_indices]
            speaker_response_utterances = [all_utterances[i+1] for i in uniform_indices]

            response_start_end = np.array([(all_starts[i], all_ends[i]) for i in uniform_indices])
            speaker_response_start_end = np.array([(all_starts[i+1], all_ends[i+1]) for i in uniform_indices])
            n_words = np.array([all_n_words[i] for i in uniform_indices])


        prompt = ["You are having a conversation with a User, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits. You are given the dialogue history below, generate a relevant response to the User's last response. Don't generate User's response"]
        dialogue_prompt = [] 


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

            
            #finished populating contexzt
            #add response
            if self.train_setup == 'reward':
                if self.curriculum:
                    context_utterances.append("</s>"+ response_utterances[context_ind] + "</s>" + "</s>"+" User: " + speaker_response_utterances[context_ind] ) #+ "</s>")
                else:
                    context_utterances.append("</s>"+ response_utterances[context_ind]) #+ "</s>")
            if self.train_setup == 'rl':
                pass

            if len(context_utterances) > 20:
                context_utterances = context_utterances[-20:]

            #add prompt in the beginning 
            if self.train_setup == 'rl':
                samples.append(" ".join(prompt + context_utterances))
            else:
                samples.append(" ".join(context_utterances))


        aligned_audio_visual_all = []
        intensity_score = []

        audio_visual_data_df = audio_visual_data_df.rename(columns={'prob_face_anger': 0, 'prob_face_contempt': 1, 'prob_face_disgust':2, 'prob_face_fear':3, 'prob_face_sadness':4, 'prob_face_neutral':5, 'prob_face_happiness':6, 'prob_face_surprise':7})
        
        for i in range(len(uniform_indices)):
            # aligned_audio_visual_subset = audio_visual_data_df[(audio_visual_data_df["seconds"] >= response_start_end[i][0]) & (audio_visual_data_df["seconds"] <= response_start_end[i][1])]
            # aligned_audio_visual_subset = aligned_audio_visual_subset[aligned_audio_visual_subset['user_id'] == speaker_A_name]
            # aligned_audio_visual_subset = aligned_audio_visual_subset[[0,1,2,3,4,5,6,7]]
            # aligned_audio_visual_subset = aligned_audio_visual_subset.dropna()
            # if all_n_words[uniform_indices[i]] <=7 or aligned_audio_visual_subset.empty:
            #     aligned_audio_visual_all.append(5)
            # else:
            #     aligned_audio_visual_all.append(aligned_audio_visual_subset.idxmax(axis=1).value_counts().index.tolist()[0])

            aligned_audio_visual_subset = audio_visual_data_df[(audio_visual_data_df["seconds"] >= response_start_end[i][0]) & (audio_visual_data_df["seconds"] <= response_start_end[i][1])]
            aligned_audio_visual_subset = aligned_audio_visual_subset[aligned_audio_visual_subset['user_id'] == speaker_A_name]
            aligned_audio_visual_subset = aligned_audio_visual_subset[[0,1,2,3,4,5,6,7]]
            aligned_audio_visual_subset = aligned_audio_visual_subset.dropna()
            
            if all_n_words[uniform_indices[i]] <=5: 
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

            if self.reaction_type == 'mean':
                if aligned_audio_visual_subset.empty:
                    intensity_score.append(0) 
                else:
                    intensity = (aligned_audio_visual_subset.idxmax(axis=1).values == 6).mean()
                    intensity_score.append(intensity) 

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

        
        
        
        batch['convo_length'] = len_episode
        batch['indices'] = uniform_indices
        batch['reaction_visual'] = torch.Tensor(aligned_audio_visual_all)
        batch['reaction_visual_intensity'] = intensity_score
        # for k,v in batch.items():
        #     print(v.shape)
        return batch 
    

#'overall_affect', 'overall_memory_rating', 'how_enjoyable','i_like_you', 'in_common', 'conversationalist', 'good_for_advice', 'you_are_intelligent', 'you_are_quickwitted', 'you_are_competent', 'you_are_kind', 'you_are_friendly', 'you_are_warm'
class CANDOR_LLAMA_K_SmallVal(torch.utils.data.Dataset):
    def __init__(self, tokenizer, K=32, flip = False, DATA_PATH = '/u/dongwonl/rlmf/candor', utils = utils, split = 'train', train_size = 1000, train_setup = 'reward', sample_middle = False, reward_class = 'overall_affect', curriculum = False, reaction_type = 'mode'):
        
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
        self.reaction_type = reaction_type


        if getattr(self.tokenizer, "pad_token", None) is None:
            if self.train_setup == 'reward':
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            if self.train_setup == 'rl':
                self.tokenizer.pad_token = self.tokenizer.eos_token
 
        if split == 'train':
            if train_setup == 'reward':
                self.interaction_path_list = self.interaction_path_list[0:train_size]
            if train_setup == 'rl':
                self.interaction_path_list = self.interaction_path_list[300: 300+train_size]
            # self.interaction_path_list = self.interaction_path_list[:-100]

        if split == 'val':
            self.interaction_path_list = self.interaction_path_list[-50:] #[-100:]
            # self.interaction_path_list = self.interaction_path_list[0:train_size]
            # self.interaction_path_list = self.interaction_path_list[-10:] #[-100:]
        

        for path in tqdm(self.interaction_path_list):
            if path not in self.nan_list:
                self.transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
                self.survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
                self.audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
                self.audio_visual_data = pd.read_csv(self.audio_visual)
                self.audio_visual_data['seconds'] = pd.to_timedelta(self.audio_visual_data['timedelta']).dt.total_seconds()
                self.transcript = pd.read_csv(self.transcript_path)
                self.survey = pd.read_csv(self.survey_path)
                self.audio_visual_list.append(self.audio_visual_data)
                self.transcript_list.append(self.transcript)
                self.survey_list.append(self.survey)
        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def __len__(self):
        return len(self.transcript_list)

    def __getitem__(self, index):
        transcript_df = self.transcript_list[index]
        survey_df = self.survey_list[index]
        audio_visual_data_df = self.audio_visual_list[index]


        #get_visual
        

        # 'prob_face_anger', 'prob_face_contempt', 'prob_face_disgust', 'prob_face_fear', 'prob_face_happiness', 'prob_face_neutral', 'prob_face_sadness', 'prob_face_surprise',


        #'prob_face_anger', 
        #'prob_face_contempt', 
        # 'prob_face_disgust', 
        # 'prob_face_fear',
        #  'prob_face_happiness',
        #  'prob_face_neutral', 
        # 'prob_face_sadness', 'prob_face_surprise',


        if self.sample_middle: 
            #sample 100 from middle
            mid_start = max(0, len(transcript_df)//2 - 50)
            mid_end = min(len(transcript_df)//2 + 50, len(transcript_df))
            transcript_df = transcript_df.iloc[mid_start:mid_end]


        speaker_A_name = transcript_df['speaker'].iloc[0]
        speaker_B_name = transcript_df['speaker'].iloc[1]


        
        all_utterances = transcript_df['utterance'].tolist()
        all_starts = transcript_df['start'].tolist()
        all_ends = transcript_df['stop'].tolist()
        all_n_words = transcript_df['n_words'].tolist()

        #sample index 

        
        # all_utterances = [s + self.tokenizer.sep_token for s in all_utterances]
        if not self.flip:
            len_episode = len(all_utterances)
            index_list = list(range(len(all_utterances)))
        
        if self.flip:
            all_utterances[1:]
            len_episode = len(all_utterances)
            index_list = list(range(len(all_utterances)))
        
        if self.K:
            uniform_indices = random.choices(index_list[1:-1:2], k = self.K)
            uniform_indices = np.sort(uniform_indices)
            sampled_utterances = [all_utterances[:i] for i in uniform_indices]
            response_utterances = ["AI: " + all_utterances[i] for i in uniform_indices]
            speaker_response_utterances = [all_utterances[i+1] for i in uniform_indices]

            response_start_end = np.array([(all_starts[i], all_ends[i])for i in uniform_indices])
            speaker_response_start_end = np.array([(all_starts[i+1], all_ends[i+1]) for i in uniform_indices])
            n_words = np.array([all_n_words[i] for i in uniform_indices ])

        else: 
            uniform_indices = list(range(len(all_utterances)))[1:-1:2]
            sampled_utterances = [all_utterances[1:i] for i in uniform_indices]
            response_utterances = ["AI: " + all_utterances[i] for i in uniform_indices]
            speaker_response_utterances = [all_utterances[i+1] for i in uniform_indices]

            response_start_end = np.array([(all_starts[i], all_ends[i]) for i in uniform_indices])
            speaker_response_start_end = np.array([(all_starts[i+1], all_ends[i+1]) for i in uniform_indices])
            n_words = np.array([all_n_words[i] for i in uniform_indices])


        prompt = ["You are having a conversation with a User, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits. You are given the dialogue history below, generate a relevant response to the User's last response. Don't generate User's response"]
        dialogue_prompt = [] 


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

            
            #finished populating contexzt
            #add response
            if self.train_setup == 'reward':
                if self.curriculum:
                    context_utterances.append("</s>"+ response_utterances[context_ind] + "</s>" + "</s>"+" User: " + speaker_response_utterances[context_ind] ) #+ "</s>")
                else:
                    context_utterances.append("</s>"+ response_utterances[context_ind]) #+ "</s>")
            if self.train_setup == 'rl':
                pass

            if len(context_utterances) > 20:
                context_utterances = context_utterances[-20:]

            #add prompt in the beginning 
            if self.train_setup == 'rl':
                samples.append(" ".join(prompt + context_utterances))
            else:
                samples.append(" ".join(context_utterances))


        aligned_audio_visual_all = []
        intensity_score = []

        audio_visual_data_df = audio_visual_data_df.rename(columns={'prob_face_anger': 0, 'prob_face_contempt': 1, 'prob_face_disgust':2, 'prob_face_fear':3, 'prob_face_sadness':4, 'prob_face_neutral':5, 'prob_face_happiness':6, 'prob_face_surprise':7})
        
        for i in range(len(uniform_indices)):
            # aligned_audio_visual_subset = audio_visual_data_df[(audio_visual_data_df["seconds"] >= response_start_end[i][0]) & (audio_visual_data_df["seconds"] <= response_start_end[i][1])]
            # aligned_audio_visual_subset = aligned_audio_visual_subset[aligned_audio_visual_subset['user_id'] == speaker_A_name]
            # aligned_audio_visual_subset = aligned_audio_visual_subset[[0,1,2,3,4,5,6,7]]
            # aligned_audio_visual_subset = aligned_audio_visual_subset.dropna()
            # if all_n_words[uniform_indices[i]] <=7 or aligned_audio_visual_subset.empty:
            #     aligned_audio_visual_all.append(5)
            # else:
            #     aligned_audio_visual_all.append(aligned_audio_visual_subset.idxmax(axis=1).value_counts().index.tolist()[0])

            aligned_audio_visual_subset = audio_visual_data_df[(audio_visual_data_df["seconds"] >= response_start_end[i][0]) & (audio_visual_data_df["seconds"] <= response_start_end[i][1])]
            aligned_audio_visual_subset = aligned_audio_visual_subset[aligned_audio_visual_subset['user_id'] == speaker_A_name]
            aligned_audio_visual_subset = aligned_audio_visual_subset[[0,1,2,3,4,5,6,7]]
            aligned_audio_visual_subset = aligned_audio_visual_subset.dropna()
            
            if all_n_words[uniform_indices[i]] <=5: 
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

            if self.reaction_type == 'mean':
                if aligned_audio_visual_subset.empty:
                    intensity_score.append(0) 
                else:
                    intensity = (aligned_audio_visual_subset.idxmax(axis=1).values == 6).mean()
                    intensity_score.append(intensity) 

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

        
        
        
        batch['convo_length'] = len_episode
        batch['indices'] = uniform_indices
        batch['reaction_visual'] = torch.Tensor(aligned_audio_visual_all)
        batch['reaction_visual_intensity'] = intensity_score
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


    
    train_dataset_unflip = CANDOR_LLAMA_K(tokenizer = tokenizer, K = None, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = 'rl', sample_middle = True)
    # train_dataset_unflip = CANDOR_LLAMA_K(tokenizer = tokenizer, K = 1, flip = False, utils = utils, split = 'train', train_size=train_size)


    data_loader = DataLoader(
        dataset=train_dataset_unflip,
        batch_size=batch_size
        # generator=torch.Generator(device='cuda')
    )


    #, interaction_id ='1c0e34d9-5c5c-4e74-9046-216fcd5cbd6d'

    # DataLoader(DS, batch_size = 2, shuffle = False, num_workers = 0)

    for idx, batch in enumerate(tqdm(data_loader)):
        batch = batch
        

        print(idx)
