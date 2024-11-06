import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer, BartForSequenceClassification, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, AutoModelForSeq2SeqLMWithValueHead
from trl.core import respond_to_batch

from data_loader_reward_single import CANDOR_LLAMA_K
import pdb
from tqdm import tqdm
from transformers import pipeline
from trl.core import LengthSampler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from transformers import LongformerModel, AutoModelForCausalLM, Adafactor
import torch.nn.functional as F
import utils
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import os

from options import parser
import copy
import json
import numpy as np

args = parser.parse_args()

model_name = args.model_name
print("Model Chosen: {}".format(model_name))


import logging
log_path = "./logs/{}_{}.log".format(model_name, args.exp_name, )
logging.basicConfig(filename=log_path,level=logging.DEBUG)
logging.error('staart')

batch_size=int(args.batch_size)
learning_rate=1.4e-5
config = PPOConfig(
    model_name=model_name,
    # learning_rate=1.4e-10,
    learning_rate=learning_rate,
    mini_batch_size=6,
    batch_size=batch_size,
    gradient_accumulation_steps=4,
    log_with='wandb',
    target_kl=1,
    ppo_epochs=1,
    seed = 1,
    vf_coef= 0.1,
    early_stopping=True,
    init_kl_coef=0.05,
    adap_kl_ctrl=True,
    use_score_scaling = True,
    use_score_norm= True,
    # cliprange = 1000,
    # cliprange_value= 1000
    
)


if not args.no_accel:
    config = PPOConfig(
    model_name=model_name,
    # learning_rate=1.4e-10,
    learning_rate=learning_rate,
    mini_batch_size=4,
    batch_size=batch_size,
    gradient_accumulation_steps=2,
    log_with='wandb',
    early_stopping=False,
    target_kl=1,
    ppo_epochs=1,
    # early_stopping=False,
    init_kl_coef=0.15,
    adap_kl_ctrl=True,
    use_score_scaling = True,
    use_score_norm= True
)




if model_name == 'llama2':

    
    lm_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    if getattr(lm_tokenizer, "pad_token", None) is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    lm_tokenizer.padding_side='left'
    lm_tokenizer.truncation_side='left'


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    if args.no_accel:
        lora_config = LoraConfig(
                    r=24,
                    lora_alpha=48,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
        llama2_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto") #, load_in_8bit=True,)

        # this line is very important
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model = get_peft_model(llama2_model, lora_config)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model) #, load_in_8bit=True)

    else:

        lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
        llama2_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map={"": Accelerator().local_process_index})

        # this line is very important
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        llama2_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        peft_model = get_peft_model(llama2_model, lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            peft_model,
            load_in_8bit=True,
            device_map={"": Accelerator().local_process_index},
        )


    print_trainable_parameters(model)

    pdb.set_trace()
    


if model_name == 'llama':
    hf_model_name = "hf-internal-testing/llama-tokenizer"
    lm_tokenizer = LlamaTokenizer.from_pretrained("./llama_hf", legacy=False)
    if getattr(lm_tokenizer, "pad_token", None) is None:
        lm_tokenizer.pad_token = lm_tokenizer.unk_token #lm_tokenizer.eos_token #add_special_tokens({'pad_token': '[PAD]'})

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        # pretrained_model_name_or_path= "./finetuned/llama_finetune_llama_jul18",
        pretrained_model_name_or_path= "./llama_hf",
        # pretrained_model_name_or_path= "./llama_hf",
        device_map="auto",#{"": current_device},
        peft_config=lora_config,
        )
    
    ref_model = create_reference_model(model)


max_length = 256


train_size = 600 #600
DATA_PATH = './candor'
interaction_path_list = os.listdir(DATA_PATH)
nan_list = utils.nan_list

if args.train:
    all_datasets = []
    for path in tqdm(interaction_path_list[300:300+train_size]):
        if path not in nan_list:
            transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
            survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
            audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
        
            train_dataset = CANDOR_LLAMA_K(transcript_path, survey_path, audio_visual, tokenizer = lm_tokenizer, K = 96, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = 'rl', sample_middle = True)
            all_datasets.append(train_dataset)

    full_train_set = torch.utils.data.ConcatDataset(all_datasets)

    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

    ppo_trainer = PPOTrainer(config, model, ref_model = None, tokenizer = lm_tokenizer, dataset=full_train_set)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 1 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug


if args.val:
    all_datasets = []
    for path in tqdm(interaction_path_list[1000:1000+10]): #fix this
        if path not in nan_list:
            transcript_path = os.path.join(DATA_PATH, path, 'transcription','transcript_cliffhanger.csv') #cliffhanger
            survey_path =  os.path.join(DATA_PATH, path, 'survey.csv') 
            audio_visual = os.path.join(DATA_PATH, path, 'audio_video_features.csv')
        
            train_dataset = CANDOR_LLAMA_K(transcript_path, survey_path, audio_visual, tokenizer = lm_tokenizer, K = 15, flip = False, utils = utils, split = 'train', train_size=train_size, train_setup = 'rl', sample_middle = True)
            all_datasets.append(train_dataset)

    val_train_set = torch.utils.data.ConcatDataset(all_datasets)

    val_ppo_trainer = PPOTrainer(config, model, ref_model, lm_tokenizer, dataset=val_train_set)

    device = val_ppo_trainer.accelerator.device
    if val_ppo_trainer.accelerator.num_processes == 1:
        device = 1 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug





#pretrained sentiment classifier 
if 'redistributed_reward' in args.reward:

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
        


    if args.rf_model == 'GELI':
        curr_path = "./visual_feedback_baselines_convo_overall_affect_baseline_RRD_K_160"
        PATH = os.path.join(curr_path, "reward1_" + "eager-glade-5")
        reward_function1 = torch.load(PATH).cuda()
        reward_function1.eval()



    #this is best performing reward 
    gt_saved_path = "./visual_feedback_baselines_convo_overall_affect_baseline_RRD_K_160"
    PATH = os.path.join(gt_saved_path, "reward1_" + "eager-glade-5")
    gt_reward_function = torch.load(PATH).cuda()
    gt_reward_function.eval()



#START HERE
output_min_length = 32
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

generation_kwargs = {
    # "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "eos_token_id": 100_000, #lm_tokenizer.eos_token_id,
    "pad_token_id": lm_tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    # "max_new_tokens": 128, # specify how many tokens you want to generate at most
}

val_generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 50, # no top-k sampling
    "top_p": 0.95, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "eos_token_id": lm_tokenizer.eos_token_id,
    "pad_token_id": lm_tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": 128, # specify how many tokens you want to generate at most
    "num_beams":3, 
    "no_repeat_ngram_size":2,
    "early_stopping": True
}




best_reward = -10e5
num_epochs = 1

if args.train:
    for epoch_train in range(num_epochs):
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            if step % 10 == 0 and step > 1:
                path_to_save = "./rlmf_weights/{}_{}_{}".format(args.exp_name, args.model_name,step)
                if not os.path.exists(path_to_save):
                    # Create a new directory because it does not exist
                    os.makedirs(path_to_save)
                model.save_pretrained(path_to_save, from_pt=True) 
                lm_tokenizer.save_pretrained(path_to_save)

            if step % 200 == 0 and step > 1:
                break 

        
            query_tensor = batch['A_sentence_llama']
            query_tensors = list(batch['A_sentence_llama']['input_ids'].cuda().squeeze())
            
            for k,v in query_tensor.items(): query_tensor[k] = v.squeeze(1).cuda()
            query_text = lm_tokenizer.batch_decode(query_tensor['input_ids'], skip_special_tokens  = True)
            

            ground_truth_answers = batch['B_sentence_llama'] 
            for k,v in ground_truth_answers.items(): ground_truth_answers[k] = v.squeeze(1).cuda()
            ground_truth_text = lm_tokenizer.batch_decode(ground_truth_answers['input_ids'], skip_special_tokens  = True)
            
            ppo_input = lm_tokenizer(query_text)['input_ids']
            ppo_input = [torch.Tensor(ppo_tokens).to(model.pretrained_model.device).long() for ppo_tokens in ppo_input]

            response_tensors = ppo_trainer.generate(ppo_input, return_prompt=False, length_sampler = output_length_sampler, **generation_kwargs)
            generated_text = lm_tokenizer.batch_decode(response_tensors,  skip_special_tokens  = False)

            query_generated_text = []

                
            for i in range(batch_size): 

                print("\nOriginal_RESPONSE\n")
                print(generated_text[i])

                eos = generated_text[i].find("</s>")
                if eos != -1:
                    generated_text[i] = generated_text[i][:eos]

                user_eos = generated_text[i].find("User:")
                if user_eos != -1:
                    generated_text[i] = generated_text[i][:user_eos]

                manual_eos = generated_text[i].find("AI:")
                if manual_eos != -1:
                    generated_text[i] = generated_text[i][manual_eos:]

                else:
                    pass
            

                context = query_text[i][query_text[i].find("AI:"):query_text[i].rfind(" AI:")]

                
                response = generated_text[i]

                query_generated_text.append(context + "</s></s>"+ " AI: "+ response)
                print("\nCONTEXT\n")
                logging.error("\nCONTEXT\n")
                print(query_text[i])
                logging.error(query_text[i])


                print("\nRESPONSE\n")
                logging.error("\nRESPONSE\n")
                print(generated_text[i])
                logging.error(generated_text[i])
                    
            encoded_tokens = longformer_tokenizer(query_generated_text, return_tensors="pt", padding = 'max_length',truncation = True, max_length = 1024).to(model.pretrained_model.device)
            

            with torch.no_grad():
                part = lang_model(**encoded_tokens)["encoder_last_hidden_state"] 
                eos_mask = encoded_tokens["input_ids"].squeeze().eq(longformer_tokenizer.eos_token_id).to(model.pretrained_model.device)
                sentence_representation = part[eos_mask, :].view(part.size(0), -1, part.size(-1))
                sentence_representation = sentence_representation[:,-1,:]

                preds = 1000*reward_function1(sentence_representation) #try 100 
                # preds = (preds - preds.mean()) / (preds.std()+ 1e-5)
                rewards = list(preds.cuda())

                gt_rewards = gt_reward_function(sentence_representation)
                gt_rewards = list(gt_rewards.cuda())
        

            ppo_batch = {}
            ppo_batch["query"]= query_text
            ppo_batch["response"] = generated_text



            list_query_tensor = ppo_input #lm_tokenizer(query_text)['input_ids']
            list_response_tensor = list(response_tensors)#lm_tokenizer(generated_text)['input_ids']
            
            if args.ppo_off:
                pass
            else:
                ppo_batch["ref_response"] = ground_truth_text
                ppo_batch["ref_rewards"] = gt_rewards
                stats = ppo_trainer.step(list_query_tensor, list_response_tensor, rewards)

                print(torch.stack(gt_rewards).mean() * 1000)

                ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

        
