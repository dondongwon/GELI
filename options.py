import os, sys, pdb

import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(description='Parameters for RLMF')



parser.add_argument('--model_name',  choices=('llama2','llama', 'cosmo', 'blenderbot-1b','blenderbot-400m', 'dialogpt'))
parser.add_argument('--reward', default = 'redistributed_reward')
parser.add_argument('--exp_name', required = True)
parser.add_argument('--generate', action='store_true')
parser.add_argument('--peft', action='store_true')
parser.add_argument('--ppo_off', action='store_true')

parser.add_argument('--reward_class',  choices=('overall_affect'))
parser.add_argument('--rf_model',  choices=('GELI', 'GE', 'LI', 'sentiment', 'base', 'VFRRD_funny', 'VFRRD_intelligent', 'VFRRD_humble', 'VFRRD_sync'))
parser.add_argument('--train', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--batch_size', required = True )
parser.add_argument('--no_accel', action='store_true')
