import os, sys, pdb

import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(description='Parameters for Reward Training')



parser.add_argument('--reward_class',  choices=('you_are_funny', 'you_are_intelligent', 'our_thoughts_synced_up_sr1', 'you_are_humble' , 'end_affect', 'overall_affect', 'overall_memory_rating', 'how_enjoyable','i_like_you', 'in_common', 'conversationalist', 'good_for_advice', 'you_are_intelligent', 'you_are_quickwitted', 'you_are_competent', 'you_are_kind', 'you_are_friendly', 'you_are_warm'))
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--model', required = True, choices=('longformer', 'longT5', 'convo', 'LLAMA2'))
parser.add_argument('--batch_size', required = True)
parser.add_argument('--K', required = True)
parser.add_argument('--train_size', required = True)
parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--hard_shrink', action='store_true')
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--bar_chart', action='store_true')
parser.add_argument('--small_model', action='store_true')
parser.add_argument('--small_model_shrink', action='store_true')
parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--curriculum_exposure', action='store_true')
parser.add_argument('--affect_path')
parser.add_argument('--lang_model_path')
parser.add_argument('--redist_type')
parser.add_argument('--reaction_type', default = 'mode', choices=('mode', 'mean'))
