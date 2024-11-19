import pandas as pd

from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from matplotlib import pyplot as plt
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp', # based on the trainval loss on all train-val data
    # 'run_scores_for_each_exp',
    # 'select_model_perf_for_each_exp', # based on the loss of specific val data; scores should be calculated before this
]

exp_folders = [
    'exp_seg_akamrat49_distill_omni',
]

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'NTSCog',
                                 additional_rnn_keys={'model_identifier_keys': ['include_embedding','embedding_dim_c0','embedding_dim_c1','trainval_size','distill']},
                                 # additional_cog_keys={'model_identifier_keys':['trainval_size','distill']}
                                 has_cog=False,
                                 )

# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder, model_filter={'distill': 'teacher',
                                                 'hidden_dim': 50,
                                                 },
                       overwrite_config={
                           'behav_data_spec': ['animal_name', 'max_segment_length', 'include_embedding','augment'],
                           'augment': True},
                       pointwise_loss=True, # add pointwise loss to the total_score.pkl
                       )

if 'select_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        select_best_models_spec_loss_for_exp(exp_folder,
                                 additional_rnn_keys={'model_identifier_keys': ['trainval_size','distill'],
                                                      'compete_from_keys': ['include_embedding','embedding_dim'],
                                                      },
                                             model_filter={'distill': 'teacher',
                                                           'hidden_dim': 20,
                                                           },
                                 )