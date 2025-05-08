from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *

analyzing_pipeline = [
    #'analyze_model_perf_for_each_exp',
     #'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
    'analyze_finetuned'
]

exp_folders = [
     'exp_finetuned_monkeyV',
    #'exp_sim_monkeyW',
]

model_based = ['base','100_pre_trained',
                      '70_pre_trained','50_pre_trained','20_pre_trained']

## perf
if 'analyze_finetuned' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp_finetuned(exp_folder, 'PRLCog',
                             additional_rnn_keys={'model_identifier_keys': ['trainval_percent']},
                             #additional_cog_keys={'model_identifier_keys': ['agent_name']},
                                 has_rnn=True,
                                 include_acc=True,
                                 dataset_loading_every_time=True,
                                 has_cog=False, model_based=model_based) # multiple datasets in one folder; must be re-loaded every time


