"""
Bahrami Human, four-armed bandit.
"""
import pandas as pd

from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from matplotlib import pyplot as plt
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    'compile_perf_for_all_exps',
    # 'run_scores_for_each_exp',
    # 'run_dynamical_regression_for_each_exp',
]

# cross-subject
exp_folders = [
    'exp_Bahrami',
    'exp_Bahrami_aug2',
    'exp_Bahrami_cog',
]
compile_folder = 'exp_Bahrami_sub_avg'
additional_rnn_keys={'model_identifier_keys': ['distill','augment',]} # augment can = 0, 1, 2

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'MABCog',
                                 additional_rnn_keys={'model_identifier_keys': ['block','distill','augment']},
                                 additional_cog_keys={'model_identifier_keys': ['block','distill']},
                                 rnn_sort_keys=['block', 'hidden_dim'],
                                 cog_sort_keys=['block'],
                                 has_rnn=False, #False, #
                                 has_cog=True,
                                 cog_hidden_dim={
                                     'MF': 4,
                                     'MFD': 4,
                                     'MFDp': 8,
                                     'MFL': 4,
                                     'MFLabs': 4,
                                 },
                                 return_dim_est=True,
                                 include_acc_filter=lambda row: 'hidden_dim' not in row or row['hidden_dim'] <=20,
                                 check_missing=False,
                                 )


if 'compile_perf_for_all_exps' in analyzing_pipeline:
    # compile_perf_for_exps(exp_folders, 'exp_Bahrami_sub_avg',
    #                       additional_rnn_keys={'model_identifier_keys': ['distill','augment',]},
    #                       rnn_filter={'distill': 'student'},
    #                       lambda_filter=lambda dt, dt_ref: dt['block'].isin(
    #                           pd.unique(dt_ref[dt_ref['rnn_type'] == 'MLR']['block'])),
    #                       lambda_filter_name='_ref_MLRblock',
    #                       )
    # compile_perf_for_exps(exp_folders, compile_folder,
    #                       additional_rnn_keys=additional_rnn_keys,
    #                       additional_cog_keys={'model_identifier_keys': ['distill', ]},
    #                       # rnn_filter={'distill': 'student', 'rnn_type': 'GRU', 'augment': 2},
    #                       # lambda_filter=lambda dt, dt_ref: dt['block'].isin(
    #                       #     pd.unique(dt_ref[dt_ref['rnn_type'] == 'GRU']['block'])),
    #                       # lambda_filter_name='_noMLR',
    #                       has_rnn=True,
    #                       has_cog=True,
    #                       )
    compile_perf_for_exps(exp_folders, compile_folder,
                          additional_rnn_keys=additional_rnn_keys,
                          additional_cog_keys={'model_identifier_keys': ['distill', ]},
                          rnn_filter={'distill': 'student', 'rnn_type': 'GRU', 'augment': 2},
                          cog_filter={'block': -1},
                          rnn_lambda_filter=lambda dt, dt_ref: dt['block'].isin(
                              pd.unique(dt_ref[dt_ref['rnn_type'] == 'GRU']['block'])),
                          # cog_lambda_filter is all true for each row
                          cog_lambda_filter=lambda dt, dt_ref: pd.Series([True] * len(dt)),
                          lambda_filter_name='_noMLR_cog_-1',
                          has_rnn=True,
                          has_cog=True,
                          )
# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        # # generate scores for KD from teacher RNN
        # # for cog models and KD student models, 'augment': 2, include_data='test_augment'
        # run_scores_exp(exp_folder, demask=False, pointwise_loss=True,
        #                model_filter={'distill': 'student', 'rnn_type': 'GRU'},
        #                overwrite_config={
        #                    'behav_data_spec': {'augment': 2, # second-level augmentation
        #                                        }
        #                },
        #                include_data='test_augment',
        #                has_cog=True,
        #                )
        # for tch models, 'augment': 2, include_data='all'
        run_scores_exp(exp_folder, demask=False, pointwise_loss=True,
                       model_filter={'distill': 'none', 'rnn_type': 'GRU'},
                       overwrite_config={
                           'behav_data_spec': {'augment': 2
                                               },
                           'device': 'cpu',
                       },
                       include_data='all',
                       has_cog=False,
                       )

if 'run_dynamical_regression_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_dynamical_regression_exp(exp_folder, func='bahrami',
                                     filters = lambda row:
                                     ('cog_type' in row and row['cog_type'] in ['MF', 'MFD','MFL','MFLabs']) or \
                                    ('rnn_type' in row and row['rnn_type'] == 'GRU' and 'augment' in row and row['augment']==2 and not row['readout_FC']),
                       )