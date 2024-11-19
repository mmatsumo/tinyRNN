from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    'run_scores_for_each_exp',
]

exp_folders = [
    # 'exp_classify_millerrat55GRU_2vsMXs',
    # 'exp_classify_millerrat55SGRU_1vsMB0s',
    # 'exp_classify_millerrat55SGRU_1vsMXs',
    # 'exp_classify_millerrat55GRU_1vsMB0s',
    # 'exp_classify_millerrat55GRU_2vsMB1',
    # 'exp_classify_millerrat55MB0vsMB1',
    # 'exp_classify_millerrat55GRU_1vsMB0s_ntrials1200',
    # 'exp_classify_millerrat55GRU_1vsMXs_ntrials1200',
    # 'exp_classify_millerrat55GRU_2vsMXs_ntrials1200',
    'exp_classify_millerrat55MFsvsMB0s_ntrials1200',
]


## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, '',
                             additional_rnn_keys={'model_identifier_keys': []},
                                 has_cog=False,)


# dynamics
if 'run_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder, pointwise_loss=True,has_cog=False,)
        run_scores_exp(exp_folder, pointwise_loss=True,has_cog=False, overwrite_config={
            'behav_data_spec':{
                'misc':[
                    {
                        'dataset': 'MillerRat',
                        'label': 0,
                        'animal_name': 'm55',
                        #'max_segment_length': 500,
                    }]
            }}, suffix_name='_m55'
            )
