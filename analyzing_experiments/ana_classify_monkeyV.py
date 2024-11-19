from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    'run_scores_for_each_exp',
]

exp_folders = [
    'exp_classify_monkeyVGRU_2vsMB1',
    'exp_classify_monkeyVSGRU_1vsMB0s',
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
                        'dataset': 'BartoloMonkey',
                        'label': 0,
                        'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70),
                    }]
            }}, suffix_name='_V'
            )
