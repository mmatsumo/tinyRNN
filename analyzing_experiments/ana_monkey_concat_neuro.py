from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
]

exp_folders = [
    'exp_monkeyV_concat_neuro',
]

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, '',
                                 additional_rnn_keys={'model_identifier_keys': ['pca_num']},
                                 has_cog=False,
                                 )