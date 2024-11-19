import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
from plotting_experiments.plotting import *
import joblib
from utils import goto_root_dir
goto_root_dir.run()
analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    'compile_perf',
]

exp_folders = [
    'exp_sim_millerrat55_splittrial',
]


## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                             additional_rnn_keys={'model_identifier_keys': ['agent_name']},
                             additional_cog_keys={'model_identifier_keys': ['agent_name']})

if 'compile_perf' in analyzing_pipeline:
    ana_exp_path = ANA_SAVE_PATH / exp_folders[0]
    cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')
    rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
    rnn_perf = rnn_perf[rnn_perf['hidden_dim'] == 10]
    plot_label = True
    fig, ax = plot_start()
    for agent_name in ['MB0s_seed0', 'LS0_seed0', 'MB0_seed0', 'MB1_seed0','MXs_seed0']:
        this_cog_perf = cog_perf[(cog_perf['agent_name'] == agent_name)].iloc[0]
        this_rnn_perf = rnn_perf[(rnn_perf['agent_name'] == agent_name)].iloc[0]
        for loss in ['train_loss', 'val_loss', 'test_loss']:
            label = None
            color = {'train_loss': 'C0', 'val_loss': 'C1', 'test_loss': 'C2'}[loss]
            if plot_label:
                label = {'train_loss': 'training', 'val_loss': 'validation', 'test_loss': 'test'}[loss]
            plt.scatter(this_cog_perf[loss], this_rnn_perf[loss], label=label, s=5, color=color, facecolors='none')
        plot_label = False
    plt.plot([0.35, 0.51], [0.35, 0.51], 'k--', alpha=0.3, linewidth=1)
    plt.xticks([0.4, 0.5])
    plt.yticks([0.4, 0.5])
    plt.xlabel('Ground-truth model loss')
    plt.ylabel('RNN loss')
    plt.legend(loc='upper left')
    plt.show()
    os.makedirs(FIG_PATH / exp_folders[0], exist_ok=True)
    plt.savefig(FIG_PATH / exp_folders[0] / 'cog_vs_rnn_loss.pdf', bbox_inches='tight')
