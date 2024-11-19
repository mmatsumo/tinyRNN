import numpy as np
import joblib
import matplotlib.pyplot as plt
from plotting_classify import *


for comp, ref_score_labels, ylim_max in [
    # ('SGRU_1vsMB0s', ['GRU (d=1)', 'MB (d=1)'], 5),
    # ('GRU_2vsMXs', ['GRU (d=2)', 'MB (d=2)'], 4),
    # ('SGRU_1vsMXs', ['GRU (d=1)', 'MB (d=2)'], 4),
    ('GRU_1vsMB0s_ntrials1200', ['GRU (d=1)', 'MB (d=1)'], 5),
    ('GRU_2vsMXs_ntrials1200', ['GRU (d=2)', 'MB (d=2)'], 4),
    ('GRU_1vsMXs_ntrials1200', ['GRU (d=1)', 'MB (d=2)'], 4),
    ('MFsvsMB0s_ntrials1200', ['MF (d=1)', 'MB (d=1)'], 5),
    ]:
    parent_dir = rf'exp_classify_millerrat55{comp}\rnn_type-GRU.hidden_dim-10.l1_weight-1e-05'
    # for all subfolders in parent_dir, plot the model evidence
    for sub_dir in os.listdir(ANA_SAVE_PATH / parent_dir):
        if not os.path.isdir(ANA_SAVE_PATH / parent_dir / sub_dir):
            continue
        plot_evidence_for_all_scores(rf'{parent_dir}\{sub_dir}\total_scores.pkl',
                                     rf'{parent_dir}\{sub_dir}\total_scores_m55.pkl',
                                     ref_score_labels=ref_score_labels, sub_score_labels=['Rat 55'],
                                     ref_plot_every_n=20, exp_fig_path=f'exp_classify_millerrat55',
                                     fig_name=f'{sub_dir}_model_evidence_{comp}',xlim_max=None, ylim_max=None)
