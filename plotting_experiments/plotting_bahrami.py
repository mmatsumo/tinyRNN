from plotting import *
from plotting_dynamics import *

save_pdf = True
plotting_pipeline = [
    'plot_perf_for_all_subs',
    # 'plot_dim_for_all_exps',
    # 'plot_model_num_par',
    # 'plot_ev_for_each_exp',
    # 'plot_dynamics_for_each_exp',
    # 'plot_dynamical_regression_for_each_exp',
    # 'plot_dynamical_regression_distribution',
    # 'plot_1d_for_each_exp',
]
dynamics_plot_pipeline = [
    ## global options
    # 'relative_action', # note this option will change all results for 2d_logit_ and 2d_pr_ to relative action
    # 'hist', # note this option will change all results for 2d_logit_ and 2d_pr_ to histogram
    # 'show_curve', # show curve instead of dots; only for 1d models
    # 'legend', # show legend; only for 2d_logit_change and show_curve

    ## logit and pr analysis
    '2d_logit_change', # logit vs logit change
    # '2d_logit_next', # logit vs logit next
    # '2d_logit_nextpr', # logit vs pr next
    # '2d_pr_nextpr', # pr vs pr next
    # '2d_pr_change', # pr vs pr change
    # '2d_logit_nextpr_ci', # logit vs pr next with confidence interval; only for 1d models
    # '2d_logit_nextpr_ci_log_odds_ratio', # logit vs pr next, with log odds ratio calculated for confidence interval; only for 1d models

    ## other analysis
    # '2d_vector_field',
    ]

exp_folders = [
    'exp_Bahrami',
    'exp_Bahrami_aug2',
]
goto_root_dir.run()

dot_alpha = 0.9
curve_alpha= 0.9
markersize = 10
curve_markersize = 5
GRU_color = 'C0'
SGRU_color =  'C5'
LS_color = 'C1'
MF_color = 'C4'
MB_color = 'C3'
PNR_color = 'C2'
model_curve_setting = {
    'GRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),

    # MF: C4, LS: C5, MB/MFMB: C3, RC: C4
    'MF': ModelCurve('MF', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MFD': ModelCurve('MF_decay', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MFDp': ModelCurve('MF_decay_pers', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MFL': ModelCurve('MFL', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MFLabs': ModelCurve('MFLabs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
}


plot_perf_exp_folders = exp_folders if 'plot_model_perf_for_each_exp' in plotting_pipeline else []
plot_perf_exp_folders += ['exp_Bahrami_sub_avg'] if 'plot_perf_for_all_subs' in plotting_pipeline else []
for exp_folder in plot_perf_exp_folders:
    for add_text in [True, False]:
        for load_file_suffix, figname in [
            #('_noMLR', 'loss_all_models'),
            ('_noMLR_cog_-1', 'loss_all_models_cog_cross_sub'),
        ]:
            plot_all_model_losses(exp_folder,
                          rnn_types=['GRU',
                                     ],
                          cog_types=['MF',
                                        'MFD',
                                        'MFDp',
                                        # 'MFL',
                                        # 'MFLabs',
                                     ],
                          rnn_filters={'readout_FC': True},
                          xlim=[0.91, 22],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          # ylim=[0.45, 0.65],
                          # yticks=[0.45, 0.55, 0.65],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=False,
                          title=exp_folder[4:],
                          load_file_suffix=load_file_suffix,
                          figname=figname,
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          save_pdf=save_pdf,
                          )

if 'plot_ev_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf, plot_ev=True)

if 'plot_dim_for_all_exps' in plotting_pipeline:
    plot_dim_distribution('exp_Bahrami_sub_avg', suffix='_noMLR',
                          bins=[1, 2, 3, 4, 5, 6, 7],
                          xticks=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                          xticks_label=[1, 2, 3, 4, 5, '>5'],
                          yticklabel_every=100,
                          )

# dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf,func='bahrami',rnn_filters={'hidden_dim': 20})

if 'plot_1d_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        # plot_1d_logit_coef(exp_folder)
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='intercept')
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='slope')


if 'plot_dynamical_regression_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plt_dynamical_regression_bahrami(exp_folder, save_pdf=True,
                                         plot_regression=True, plot_perf=True,
                                        model_filters={'hidden_dim': 4}
                                        )


if 'plot_dynamical_regression_distribution' in plotting_pipeline:
    for cog_type, percent in [
        # ('MF', 5),
        # ('MFD', 5),
        # ('MFL', 5),
        ('MFLabs', 7),
    ]:
        plt_dynamical_regression_violin_bahrami('exp_Bahrami', model_filters={'cog_type': cog_type}, percent=percent)
    # plt_dynamical_regression_violin_bahrami('exp_Bahrami_aug2', model_filters={'rnn_type': 'GRU'}, percent=5)