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
    # 'merge_dynamical_regression_pdf',
]
dynamics_plot_pipeline = [
    ## global options
    # 'legend', # show legend; only for 2d_logit_change and show_curve

    ## logit and pr analysis
    # '2d_logit_change', # logit vs logit change

    ## other analysis
    # '2d_vector_field',
    # '3d_vector_field',
    ]

exp_folders = [
    'exp_Gillan',
    'exp_Gillan1',

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
    'MFs': ModelCurve('MF_corr', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MF': ModelCurve('MF', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MFsr': ModelCurve('MFsr', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MBs': ModelCurve('MB_corr', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MB': ModelCurve('MB', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MXs': ModelCurve('Mix_corr', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MX': ModelCurve('Mix', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),

}


plot_perf_exp_folders = exp_folders if 'plot_model_perf_for_each_sub' in plotting_pipeline else []
plot_perf_exp_folders += ['exp_Gillan_sub_avg'] if 'plot_perf_for_all_subs' in plotting_pipeline else []
for exp_folder in plot_perf_exp_folders:
    for add_text in [True, False]:
        for load_file_suffix, figname in [
            #('_noMLR', 'loss_all_models'),
            ('_noMLR_cog_-1', 'loss_all_models_cog_cross_sub'),
        ]:
            plot_all_model_losses(exp_folder,
                          rnn_types=['GRU',
                                     ],
                          cog_types=[
                                'MFs', 'MBs', 'MXs',
                                'MF', 'MB', 'MX',
                              # 'MFsr',
                          ],
                          rnn_filters={'readout_FC': True},
                          xlim=[0.91, 22],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          # ylim=[0.45, 0.65],
                          # yticks=[0.45, 0.55, 0.65],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          title=exp_folder[4:],
                          load_file_suffix=load_file_suffix,
                          figname=figname,
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          save_pdf=save_pdf,
                          )

if 'plot_dim_for_all_exps' in plotting_pipeline:
    plot_dim_distribution('exp_Gillan_sub_avg', suffix='_noMLR',
                          bins=[1, 2, 3, 4, 5, 6, 7, 8],
                          xticks=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                          xticks_label=[1, 2, 3, 4, 5, 6, '>6'],
                          yticklabel_every=100,
                          )


# dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf,func='gillan',
                                     )

if 'plot_dynamical_regression_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plt_dynamical_regression_gillan(exp_folder, save_pdf=True,
                                        plot_regression=False,
                                        # model_filters={'rnn_type':'GRU', 'hidden_dim': 6}
                                        )

if 'plot_dynamical_regression_distribution' in plotting_pipeline:
    # plt_dynamical_regression_distribution_gillan('exp_Gillan1')
    for cog_type, percent in [
        # ('MFs', 5),
        # ('MBs', 5),
        # ('MXs', 5),
        ('MFsr', 7), # for visualization purpose
         ]:
        plt_dynamical_regression_violin_gillan('exp_Gillan', model_filters={'cog_type': cog_type}, percent=percent)
    # plt_dynamical_regression_violin_gillan('exp_Gillan1', model_filters={'rnn_type': 'GRU', 'hidden_dim': 3},percent=5)



if 'merge_dynamical_regression_pdf' in plotting_pipeline:
    from plotting_utils import concatenate_pdfs
    for subblock in range(1961):
        for target in ['x1_change', 'x2_change', 'x3_change']:
            pdf_files = [FIG_PATH / folder / 'dynamical_regression' / f'subblock-{subblock}' / f'{m}_{target}_{x}.pdf'
             for folder, m in zip(['exp_Gillan', 'exp_Gillan', 'exp_Gillan1', 'exp_Gillan1'],['MFs','MBs','GRU','GRU6'])
             for x in ['b0', 'b1', 'b2', 'b3','score','perf']
             ]
            fig_exp_path = FIG_PATH / 'exp_Gillan' / 'dynamical_regression' / f'subblock-{subblock}'
            concatenate_pdfs(pdf_files, fig_exp_path / f'{target}_merged.pdf', n_cols=6, n_rows=4)
        print(f'subblock {subblock} done')