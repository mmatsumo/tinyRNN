from plotting import *
from plotting_experiments.plotting_dynamics import *

goto_root_dir.run()

plotting_pipeline = [
    'plot_model_perf_data_proportion_for_each_exp',
]

exp_folders = [
     'exp_finetuned_monkeyV',
    #'exp_sim_monkeyW',
]

dot_alpha = 0.9
curve_alpha= 0.9
markersize = 10
curve_markersize = 5
base_color = 'C0'
tran100_color =  'C9'
tran70_color = 'C1'
tran50_color = 'C4'
tran20_color = 'C3'

model_curve_setting = { # for monkey, all MBs are MFs
    'base' : ModelCurve('base', 'base', base_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    '100_pre_trained' : ModelCurve('100_pre_trained', '100_pre_trained', tran100_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    '70_pre_trained' : ModelCurve('70_pre_trained', '70_pre_trained', tran70_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    '50_pre_trained' : ModelCurve('50_pre_trained', '50_pre_trained', tran50_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    '20_pre_trained' : ModelCurve('20_pre_trained', '20_pre_trained', tran20_color, curve_alpha, 'x', curve_markersize, 1, '-'),

}



if 'plot_model_perf_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        for sim_agent_name in ['MB1_seed0']:
            for add_text in [True, False]:
                for perf_type, figname, ylim, yticks in [
                         ('test_loss','loss_all_models', [0.35, 0.65],[0.4, 0.5, 0.6]),
                        # ('test_acc','acc_all_models', [0.65,0.82], [0.7,0.8])
                    ]:
                    plot_all_model_losses_f(exp_folder,
                          finetuned_types=['base','100_pre_trained',
                      '70_pre_trained','50_pre_trained','20_pre_trained'],
                          finetuned_filters={'agent_name': sim_agent_name},
                          xlim=[0.91, 5],
                          xticks=[1, 2, 3, 4, #5, 10, 20
                                  ],
                              ylim=ylim,
                            yticks=yticks,
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=False,
                          title=f'{sim_agent_name}_fit_{exp_folder[8:]}',
                          perf_type=perf_type,
                          figname=f'{sim_agent_name}_{figname}',
                          add_text=add_text,
                            model_curve_setting=model_curve_setting,
                          save_pdf=True,
                          )


if 'plot_model_perf_data_proportion_for_each_exp' in plotting_pipeline:
    exp_folders_dataprop = [
    'exp_finetuned_monkeyV',
    # 'exp_monkeyW_dataprop',
    ]
    for exp_folder in exp_folders_dataprop:
        plot_all_model_losses_dataprop_f(exp_folder,
                              finetuned_types=['base','100_pre_trained',
                      '70_pre_trained','50_pre_trained','20_pre_trained'],
                              xlim=[0, 5000],
                              xticks=[0,1000, 2000,4000,6000,8000],
                              minorticks=False,
                              figsize=(1.5, 1.5),
                              legend=True,
                              title=exp_folder.replace('exp_seg_', ''),
                              figname='loss_all_models_dataprop',
                              model_curve_setting=model_curve_setting,
                          save_pdf=True,
                              )