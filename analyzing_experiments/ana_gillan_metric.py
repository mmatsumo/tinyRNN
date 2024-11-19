import pandas as pd

# import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import numpy as np

# rnn
# coef_path = r'D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\analysis\exp_Gillan1\dynamical_regression_summary.csv'
# coef_df = pd.read_csv(coef_path)
# coef_df = coef_df[coef_df['hidden_dim'] == 3]
# ttm = {} # trial_type_mapping
# for col in coef_df.columns:
#     if '_name' in col:
#         type_index = col.replace('_name', '')
#         type_name = coef_df.iloc[0][col]
#         ttm[type_name] = type_index
# regressor_names = [col for col in coef_df.columns if 'change' in col]
#
# # pca
# X = coef_df[regressor_names].values
# from sklearn.decomposition import PCA
# pca = PCA(n_components=len(regressor_names))
# pca.fit(X)
# print(pca.explained_variance_ratio_)
# X_pca = pca.transform(X)
# coef_df['pca1'] = X_pca[:,0]
# coef_df['pca2'] = X_pca[:,1]

from scipy.stats import zscore
from matplotlib import pyplot as plt
# cog model


y_name_mapping = {
    'lsas_total': 'Social Anxiety',
    'bis_total': 'Impulsivity',
    'sds_total': 'Depression',
    'scz_total': 'Schizotypy',
    'aes_total': 'Apathy',
    'stai_total': 'Trait Anxiety',
    'eat_total': 'Eating Disorders',
    'audit_total': 'Alcohol Addiction',
    'oci_total': 'OCD'
}
for cog_idx, cog_type in enumerate(['MFs', 'MBs', 'MXs',
                                    'MFsr']):
    coef_path = rf'D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\analysis\exp_Gillan\{cog_type}_param_summary.csv'
    coef_df = pd.read_csv(coef_path)
    if cog_type == 'MXs':
        coef_df['beta1_w'] = coef_df.apply(lambda row: row['beta1'] * row['w'], axis=1)
    regressor_names = list(coef_df.columns)
    # zscore
    for col in regressor_names:
        coef_df[col] = zscore(coef_df[col])

    print(cog_type, regressor_names)
    # add block index = row index
    coef_df['block'] = coef_df.index

    assert len(coef_df) == 1961 # 1961 subjects

    study1_subs = coef_df['block'] < 548
    study2_subs = coef_df['block'] >= 548
    assert sum(study1_subs) == 548 # 548 subjects in study 1
    assert sum(study2_subs) == 1413 # 1413 subjects in study 2

    study1_path = r'D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\Gillandata\Experiment 1\self_report_study1.csv'
    study2_path = r'D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\Gillandata\Experiment 2\self_report_study2.csv'
    study1_df = pd.read_csv(study1_path)
    study2_df = pd.read_csv(study2_path)
    assert len(study1_df) == 548
    assert len(study2_df) == 1413
    y_list_study1 = ['oci_total', 'stai_total', 'sds_total']
    y_list_study2 = ['lsas_total', 'bis_total', 'sds_total', 'scz_total', 'aes_total', 'stai_total', 'eat_total', 'audit_total', 'oci_total']

    # suffix = '_x1_change_b0'
    # R1_common = (coef_df[f'{ttm["A1S1B1R1"]}{suffix}'] + coef_df[f'{ttm["A1S1B2R1"]}{suffix}'] - coef_df[f'{ttm["A2S2C1R1"]}{suffix}'] - coef_df[f'{ttm["A2S2C2R1"]}{suffix}']) / 4
    # R0_common = (coef_df[f'{ttm["A1S1B1R0"]}{suffix}'] + coef_df[f'{ttm["A1S1B2R0"]}{suffix}'] - coef_df[f'{ttm["A2S2C1R0"]}{suffix}'] - coef_df[f'{ttm["A2S2C2R0"]}{suffix}']) / 4
    # R1_rare = (coef_df[f'{ttm["A1S2C1R1"]}{suffix}'] + coef_df[f'{ttm["A1S2C2R1"]}{suffix}'] - coef_df[f'{ttm["A2S1B1R1"]}{suffix}'] - coef_df[f'{ttm["A2S1B2R1"]}{suffix}']) / 4
    # R0_rare = (coef_df[f'{ttm["A1S2C1R0"]}{suffix}'] + coef_df[f'{ttm["A1S2C2R0"]}{suffix}'] - coef_df[f'{ttm["A2S1B1R0"]}{suffix}'] - coef_df[f'{ttm["A2S1B2R0"]}{suffix}']) / 4
    # print(R1_common.mean(), R0_common.mean(), R1_rare.mean(), R0_rare.mean())
    # interaction = (R1_common - R0_common) - (R1_rare - R0_rare)


    # interaction_study1 = interaction[study1_subs]
    # interaction_study2 = interaction[study2_subs]
    # for col in ['interaction', 'R1_common', 'R0_common', 'R1_rare', 'R0_rare']:
    #     study1_df[col] = eval(col)[study1_subs]
    #     study2_df[col] = eval(col)[study2_subs]

    # concat
    study1_df = pd.concat([study1_df, coef_df[study1_subs][regressor_names].reset_index(drop=True)], axis=1)
    study2_df = pd.concat([study2_df, coef_df[study2_subs][regressor_names].reset_index(drop=True)], axis=1)

    # z-score for study1: iq, age, sds_total (depression), stai_total (anxiety), oci_total (ocd)
    for col in ['iq', 'age'] + y_list_study1:
        study1_df[col] = zscore(study1_df[col])
    for col in ['iq', 'age'] + y_list_study2:
        study2_df[col] = zscore(study2_df[col])

    # for col in ['interaction', 'R1_common', 'R0_common', 'R1_rare', 'R0_rare']:
    #     formula = f'{col} ~ iq + age + gender + oci_total'
    #     print(formula)
    #     model = smf.ols(formula, data=study1_df).fit()
    #     print(model.summary())

    df = study2_df
    y_list = y_list_study2

    for y_name in y_list:
        formula = f'{y_name} ~ iq + age + gender'
        model = smf.ols(formula, data=study2_df).fit()
        print(formula, model.rsquared)
        df[y_name] = model.resid
        df[y_name] = zscore(df[y_name])

    corr_list = []
    for y_name in y_list:
        # # cross validation not working well for regression here
        # kf = KFold(n_splits=10, shuffle=True, random_state=0)
        # y_pred_cv = np.zeros_like(df[y_name])
        # y_pred_ncv = np.zeros_like(df[y_name])
        # # for loop
        # for train_index, test_index in kf.split(df):
        #     train_df = df.iloc[train_index]
        #     test_df = df.iloc[test_index]
        #     # train
        #     X_train = train_df[regressor_names]
        #     y_train = train_df[y_name]
        #     X_test = test_df[regressor_names]
        #     y_test = test_df[y_name]
        #     # from sklearn.decomposition import PCA
        #     # pca = PCA(n_components=2)
        #     # X_train = pca.fit_transform(X_train)
        #     # X_test = pca.transform(X_test)
        #
        #     model = RidgeCV(alphas=np.logspace(-7, 7, 15))
        #     model.fit(X_train, y_train)
        #     # test
        #     y_pred_cv[test_index] = model.predict(X_test)
        #     y_pred_ncv[train_index] = model.predict(X_train)
        #
        # corr = np.corrcoef(y_pred_cv, df[y_name])[0,1]
        # # corr = np.corrcoef(y_pred_ncv, df[y_name])[0,1]
        corr_list = []
        for col in regressor_names:
            formula = f'{y_name} ~ {col}'
            model = smf.ols(formula, data=df).fit()
            corr = model.rsquared
            corr_list.append(corr)
        # print(y_name, y_name_mapping[y_name], corr)
        print(y_name_mapping[y_name], corr_list)
    # bar_width = 0.1
    # plt.bar(np.arange(len(y_list))+cog_idx * bar_width-bar_width, corr_list,
    #         color=f'C{cog_idx}', label=cog_type, width=bar_width)
    # plt.xticks(rotation=45)
    # plt.xticks(np.arange(len(y_list)), [y_name_mapping[y_name] for y_name in y_list])
    # plt.ylabel('Correlation')
    # plt.title('Study 2')
    # plt.legend()
    # # plt.ylim(0,1)
    # plt.tight_layout()
    # plt.show()