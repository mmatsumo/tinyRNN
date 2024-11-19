import numpy as np
import joblib

a = joblib.load(r'D:\cognitive_dynamics\saved_model\exp_classify_millerrat55\rnn_type-GRU.hidden_dim-10.l1_weight-1e-05\outerfold1_innerfold0_seed0\best_pass.pkl')
import matplotlib.pyplot as plt
plt.figure()
target = a['test']['target'][1][2].cpu().numpy() # 0, 1
target_label = target * 2 - 1 # -1, 1
ce_loss = a['test']['behav_loss_total_label']
logits = (-ce_loss - np.log(1-np.exp(-ce_loss)) ) * target_label
n_trial, n_sample = logits.shape
for line_idx in range(n_sample):
    plt.plot(logits[:,line_idx], alpha=0.1, color=f'C{target[0,line_idx]}')
# plt.plot(range(n_trial),logits.mean(-1))
# plt.fill_between(range(n_trial),logits.mean(-1)-logits.std(-1),logits.mean(-1)+logits.std(-1), alpha=0.4)
plt.hlines(0,0,n_trial,linestyles='dashed',colors='k',alpha=0.5)
plt.ylim(-np.max(logits),np.max(logits))
plt.xlabel('Trial')
plt.ylabel('Model label logit')
plt.show()

inputs = a['test']['input']
input_colors = inputs[...,0] *4 + inputs[...,1] *2 + inputs[...,2]
input_colors = input_colors[:-1]
input_colors = input_colors.flatten()
input_colors = ['C'+str(int(i)) for i in input_colors]
logit_change = logits[:-1] - logits[1:]
logits = logits[:-1]
plt.figure()
plt.scatter(logits.flatten(),logit_change.flatten(), s=0.1,color=input_colors)
plt.xlabel('Model label logit')
plt.ylabel('Model label logit change')
plt.show()