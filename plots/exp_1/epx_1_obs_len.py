import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "font.size": 14,          # Base font size
    "axes.titlesize": 16,     # Title size
    "axes.labelsize": 15,     # X and Y label size
    "xtick.labelsize": 13,    # X tick size
    "ytick.labelsize": 13,    # Y tick size
    "legend.fontsize": 13,    # Legend font size
    "figure.titlesize": 16
})

data = [
    {"obs_len_min": 1, "n_embd": 64, "value": 6.680770643234253},
    {"obs_len_min": 5, "n_embd": 64, "value": 6.64389393234253},
    {"obs_len_min": 10, "n_embd": 64, "value": 6.630048810005188},
    {"obs_len_min": 15, "n_embd": 64, "value": 6.626832076072693},
    {"obs_len_min": 20, "n_embd": 64, "value": 6.6158041543960575},
    {"obs_len_min": 1, "n_embd": 128, "value": 6.624923145294189},
    {"obs_len_min": 5, "n_embd": 128, "value": 6.57939915561676},
    {"obs_len_min": 10, "n_embd": 128, "value": 6.5691175365448},
    {"obs_len_min": 15, "n_embd": 128, "value": 6.571250073432922},
    {"obs_len_min": 20, "n_embd": 128, "value": 6.577660829544067},
    {"obs_len_min": 1, "n_embd": 256, "value": 6.58676024723053},
    {"obs_len_min": 5, "n_embd": 256, "value": 6.553862809181213},
    {"obs_len_min": 10, "n_embd": 256, "value": 6.5591258592605595},
    {"obs_len_min": 15, "n_embd": 256, "value": 6.564734399795532},
    {"obs_len_min": 20, "n_embd": 256, "value": 6.574422384262085},
]

grouped = defaultdict(list)
for entry in data:
    grouped[entry["n_embd"]].append((entry["obs_len_min"], entry["value"]))

plt.figure()

for n_embd, points in sorted(grouped.items()):
    points = sorted(points, key=lambda x: x[0])
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y, marker='o', label=f'Hidden Dimension={n_embd}')

plt.xlabel("Observation horizen (minutes)")
plt.ylabel("Cross-Entropy Loss")
plt.title("Validation score of HeatMap-Based TrAISfromer")
plt.legend()
plt.savefig("plots/exp_1/obs_len_trais.png")
plt.close()


data = [
    {"obs_len_min": 5, "hidden_size": 256, "value": 72.47905521392822},
    {"obs_len_min": 1, "hidden_size": 512, "value": 82.52112561035156},
    {"obs_len_min": 20, "hidden_size": 512, "value": 68.0640233001709},
    {"obs_len_min": 20, "hidden_size": 256, "value": 68.77924668884278},
    {"obs_len_min": 1, "hidden_size": 256, "value": 80.66155865478515},
    {"obs_len_min": 15, "hidden_size": 256, "value": 69.00448278808594},
    {"obs_len_min": 10, "hidden_size": 512, "value": 68.61754537200927},
    {"obs_len_min": 10, "hidden_size": 256, "value": 69.50839395141601},
    {"obs_len_min": 5, "hidden_size": 512, "value": 71.39330184936523},
    {"obs_len_min": 15, "hidden_size": 512, "value": 68.41345510864258},
    {"obs_len_min": 1, "hidden_size": 128, "value": 80.92702737426758},
    {"obs_len_min": 5, "hidden_size": 128, "value": 75.87675308227539},
    {"obs_len_min": 10, "hidden_size": 128, "value": 74.15837610626221},
    {"obs_len_min": 15, "hidden_size": 128, "value": 73.00652192687988},
    {"obs_len_min": 20, "hidden_size": 128, "value": 73.45903952026367},
]

grouped = defaultdict(list)
for entry in data:
    grouped[entry["hidden_size"]].append((entry["obs_len_min"], entry["value"]))

plt.figure()

for hidden_size, points in sorted(grouped.items()):
    points = sorted(points, key=lambda x: x[0])
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y, marker='o', label=f'Hidden Dimension={hidden_size}')

plt.xlabel("Observation horizen (minutes)")
plt.ylabel("Average Displacement Error (m)")
plt.title("Validation score of RNN (GRU seq2seq)")
plt.legend()
plt.savefig("plots/exp_1/obs_len_rnn.png")
plt.close()



data = [
    # factorized
    {"head": "factorized", "param_count": 545412, "value": 6.507078412055969},
    {"head": "factorized", "param_count": 1168740, "value": 6.465705415725708},
    {"head": "factorized", "param_count": 2337480, "value": 6.462649014472961},

    # cnn
    {"head": "cnn", "param_count": 553688, "value": 6.571687815666198},
    {"head": "cnn", "param_count": 1184240, "value": 6.51312949180603},
    {"head": "cnn", "param_count": 2333372, "value": 6.497750424385071},

    # mixture
    {"head": "mixture", "param_count": 540222, "value": 6.892789298057556},
    {"head": "mixture", "param_count": 1170481, "value": 6.7496907482147215},
    {"head": "mixture", "param_count": 2340962, "value": 6.683947962760925},

    # lowrank
    {"head": "lowrank", "param_count": 540093, "value": 6.958539124488831},
    {"head": "lowrank", "param_count": 1170352, "value": 6.751708721160889},
    {"head": "lowrank", "param_count": 2340833, "value": 6.66746941947937},
]

grouped = defaultdict(list)
for entry in data:
    grouped[entry["head"]].append((entry["param_count"], entry["value"]))

plt.figure()

for head, points in sorted(grouped.items()):
    points = sorted(points, key=lambda x: x[0])
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y, marker='o', label=f'{head}')

plt.xlabel("Parameter Count")
plt.ylabel("Cross-Entropy")
plt.title("Validation score of Heatmap Heads")
plt.legend(loc="upper right")

plt.savefig("plots/exp_3/head_scaling.png")
plt.close()

print("Saved to plots/exp_heatmap/head_scaling.png")