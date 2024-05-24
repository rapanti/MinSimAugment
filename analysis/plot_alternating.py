import numpy as np
import matplotlib.pyplot as plt

# Provided dictionaries with their actual keys
hvp_seeds_final = {
    # 100: [74.534, 74.6, 74.402],
    # 200: [76.104, 75.826, 76.136],
    300: [76.380, 76.400, 76.324], # -> 76.368
    400: [76.536, 76.646, 76.772], # -> 76.651
    500: [76.746, 76.690, 77.068], # -> 76.835
    600: [76.662, 76.624, 76.866], # -> 76.717
    800: [76.868],
}

baseline_seeds_final = {
    # 143: [75.362, 74.54, 74.47],
    286: [76.214, 75.686, 75.944],
    429: [76.682, 76.048, 76.188], # -> 76.306
    572: [76.634, 76.282, 76.476], # -> 76.464
    715: [76.668, 76.574], # -> 76.621
    858: [76.74, 76.492], # -> 76.616
    # 858: [76.616], # -> 76.616
    1144: [75.958],
}

# baseline_seeds_final = {
#     # 100: [75.362, 74.54, 74.47],
#     # 200: [76.214, 75.686, 75.944],
#     300: [76.682, 76.048, 76.188], # -> 76.306
#     400: [76.634, 76.282, 76.476], # -> 76.464
#     500: [76.668, 76.574], # -> 76.621
#     600: [76.74, 76.492], # -> 76.616
#     800: [76.582]  # hochgerechnet
# }

knn_hvp_seeds_final = {
    # 100: [70.66, 70.69, 70.74],
    # 200: [73.110, 72.984, 73.008],
    300: [73.326, 73.526, 73.654], # -> 73.502
    400: [73.932, 74.072, 73.790], # -> 73.931
    500: [74.056, 73.864, 74.206], # -> 74.042
    600: [74.286, 73.900, 74.300], # -> 74.162
    800: [74.030],
}

knn_baseline_seeds_final = {
    # 143: [71.880, 71.308, 71.268],
    286: [73.274, 72.906, 73.226],
    429: [73.694, 73.628, 73.360], # -> 73.561
    572: [73.696, 73.512, 73.742], # -> 73.650
    715: [73.842, 73.876], # -> 73.859
    858: [73.802, 73.902], # -> 73.852
    # 858: [73.902],
    1144: [73.448],
}

# knn_baseline_seeds_final = {
#     # 100: [71.880, 71.308, 71.268],
#     # 200: [73.274, 72.906, 73.226],
#     300: [73.694, 73.628, 73.360], # -> 73.561
#     400: [73.696, 73.512, 73.742], # -> 73.650
#     500: [73.842, 73.876], # -> 73.859
#     600: [73.802, 73.902], # -> 73.852
#     800: [73.448],
# }

# Extracting epochs and calculating averages and standard deviations
hvp_epochs = list(hvp_seeds_final.keys())
baseline_epochs = list(baseline_seeds_final.keys())
knn_hvp_epochs = list(knn_hvp_seeds_final.keys())
knn_baseline_epochs = list(knn_baseline_seeds_final.keys())

hvp_averages = [np.mean(hvp_seeds_final[epoch]) for epoch in hvp_epochs]
baseline_averages = [np.mean(baseline_seeds_final[epoch]) for epoch in baseline_epochs]
knn_hvp_averages = [np.mean(knn_hvp_seeds_final[epoch]) for epoch in knn_hvp_epochs]
knn_baseline_averages = [np.mean(knn_baseline_seeds_final[epoch]) for epoch in knn_baseline_epochs]

hvp_std = [np.std(hvp_seeds_final[epoch]) for epoch in hvp_epochs]
baseline_std = [np.std(baseline_seeds_final[epoch]) for epoch in baseline_epochs]
knn_hvp_std = [np.std(knn_hvp_seeds_final[epoch]) for epoch in knn_hvp_epochs]
knn_baseline_std = [np.std(knn_baseline_seeds_final[epoch]) for epoch in knn_baseline_epochs]

# Plotting
plt.figure(figsize=(10, 6))

# Linear evaluation plots for HVP
plt.plot(hvp_epochs, hvp_averages, label='DINO + HVP (linear eval)', color='blue')
plt.fill_between(hvp_epochs, np.array(hvp_averages) - np.array(hvp_std),
                 np.array(hvp_averages) + np.array(hvp_std), color='blue', alpha=0.2)

# Linear evaluation plots for Baseline
plt.plot(baseline_epochs, baseline_averages, label='DINO (linear eval)', color='red')
plt.fill_between(baseline_epochs, np.array(baseline_averages) - np.array(baseline_std),
                 np.array(baseline_averages) + np.array(baseline_std), color='red', alpha=0.2)

# k-NN evaluation plots for HVP
plt.plot(knn_hvp_epochs, knn_hvp_averages, label='DINO + HVP (k-NN eval)', color='green')
plt.fill_between(knn_hvp_epochs, np.array(knn_hvp_averages) - np.array(knn_hvp_std),
                 np.array(knn_hvp_averages) + np.array(knn_hvp_std), color='green', alpha=0.2)

# k-NN evaluation plots for Baseline
plt.plot(knn_baseline_epochs, knn_baseline_averages, label='DINO (k-NN eval)', color='purple')
plt.fill_between(knn_baseline_epochs, np.array(knn_baseline_averages) - np.array(knn_baseline_std),
                 np.array(knn_baseline_averages) + np.array(knn_baseline_std), color='purple', alpha=0.2)

plt.rcParams.update({'font.size': 15})  # General font size
# plt.rcParams.update({'axes.titlesize': 18})  # Title font size
# plt.rcParams.update({'axes.labelsize': 18})  # X and Y label font size
# plt.rcParams.update({'xtick.labelsize': 15})  # X tick label font size
# plt.rcParams.update({'ytick.labelsize': 18})  # Y tick label font size
# plt.rcParams.update({'legend.fontsize': 12})  # Legend font size

plt.ylim(73, 77.1)  # Adjust as needed
# plt.xlim(300, 800)  # Show only 300 to 800 epochs
plt.xlim(350, 800)  # Show only 300 to 800 epochs
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Effect of applying HVP every 3rd pretraining step on ImageNet', fontsize=15)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('alternating_training.png')
plt.show()
