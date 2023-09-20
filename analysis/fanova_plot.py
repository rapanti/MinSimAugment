import matplotlib.pyplot as plt

# Data
categories = ["IoU", "Rel. Distance", "Colorjitter Distance", "Brightness", "Contrast", "Saturation", "Hue", "All"]
values = [0.2154872174, 0.05672040951, 0.02376414143, 0.1153980024, 0.03592686127, 0.01640492011, 0.04760971017, 0.04981635206]

# Create a bar plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Reduce bar width
bar_width = 0.7

plt.bar(categories, values, color='royalblue', width=bar_width)
plt.xlabel('Metrics')
plt.ylabel('Importance')
plt.title('Importance of Metrics When Applying HVS')
plt.xticks(rotation=0, fontsize=9)  # Rotate x-axis labels for better readability

# Display the plot
plt.tight_layout()

plt.savefig('fanova_plot.png')

plt.show()
