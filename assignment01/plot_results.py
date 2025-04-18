import pandas as pd
import matplotlib.pyplot as plt

# 1) Read the CSV file into a pandas DataFrame
df = pd.read_csv("task03_results.txt")

# 2) Extract unique matrix sizes (assuming n == m in your results)
matrix_sizes = sorted(df['n'].unique())

# ---------------------------------------------------------
# Figure 1: 2x2 subplots for speedups
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)

# axes[0, 0] -> Row Summation Speedup
# axes[0, 1] -> Column Summation Speedup
# axes[1, 0] -> Row Reduction Speedup
# axes[1, 1] -> Column Reduction Speedup

for size in matrix_sizes:
    subset = df[df['n'] == size]
    # Plot row summation speedup
    axes[0, 0].plot(subset['blockSize'], subset['rowSpeedup'], marker='o',
                    label=f"{size}x{size}")
    # Plot column summation speedup
    axes[0, 1].plot(subset['blockSize'], subset['colSpeedup'], marker='o',
                    label=f"{size}x{size}")
    # Plot row reduction speedup
    axes[1, 0].plot(subset['blockSize'], subset['rowRedSpeedup'], marker='o',
                    label=f"{size}x{size}")
    # Plot column reduction speedup
    axes[1, 1].plot(subset['blockSize'], subset['colRedSpeedup'], marker='o',
                    label=f"{size}x{size}")

# Set titles, labels, legends, etc.
axes[0, 0].set_title("Row Summation Speedup vs Block Size")
axes[0, 0].set_xlabel("Block Size")
axes[0, 0].set_ylabel("Speedup")

axes[0, 1].set_title("Column Summation Speedup vs Block Size")
axes[0, 1].set_xlabel("Block Size")
axes[0, 1].set_ylabel("Speedup")

axes[1, 0].set_title("Row Reduction Speedup vs Block Size")
axes[1, 0].set_xlabel("Block Size")
axes[1, 0].set_ylabel("Speedup")

axes[1, 1].set_title("Column Reduction Speedup vs Block Size")
axes[1, 1].set_xlabel("Block Size")
axes[1, 1].set_ylabel("Speedup")

# Show legends (just once per subplot if multiple lines)
for ax in axes.flat:
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Figure 2: 1x2 subplots for row/column errors
# ---------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

# axes2[0] -> Row Error vs Block Size
# axes2[1] -> Column Error vs Block Size

for size in matrix_sizes:
    subset = df[df['n'] == size]
    # Plot row error
    axes2[0].plot(subset['blockSize'], subset['rowError'], marker='o',
                  label=f"{size}x{size}")
    # Plot column error
    axes2[1].plot(subset['blockSize'], subset['colError'], marker='o',
                  label=f"{size}x{size}")

axes2[0].set_title("Row Error vs Block Size")
axes2[0].set_xlabel("Block Size")
axes2[0].set_ylabel("Relative Error")

axes2[1].set_title("Column Error vs Block Size")
axes2[1].set_xlabel("Block Size")
axes2[1].set_ylabel("Relative Error")

for ax in axes2.flat:
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()