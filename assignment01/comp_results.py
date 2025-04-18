#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Define desired matrix sizes and fixed block size
desired_sizes = [1000, 5000, 10000, 25000]
BLOCK_SIZE = 32

# Filenames for single and double precision results
single_file = "task03_results.txt"  # Single precision results
double_file = "task04_results.txt"  # Double precision results

# Read CSV data from the files
df_single = pd.read_csv(single_file)
df_double = pd.read_csv(double_file)

# Filter rows: only keep those with the fixed block size and desired matrix sizes
df_single = df_single[(df_single["blockSize"] == BLOCK_SIZE) & (df_single["n"].isin(desired_sizes))]
df_double = df_double[(df_double["blockSize"] == BLOCK_SIZE) & (df_double["n"].isin(desired_sizes))]

# Merge the two dataframes on n, m, and blockSize
# suffixes=("_single", "_double") ensures column names don't collide
df_merged = pd.merge(df_single, df_double, on=["n", "m", "blockSize"], suffixes=("_single", "_double"))
df_merged.sort_values(by="n", inplace=True)

# ------------------------------------------------------------------------------
# FIGURE 1: 2x2 subplots for speedups
# ------------------------------------------------------------------------------
fig1, axs = plt.subplots(2, 2, figsize=(10, 8))

# (0,0): Row Summation Speedup
axs[0, 0].plot(df_merged["n"], df_merged["rowSpeedup_single"], marker="o", label="Single Precision")
axs[0, 0].plot(df_merged["n"], df_merged["rowSpeedup_double"], marker="x", label="Double Precision")
axs[0, 0].set_title(f"Row Summation Speedup (BlockSize={BLOCK_SIZE})")
axs[0, 0].set_xlabel("Matrix Size (n=m)")
axs[0, 0].set_ylabel("Speedup")
axs[0, 0].grid(True)
axs[0, 0].legend()

# (0,1): Column Summation Speedup
axs[0, 1].plot(df_merged["n"], df_merged["colSpeedup_single"], marker="o", label="Single Precision")
axs[0, 1].plot(df_merged["n"], df_merged["colSpeedup_double"], marker="x", label="Double Precision")
axs[0, 1].set_title(f"Column Summation Speedup (BlockSize={BLOCK_SIZE})")
axs[0, 1].set_xlabel("Matrix Size (n=m)")
axs[0, 1].set_ylabel("Speedup")
axs[0, 1].grid(True)
axs[0, 1].legend()

# (1,0): Row Reduction Speedup
axs[1, 0].plot(df_merged["n"], df_merged["rowRedSpeedup_single"], marker="o", label="Single Precision")
axs[1, 0].plot(df_merged["n"], df_merged["rowRedSpeedup_double"], marker="x", label="Double Precision")
axs[1, 0].set_title(f"Row Reduction Speedup (BlockSize={BLOCK_SIZE})")
axs[1, 0].set_xlabel("Matrix Size (n=m)")
axs[1, 0].set_ylabel("Speedup")
axs[1, 0].grid(True)
axs[1, 0].legend()

# (1,1): Column Reduction Speedup
axs[1, 1].plot(df_merged["n"], df_merged["colRedSpeedup_single"], marker="o", label="Single Precision")
axs[1, 1].plot(df_merged["n"], df_merged["colRedSpeedup_double"], marker="x", label="Double Precision")
axs[1, 1].set_title(f"Column Reduction Speedup (BlockSize={BLOCK_SIZE})")
axs[1, 1].set_xlabel("Matrix Size (n=m)")
axs[1, 1].set_ylabel("Speedup")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# FIGURE 2: 2x2 subplots for relative errors
# ------------------------------------------------------------------------------
fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8))

# (0,0): Row Error
axs2[0, 0].plot(df_merged["n"], df_merged["rowError_single"], marker="o", label="Single Precision")
axs2[0, 0].plot(df_merged["n"], df_merged["rowError_double"], marker="x", label="Double Precision")
axs2[0, 0].set_title("Row Error vs Matrix Size")
axs2[0, 0].set_xlabel("Matrix Size (n=m)")
axs2[0, 0].set_ylabel("Relative Error")
axs2[0, 0].grid(True)
axs2[0, 0].legend()

# (0,1): Column Error
axs2[0, 1].plot(df_merged["n"], df_merged["colError_single"], marker="o", label="Single Precision")
axs2[0, 1].plot(df_merged["n"], df_merged["colError_double"], marker="x", label="Double Precision")
axs2[0, 1].set_title("Column Error vs Matrix Size")
axs2[0, 1].set_xlabel("Matrix Size (n=m)")
axs2[0, 1].set_ylabel("Relative Error")
axs2[0, 1].grid(True)
axs2[0, 1].legend()

# (1,0): Row Reduction Error
axs2[1, 0].plot(df_merged["n"], df_merged["rowRedError_single"], marker="o", label="Single Precision")
axs2[1, 0].plot(df_merged["n"], df_merged["rowRedError_double"], marker="x", label="Double Precision")
axs2[1, 0].set_title("Row Reduction Error vs Matrix Size")
axs2[1, 0].set_xlabel("Matrix Size (n=m)")
axs2[1, 0].set_ylabel("Relative Error")
axs2[1, 0].grid(True)
axs2[1, 0].legend()

# (1,1): Column Reduction Error
axs2[1, 1].plot(df_merged["n"], df_merged["colRedError_single"], marker="o", label="Single Precision")
axs2[1, 1].plot(df_merged["n"], df_merged["colRedError_double"], marker="x", label="Double Precision")
axs2[1, 1].set_title("Column Reduction Error vs Matrix Size")
axs2[1, 1].set_xlabel("Matrix Size (n=m)")
axs2[1, 1].set_ylabel("Relative Error")
axs2[1, 1].grid(True)
axs2[1, 1].legend()

plt.tight_layout()
plt.show()