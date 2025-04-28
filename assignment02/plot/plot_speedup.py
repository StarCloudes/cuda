import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df['Block'] = df['BlockX'].astype(str) + 'x' + df['BlockY'].astype(str)
pivot = df.pivot(index='Size', columns='Block', values='Speedup')
plt.figure(figsize=(8,5))
for block in pivot.columns:
    plt.plot(pivot.index, pivot[block], marker='o', label=block)

plt.xlabel('Matrix Size (N = M)')
plt.ylabel('Speedup (CPU / GPU)')
plt.title('Speedup vs. Matrix Size for Various Block Configurations')
plt.legend(title='Block Size')
plt.grid(True)
plt.tight_layout()

plt.show()
