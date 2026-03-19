import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv('optimized_benchmark.csv')

# Plot FPS over time
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['frame'], df['fps'])
plt.xlabel('Frame')
plt.ylabel('FPS')
plt.title('Performance Over Time')

# Plot memory usage
plt.subplot(1, 2, 2)
plt.plot(df['frame'], df['memory_mb'])
plt.xlabel('Frame')
plt.ylabel('Memory (MB)')
plt.title('Memory Usage')

plt.tight_layout()
plt.show()

# Print statistics
print(f"Average FPS: {df['fps'].mean():.2f}")
print(f"Average CPU: {df['cpu_percent'].mean():.1f}%")
print(f"Average Memory: {df['memory_mb'].mean():.1f}MB")