import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

class PerformanceVisualizer:
    def __init__(self):
        plt.style.use('seaborn')

    def plot_layer_times(self, layer_times: Dict[str, List[float]], title: str = "Layer-wise Execution Times"):
        """Plot average execution time for each layer."""
        avg_times = {name: np.mean(times) for name, times in layer_times.items()}
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(avg_times)), list(avg_times.values()))
        plt.xticks(range(len(avg_times)), list(avg_times.keys()), rotation=45, ha='right')
        plt.xlabel('Layer Name')
        plt.ylabel('Average Execution Time (ms)')
        plt.title(title)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms',
                    ha='center', va='bottom')

        plt.tight_layout()
        return plt.gcf()

    def plot_execution_timeline(self, layer_times: Dict[str, List[float]]):
        """Plot execution timeline showing cumulative time."""
        avg_times = {name: np.mean(times) for name, times in layer_times.items()}
        cumulative_times = np.cumsum(list(avg_times.values()))
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(cumulative_times)), cumulative_times, marker='o')
        plt.xticks(range(len(avg_times)), list(avg_times.keys()), rotation=45, ha='right')
        plt.xlabel('Layer Execution Order')
        plt.ylabel('Cumulative Time (ms)')
        plt.title('Model Execution Timeline')
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()

    def generate_performance_report(self, layer_times: Dict[str, List[float]]) -> pd.DataFrame:
        """Generate a detailed performance report as a pandas DataFrame."""
        data = []
        for name, times in layer_times.items():
            data.append({
                'Layer': name,
                'Mean Time (ms)': np.mean(times),
                'Std Dev (ms)': np.std(times),
                'Min Time (ms)': np.min(times),
                'Max Time (ms)': np.max(times),
                'Total Time (ms)': np.sum(times)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Mean Time (ms)', ascending=False)
        return df

    def plot_gpu_vs_cpu_comparison(self, gpu_times: Dict[str, List[float]], 
                                 cpu_times: Dict[str, List[float]]):
        """Plot GPU vs CPU execution time comparison."""
        gpu_avg = {name: np.mean(times) for name, times in gpu_times.items()}
        cpu_avg = {name: np.mean(times) for name, times in cpu_times.items()}
        
        layers = list(set(gpu_avg.keys()) | set(cpu_avg.keys()))
        gpu_values = [gpu_avg.get(layer, 0) for layer in layers]
        cpu_values = [cpu_avg.get(layer, 0) for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, gpu_values, width, label='GPU')
        plt.bar(x + width/2, cpu_values, width, label='CPU')
        
        plt.xlabel('Layer Name')
        plt.ylabel('Execution Time (ms)')
        plt.title('GPU vs CPU Execution Time Comparison')
        plt.xticks(x, layers, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        return plt.gcf()