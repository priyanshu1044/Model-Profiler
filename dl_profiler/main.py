import torch
import torch.nn as nn
import torchvision.models as models
from profiler import ModelProfiler
from visualizer import PerformanceVisualizer
import matplotlib.pyplot as plt

def create_sample_input(batch_size=32, channels=3, height=224, width=224):
    return torch.randn(batch_size, channels, height, width)

def profile_model(model, input_tensor, use_cuda=True):
    # Initialize profiler
    profiler = ModelProfiler(model, use_cuda=use_cuda)
    
    # Move model and input to GPU if available
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
    
    # Warm-up run
    model(input_tensor)
    profiler.reset()
    
    # Profile forward pass
    output = model(input_tensor)
    
    # Compute backward pass
    loss = output.mean()
    loss.backward()
    
    return profiler.get_layer_times(), profiler.get_bottleneck_layers()

def main():
    # Create sample model and input
    model = models.resnet18(pretrained=False)
    input_tensor = create_sample_input()
    
    # Profile on GPU
    print("Profiling on GPU...")
    gpu_times, gpu_bottlenecks = profile_model(model, input_tensor, use_cuda=True)
    
    # Profile on CPU
    print("\nProfiling on CPU...")
    cpu_times, cpu_bottlenecks = profile_model(model, input_tensor, use_cuda=False)
    
    # Initialize visualizer
    visualizer = PerformanceVisualizer()
    
    # Generate and display visualizations
    print("\nGenerating performance visualizations...")
    
    # Layer-wise timing plot
    gpu_plot = visualizer.plot_layer_times(gpu_times, "GPU Layer-wise Execution Times")
    gpu_plot.savefig('gpu_layer_times.png')
    plt.close()
    
    # Timeline plot
    timeline_plot = visualizer.plot_execution_timeline(gpu_times)
    timeline_plot.savefig('execution_timeline.png')
    plt.close()
    
    # GPU vs CPU comparison
    comparison_plot = visualizer.plot_gpu_vs_cpu_comparison(gpu_times, cpu_times)
    comparison_plot.savefig('gpu_vs_cpu_comparison.png')
    plt.close()
    
    # Generate performance report
    report = visualizer.generate_performance_report(gpu_times)
    print("\nPerformance Report:")
    print(report)
    
    # Print bottleneck analysis
    print("\nTop GPU Bottleneck Layers:")
    for layer, time in gpu_bottlenecks:
        print(f"{layer}: {time:.2f}ms")

if __name__ == '__main__':
    main()