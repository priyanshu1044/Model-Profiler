# Deep Learning Model Profiler

A lightweight developer tool for analyzing layer-wise performance of CNN models in PyTorch. This tool helps identify bottlenecks in GPU vs CPU execution paths and provides detailed performance visualizations.

## Features

- Layer-wise performance analysis of PyTorch models
- CUDA event integration for precise GPU kernel timing
- Comparative analysis between GPU and CPU execution
- Detailed performance visualizations and reports
- Bottleneck identification and optimization suggestions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/priyanshu1044/Model-Profiler.git
cd Model-Profiler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from dl_profiler.profiler import ModelProfiler
from dl_profiler.visualizer import PerformanceVisualizer

# Initialize your PyTorch model
model = your_model

# Create profiler instance
profiler = ModelProfiler(model, use_cuda=True)

# Run your model with profiling
output = model(input_tensor)
output.backward()  # if you want to profile backward pass

# Get timing results
layer_times = profiler.get_layer_times()
bottlenecks = profiler.get_bottleneck_layers()

# Visualize results
visualizer = PerformanceVisualizer()
visualizer.plot_layer_times(layer_times)
visualizer.plot_execution_timeline(layer_times)
```

## Example Output

The profiler generates several visualizations:
- Layer-wise execution time analysis
- Execution timeline
- GPU vs CPU performance comparison
- Detailed performance reports

## Performance Optimization

The tool helps identify performance bottlenecks and suggests optimizations:
- Layer fusion opportunities
- Batch size optimization
- GPU memory usage analysis
- Computation vs memory access patterns

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA Toolkit (for GPU profiling)
- matplotlib
- pandas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License