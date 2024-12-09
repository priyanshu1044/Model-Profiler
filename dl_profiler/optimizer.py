import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from .profiler import ModelProfiler

class ModelOptimizer:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model
        self.input_shape = input_shape
        self.profiler = ModelProfiler(model)

    def find_optimal_batch_size(self, min_batch: int = 1, max_batch: int = 128,
                               step: int = 8) -> Tuple[int, float]:
        """Find the optimal batch size by measuring throughput."""
        best_batch_size = min_batch
        best_throughput = 0.0
        batch_sizes = range(min_batch, max_batch + 1, step)

        for batch_size in batch_sizes:
            # Create input tensor with current batch size
            input_shape = (batch_size,) + self.input_shape[1:]
            input_tensor = torch.randn(input_shape)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                self.model.cuda()

            # Measure throughput
            self.profiler.reset()
            output = self.model(input_tensor)
            output.mean().backward()

            total_time = self.profiler.get_total_time()
            throughput = batch_size / (total_time / 1000)  # samples per second

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size

        return best_batch_size, best_throughput

    def analyze_layer_fusion(self) -> List[Dict]:
        """Analyze potential layer fusion opportunities."""
        fusion_opportunities = []
        prev_layer = None
        prev_name = None

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                if prev_layer is not None:
                    # Check for common fusion patterns
                    if (
                        (isinstance(prev_layer, nn.Conv2d) and isinstance(module, nn.BatchNorm2d)) or
                        (isinstance(prev_layer, nn.BatchNorm2d) and isinstance(module, nn.ReLU)) or
                        (isinstance(prev_layer, nn.Conv2d) and isinstance(module, nn.ReLU))
                    ):
                        fusion_opportunities.append({
                            'layer1': prev_name,
                            'layer2': name,
                            'type1': prev_layer.__class__.__name__,
                            'type2': module.__class__.__name__,
                            'recommendation': 'Consider fusing these layers for better performance'
                        })
                
                prev_layer = module
                prev_name = name

        return fusion_opportunities

    def suggest_optimizations(self) -> Dict:
        """Generate comprehensive optimization suggestions."""
        # Profile current performance
        input_tensor = torch.randn(self.input_shape)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.model.cuda()

        self.profiler.reset()
        output = self.model(input_tensor)
        output.mean().backward()

        # Get performance metrics
        layer_times = self.profiler.get_layer_times()
        bottlenecks = self.profiler.get_bottleneck_layers()

        # Find optimal batch size
        optimal_batch, throughput = self.find_optimal_batch_size()

        # Analyze fusion opportunities
        fusion_opportunities = self.analyze_layer_fusion()

        return {
            'bottleneck_layers': bottlenecks,
            'optimal_batch_size': optimal_batch,
            'max_throughput': throughput,
            'fusion_opportunities': fusion_opportunities,
            'general_recommendations': [
                'Consider using torch.jit.script for the model',
                'Enable torch.backends.cudnn.benchmark for consistent input sizes',
                'Use channels_last memory format for CNN layers',
                'Consider quantization for inference optimization'
            ]
        }