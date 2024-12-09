import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

class ModelProfiler:
    def __init__(self, model: nn.Module, use_cuda: bool = True):
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.layer_times = defaultdict(list)
        self.cuda_events = {}
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up forward and backward hooks for all layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
                self.hooks.append(
                    module.register_forward_hook(self._generate_forward_hook(name))
                )
                if module.parameters():
                    self.hooks.append(
                        module.register_backward_hook(self._generate_backward_hook(name))
                    )

    def _generate_forward_hook(self, name: str):
        def hook(module, input, output):
            if self.use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self.cuda_events[f"{name}_forward"] = (start_event, end_event)
            else:
                self.layer_times[f"{name}_forward"].append(time.time())
        return hook

    def _generate_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if self.use_cuda:
                if f"{name}_forward" in self.cuda_events:
                    start_event, end_event = self.cuda_events[f"{name}_forward"]
                    end_event.record()
                    torch.cuda.synchronize()
                    self.layer_times[f"{name}_forward"].append(
                        start_event.elapsed_time(end_event)
                    )
            else:
                if f"{name}_forward" in self.layer_times:
                    start_time = self.layer_times[f"{name}_forward"].pop()
                    self.layer_times[f"{name}_forward"].append(
                        (time.time() - start_time) * 1000  # Convert to ms
                    )
        return hook

    def get_layer_times(self) -> Dict[str, List[float]]:
        """Return the execution times for each layer."""
        return dict(self.layer_times)

    def get_total_time(self) -> float:
        """Return the total execution time across all layers."""
        return sum(sum(times) for times in self.layer_times.values())

    def get_bottleneck_layers(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-k layers with highest average execution time."""
        avg_times = [
            (name, sum(times) / len(times))
            for name, times in self.layer_times.items()
            if times
        ]
        return sorted(avg_times, key=lambda x: x[1], reverse=True)[:top_k]

    def reset(self):
        """Reset all timing measurements."""
        self.layer_times.clear()
        self.cuda_events.clear()

    def __del__(self):
        """Clean up hooks when the profiler is deleted."""
        for hook in self.hooks:
            hook.remove()