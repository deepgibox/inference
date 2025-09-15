# TensorRT Shape Inference Solutions

## The Problem
TensorRT requires all tensor shapes to be fully specified at graph compilation time. The error you're seeing:

```
TensorRT input: Resize__27:0 has no shape specified. Please run shape inference on the onnx model first.
```

This happens because:
1. Your ONNX model has dynamic shapes (dimensions marked as -1 or symbolic)
2. TensorRT cannot optimize graphs with unknown dimensions
3. The specific tensor `Resize__27:0` lacks shape information

## Solutions (in order of preference)

### 1. **Fix the ONNX Model (Recommended)**

#### Option A: Use ONNX Shape Inference
```python
import onnx
from onnx import shape_inference

# Load your model
model = onnx.load("YOLOv5.onnx")

# Run shape inference
inferred_model = shape_inference.infer_shapes(model)

# Save the updated model
onnx.save(inferred_model, "YOLOv5_with_shapes.onnx")
```

#### Option B: Fix shapes during export
When exporting from PyTorch/TensorFlow, specify explicit input shapes:
```python
# PyTorch example
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,  # Remove this or specify fixed axes
    opset_version=11
)
```

### 2. **Use TensorRT with Fallback (Your Current Approach)**
This is working correctly in your code - TensorRT fails gracefully and falls back to CUDA.

### 3. **Configure TensorRT for Dynamic Shapes**
For models that must have dynamic shapes, configure optimization profiles:

```rust
TensorRTExecutionProvider::default()
    .with_trt_optimization_profiles(profiles)  // If available in your ort version
    .build()
```

### 4. **Selective TensorRT Usage**
Only use TensorRT for specific subgraphs:

```rust
TensorRTExecutionProvider::default()
    .with_min_subgraph_size(10)  // Higher threshold = fewer subgraphs attempted
    .build()
```

## Immediate Workaround

Your current fallback mechanism is the best approach for now:

```rust
// TensorRT -> CUDA -> CPU fallback (exactly what you have)
```

## For Production

1. **Pre-process your ONNX model** with shape inference
2. **Use fixed input sizes** if your use case allows
3. **Keep the fallback mechanism** for robustness

## Model-Specific Notes

For YOLOv5:
- Input shape is typically `(1, 3, 640, 640)` or `(1, 3, 512, 512)`
- Consider using models exported with fixed shapes
- YOLOv5 models from Ultralytics often work better with TensorRT when exported correctly

## Testing Commands

```bash
# Test with your current fallback mechanism
cargo run

# Test the advanced solutions
cargo run --example tensorrt_shape_fix
```
