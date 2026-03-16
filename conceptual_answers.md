
# Conceptual Answers

## 1. Tracing vs Scripting

Tracing records operations executed with example inputs. If control flow depends on tensor values, tracing may capture only the path taken during tracing.

Scripting analyzes Python code directly and preserves control flow.

Example:

def f(x):
    if x.sum() > 0:
        return x * 2
    else:
        return x / 2

Tracing records only one branch depending on the example input, while scripting correctly handles both branches.

## 2. JAX Performance Issue

The loop builds a Python list inside a JIT function, which prevents optimal compilation and vectorization.

Fix:

Use `jax.vmap` or reshape the tensor and compute sums in parallel.

## 3. Operator Fusion

Eager:
- 3 kernel launches
- Each op reads and writes global memory

Fused:
- 1 kernel launch
- Intermediate results stored in registers

Bandwidth reduction ≈ 3x.

## 4. Dynamic Control Flow

Dynamic control flow depends on runtime tensor values which JIT compilers cannot easily determine at compile time.

JAX uses primitives like `jax.lax.cond` and `jax.lax.scan`.

PyTorch Dynamo attempts graph capture and falls back to eager execution when control flow cannot be compiled.

## 5. First torch.compile Call

The first call performs:
- Graph capture
- FX graph creation
- Graph optimization
- Kernel generation with TorchInductor
- Compilation to CUDA kernels

Later calls reuse the compiled kernels.
