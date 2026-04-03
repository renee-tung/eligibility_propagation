# TensorFlow 2.x Migration Guide for E-prop

## Overview

This directory now includes TensorFlow 2.x compatible versions of the e-prop training code while maintaining **exact mathematical equivalence** with the original TensorFlow 1.x implementation.

### Key Files

- **models_tf2.py**: TF 2.x compatible neural network models (direct replacement for models.py)
- **tutorial_evidence_accumulation_with_alif_tf2.py**: TF 2.x training script
- **verify_tf2_implementation.py**: Comprehensive test suite to verify numerical correctness

## Critical Design Decisions

### The E-prop Implementation is Preserved Exactly

The `EligALIF.compute_eligibility_traces()` and `EligALIF.compute_loss_gradient()` methods are **identical** to the original implementation:

1. All mathematical operations are preserved line-for-line
2. The `@tf.custom_gradient` decorator for `SpikeFunction` works unchanged in TF 2.x
3. The same `tf.scan` operations are used for eligibility trace computation
4. Numerically, results should be identical (within floating-point precision)

### What Changed

#### 1. **Model Architecture** (models_tf2.py)

**Old (TF 1.x)**:
```python
import tensorflow.compat.v1 as tf
with tf.variable_scope('InputWeights'):
    self.w_in_var = tf.Variable(...)
```

**New (TF 2.x)**:
```python
import tensorflow as tf
self.w_in_var = tf.Variable(..., trainable=True, name='InputWeights')
```

- Removed `tf.variable_scope` usage (not needed in eager execution)
- `EligALIF` inherits from `tf.Module` instead of generic RNN cell classes
- Variable creation is direct and explicit
- The class works with eager execution by default

#### 2. **Training Script** (tutorial_evidence_accumulation_with_alif_tf2.py)

**Old (TF 1.x)**:
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
losses = sess.run([loss_cls, loss_reg_f], feed_dict={input_spikes: data, ...})
```

**New (TF 2.x)**:
```python
# Eager execution - no session needed
z, s, v, b, losses = unroll_network(cell, input_spikes, model_dict, FLAGS)
# Gradients computed with GradientTape
with tf.GradientTape() as tape:
    loss = compute_loss(...)
grads = tape.gradient(loss, trainable_vars)
```

Key improvements:
- No `tf.Session` or explicit initialization
- No `feed_dict` - direct tensor inputs
- Variables auto-initialized on first assignment
- Automatic gradient computation with `tf.GradientTape`

#### 3. **Command-line Arguments** 

**Old (TF 1.x)**:
```python
tf.app.flags.DEFINE_integer('n_batch', 64, 'batch size')
FLAGS = tf.app.flags.FLAGS
```

**New (TF 2.x)**:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_batch', type=int, default=64)
FLAGS = parser.parse_args()
```

Run with: `python tutorial_evidence_accumulation_with_alif_tf2.py --n_batch 64 --eprop`

#### 4. **RNN Unrolling**

**Old (TF 1.x)**:
```python
outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
```

**New (TF 2.x)**:
```python
state = cell.zero_state(batch_size, dtype=tf.float32)
outputs_z = []
for t in range(input_spikes.shape[1]):
    input_t = input_spikes[:, t, :]
    [z, s], state = cell(input_t, state)
    outputs_z.append(z)
z_tensor = tf.stack(outputs_z, axis=1)
```

- Manual loop gives better control and debugging capability
- Allows @tf.function decoration for graph-mode execution
- Equivalent computational graph to `tf.nn.dynamic_rnn`

## Verification

Before using in production, verify the TF 2.x implementation produces identical results:

```bash
python verify_tf2_implementation.py
```

This runs 6 comprehensive tests:
1. **SpikeFunction** - Forward/backward pass correctness
2. **Exponential Convolution** - Filter math validation
3. **Eligibility Trace Computation** - Core e-prop algorithm
4. **Loss Gradient Computation** - Gradient computation
5. **RNN Unrolling** - Temporal unfolding
6. **Gradient Flow** - End-to-end backpropagation

All tests should pass with `✓` marks.

## Usage

### Basic Training

```bash
# E-prop with hardcoded gradients
python tutorial_evidence_accumulation_with_alif_tf2.py \
    --eprop \
    --eprop_impl hardcoded \
    --n_batch 64 \
    --n_iter 2000

# BPTT baseline
python tutorial_evidence_accumulation_with_alif_tf2.py \
    --n_iter 2000
```

### Configuration Options

```bash
# Network parameters
--tau_v 20              # Membrane time constant [ms]
--thr 0.6              # Spike threshold
--tau_a 2000           # Adaptation time constant [ms]
--n_ref 5              # Refractory steps

# Training parameters
--learning_rate 0.005
--n_batch 64
--n_iter 2000
--validate_every 10

# E-prop specific
--eprop                # Enable e-prop (disable for BPTT)
--eprop_impl hardcoded # ['hardcoded', 'autodiff']
--feedback symmetric   # ['symmetric', 'random']
--f_regularization_type simple  # ['simple', 'online']

# Output
--output_dir results
--save_outputs         # Save results to JSON
--do_plot              # Live plotting during training
```

## Migration Checklist

If migrating other scripts in this repository:

- [ ] Replace `import tensorflow.compat.v1 as tf` with `import tensorflow as tf`
- [ ] Remove `tf.constant()` from layer initialization (use direct computation)
- [ ] Replace `tf.placeholder()` with direct function arguments
- [ ] Replace `tf.Session() + sess.run()` with direct function calls or `@tf.function`
- [ ] Replace `tf.variable_scope()` with module classes
- [ ] Replace `tf.get_variable()` with direct `tf.Variable()`
- [ ] Replace `tf.app.flags` with `argparse`
- [ ] Run `verify_tf2_implementation.py` equivalent for your models
- [ ] Use `tf.GradientTape()` for custom training loops
- [ ] Test numerical equivalence on identical random seeds

## Performance Notes

TensorFlow 2.x with eager execution:
- **Pros**: Easier debugging, more Pythonic, compatible with modern TF ecosystem
- **Cons**: Slightly slower than graph-mode TF 1.x (can be mitigated with `@tf.function`)

To optimize performance, decorate the training step with `@tf.function`:

```python
@tf.function
def train_step(cell, inputs, targets):
    with tf.GradientTape() as tape:
        loss = compute_loss(...)
    grads = tape.gradient(...)
    return loss
```

## Numerical Precision

The TF 2.x version should produce **numerically identical** results to TF 1.x when:
1. Using identical random seeds
2. Running on the same hardware (CPU/GPU)
3. Using the same batch sizes and sequence lengths

Small differences (< 1e-6 relative) are expected due to:
- Different floating-point operation ordering in graph construction
- Slight variations in kernel implementations between versions
- GPU non-determinism (use `tf.config.experimental.enable_op_determinism()` if needed)

## Troubleshooting

### Issue: NaN or Inf values appear during training

Check:
1. Learning rate is appropriate (try lower: 0.001 or 0.0001)
2. Input data is properly normalized
3. Verify with `verify_tf2_implementation.py` - if tests pass, issue is likely in data

### Issue: Gradients are zero

Check:
1. Variables are explicitly part of the trained model
2. `with tf.GradientTape() as tape:` context is properly used
3. Loss computation path includes variables (check with `tape.watched_variables()`)

### Issue: Memory usage is high

Solutions:
1. Reduce batch size with `--n_batch`
2. Reduce sequence length in data generation
3. Use `@tf.function` to reduce Python overhead
4. Run validation on CPU only (small batch size)

## References

- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [GradientTape Documentation](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
- [Custom Gradients](https://www.tensorflow.org/guide/custom_gradients)
- Original e-prop paper: "A solution to the learning dilemma for recurrent networks of spiking neurons" (2021)

## Contact

For questions about the e-prop implementation or migration issues, refer to the original repository documentation.
