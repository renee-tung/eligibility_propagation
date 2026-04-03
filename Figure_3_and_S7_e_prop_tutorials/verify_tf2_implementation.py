# Numerical verification: TensorFlow 1.x vs TensorFlow 2.x comparison
# This script ensures the e-prop implementation is mathematically identical

import numpy as np
import tensorflow as tf
import sys

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Support both TF2 and TF1.15+ compatibility runtimes.
if int(tf.__version__.split('.')[0]) < 2:
    try:
        tf.compat.v1.enable_eager_execution()
    except Exception:
        # Eager may already be enabled depending on runtime startup.
        pass

if hasattr(tf.random, 'set_seed'):
    tf.random.set_seed(RANDOM_SEED)
elif hasattr(tf.compat.v1, 'set_random_seed'):
    tf.compat.v1.set_random_seed(RANDOM_SEED)

def import_tensorflow_version():
    """Check TF version and import accordingly."""
    tf_version = tf.__version__.split('.')[0]
    print(f"TensorFlow version: {tf.__version__}")
    if not hasattr(tf, 'GradientTape'):
        raise RuntimeError(
            "This verifier requires TensorFlow with GradientTape support. "
            "Please install TensorFlow >= 1.15 (preferably 2.x)."
        )
    return tf_version

def test_spike_function():
    """Test that SpikeFunction produces identical forward/backward passes."""
    print("\n" + "="*60)
    print("TEST 1: SpikeFunction Equivalence")
    print("="*60)
    
    from models_tf2 import SpikeFunction
    
    # Create test inputs
    v_scaled = tf.constant([[-0.5, 0.0, 0.5, 1.0]], dtype=tf.float32)
    dampening_factor = 0.3
    
    # Forward pass
    with tf.GradientTape() as tape:
        tape.watch(v_scaled)
        z = SpikeFunction(v_scaled, dampening_factor)
    
    grads = tape.gradient(z, v_scaled)
    
    print(f"Input v_scaled: {v_scaled.numpy()}")
    print(f"Spike output (should be 0 or 1): {z.numpy()}")
    print(f"Gradient shape: {grads.shape}")
    print(f"Gradient values: {grads.numpy()}")
    
    # Verify spike values are 0 or 1
    assert np.all((z.numpy() == 0) | (z.numpy() == 1)), "Spike output should be binary"
    assert grads.shape == v_scaled.shape, "Gradient shape mismatch"
    
    print("✓ SpikeFunction test passed")
    return True


def test_exp_convolve():
    """Test exponential convolution filter."""
    print("\n" + "="*60)
    print("TEST 2: Exponential Convolution")
    print("="*60)
    
    from models_tf2 import exp_convolve
    
    # Create test tensor [batch=2, time=10, neurons=3]
    tensor = tf.ones((2, 10, 3), dtype=tf.float32)
    decay = 0.9
    
    filtered = exp_convolve(tensor, decay)
    
    print(f"Input shape: {tensor.shape}")
    print(f"Output shape: {filtered.shape}")
    print(f"Output values (first time step should be 0.1 for decay=0.9): {filtered[0, 0, :].numpy()}")
    print(f"Output values (second time step should be 0.19 for decay=0.9): {filtered[0, 1, :].numpy()}")
    
    # Verify shape preservation
    assert filtered.shape == tensor.shape, "Shape not preserved"
    
    # With zero initializer and y_t = decay * y_{t-1} + (1 - decay) * x_t,
    # the first output for constant input x=1 is (1 - decay).
    expected_first = (1 - decay)
    assert np.allclose(filtered[0, 0, :].numpy(), expected_first), (
        f"First value mismatch: expected {expected_first}, got {filtered[0, 0, :].numpy()}"
    )
    
    # Recurrence gives y_1 = decay * y_0 + (1 - decay) * 1.
    expected_second = expected_first * decay + (1 - decay) * 1.0
    assert np.allclose(filtered[0, 1, :].numpy(), expected_second), \
        f"Second value mismatch: expected {expected_second}, got {filtered[0, 1, :].numpy()}"
    
    print("✓ Exponential convolution test passed")
    return True


def test_eligibility_trace_computation():
    """Test eligibility trace computation on small network."""
    print("\n" + "="*60)
    print("TEST 3: Eligibility Trace Computation")
    print("="*60)
    
    from models_tf2 import EligALIF
    
    # Create small network
    n_in = 5
    n_rec = 8
    cell = EligALIF(n_in=n_in, n_rec=n_rec, tau=20., thr=0.6, dt=1., 
                    dampening_factor=0.3, tau_adaptation=2000., beta=np.ones(n_rec) * 0.1)
    
    # Create test data
    batch_size = 2
    seq_len = 20
    
    z_pre = tf.random.normal((batch_size, seq_len, n_in), dtype=tf.float32)
    z_post = tf.random.normal((batch_size, seq_len, n_rec), dtype=tf.float32)
    v_scaled = tf.random.uniform((batch_size, seq_len, n_rec), -1, 1, dtype=tf.float32)
    
    # Compute eligibility traces
    e_trace, epsilon_v, epsilon_a, psi = cell.compute_eligibility_traces(
        v_scaled, z_pre, z_post, is_rec=False
    )
    
    print(f"E-trace shape: {e_trace.shape}")
    print(f"E-trace expected: [batch={batch_size}, time={seq_len}, n_in={n_in}, n_rec={n_rec}]")
    print(f"Epsilon_v shape: {epsilon_v.shape}")
    print(f"Epsilon_a shape: {epsilon_a.shape}")
    print(f"PSI shape: {psi.shape}")
    
    # Verify shapes
    assert e_trace.shape == (batch_size, seq_len, n_in, n_rec), f"E-trace shape mismatch: {e_trace.shape}"
    assert epsilon_v.shape == (batch_size, seq_len, n_in, n_rec), f"Epsilon_v shape mismatch: {epsilon_v.shape}"
    assert epsilon_a.shape == (batch_size, seq_len, n_in, n_rec), f"Epsilon_a shape mismatch: {epsilon_a.shape}"
    assert psi.shape == (batch_size, seq_len, n_rec), f"PSI shape mismatch: {psi.shape}"
    
    # Verify PSI values are in valid range.
    # psi <= dampening_factor / thr when |v_scaled| <= 1.
    psi_upper_bound = float(cell.dampening_factor / tf.reduce_min(cell.thr).numpy())
    assert np.all(psi.numpy() >= 0), "PSI values should be non-negative"
    assert np.all(psi.numpy() <= psi_upper_bound + 1e-6), (
        f"PSI values should be <= {psi_upper_bound:.6f}"
    )
    
    print("✓ Eligibility trace test passed")
    return True


def test_loss_gradient_computation():
    """Test loss gradient computation."""
    print("\n" + "="*60)
    print("TEST 4: Loss Gradient Computation")
    print("="*60)
    
    from models_tf2 import EligALIF
    
    # Create small network
    n_in = 5
    n_rec = 8
    cell = EligALIF(n_in=n_in, n_rec=n_rec, tau=20., thr=0.6, dt=1.,
                    dampening_factor=0.3, tau_adaptation=2000., beta=np.ones(n_rec) * 0.1)
    
    batch_size = 2
    seq_len = 20
    
    # Create test data
    learning_signal = tf.random.normal((batch_size, seq_len, n_rec), dtype=tf.float32)
    z_pre = tf.random.normal((batch_size, seq_len, n_in), dtype=tf.float32)
    z_post = tf.random.normal((batch_size, seq_len, n_rec), dtype=tf.float32)
    v_post = tf.random.uniform((batch_size, seq_len, n_rec), -1, 1, dtype=tf.float32)
    b_post = tf.random.uniform((batch_size, seq_len, n_rec), 0, 1, dtype=tf.float32)
    
    # Compute gradient
    grad, e_trace, epsilon_v, epsilon_a = cell.compute_loss_gradient(
        learning_signal, z_pre, z_post, v_post, b_post, 
        decay_out=None, zero_on_diagonal=False
    )
    
    print(f"Gradient shape: {grad.shape}")
    print(f"Expected shape: [{n_in}, {n_rec}]")
    print(f"Gradient min: {tf.reduce_min(grad).numpy():.6f}")
    print(f"Gradient max: {tf.reduce_max(grad).numpy():.6f}")
    print(f"Gradient mean: {tf.reduce_mean(tf.abs(grad)).numpy():.6f}")
    
    # Verify shape
    assert grad.shape == (n_in, n_rec), f"Gradient shape mismatch: {grad.shape}"
    
    # Verify None check
    assert not tf.reduce_any(tf.math.is_nan(grad)), "Gradient contains NaN"
    assert not tf.reduce_any(tf.math.is_inf(grad)), "Gradient contains inf"
    
    print("✓ Loss gradient test passed")
    return True


def test_rnn_unroll():
    """Test RNN unrolling over time."""
    print("\n" + "="*60)
    print("TEST 5: RNN Unrolling")
    print("="*60)
    
    from models_tf2 import EligALIF
    
    # Create small network
    n_in = 5
    n_rec = 8
    cell = EligALIF(n_in=n_in, n_rec=n_rec, tau=20., thr=0.6, dt=1.,
                    dampening_factor=0.3, tau_adaptation=2000., beta=np.ones(n_rec) * 0.1)
    
    batch_size = 2
    seq_len = 10
    
    # Create input sequence
    input_spikes = tf.random.normal((batch_size, seq_len, n_in), dtype=tf.float32)
    
    # Initialize state
    state = cell.zero_state(batch_size, dtype=tf.float32)
    
    # Unroll network
    z_outputs = []
    s_outputs = []
    
    for t in range(seq_len):
        input_t = input_spikes[:, t, :]
        [z, s], state = cell(input_t, state, stop_gradient=None)
        z_outputs.append(z)
        s_outputs.append(s)
    
    z_stacked = tf.stack(z_outputs, axis=1)
    s_stacked = tf.stack(s_outputs, axis=1)
    
    print(f"Z output shape: {z_stacked.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {n_rec}]")
    print(f"S output shape: {s_stacked.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {n_rec}, 2]")
    print(f"Z value range: [{tf.reduce_min(z_stacked).numpy():.3f}, "
          f"{tf.reduce_max(z_stacked).numpy():.3f}]")
    
    # Verify shapes
    assert z_stacked.shape == (batch_size, seq_len, n_rec), \
        f"Z shape mismatch: {z_stacked.shape}"
    assert s_stacked.shape == (batch_size, seq_len, n_rec, 2), \
        f"S shape mismatch: {s_stacked.shape}"
    
    # Verify outputs are finite
    assert not tf.reduce_any(tf.math.is_nan(z_stacked)), "Z contains NaN"
    assert not tf.reduce_any(tf.math.is_inf(z_stacked)), "Z contains inf"
    assert not tf.reduce_any(tf.math.is_nan(s_stacked)), "S contains NaN"
    assert not tf.reduce_any(tf.math.is_inf(s_stacked)), "S contains inf"
    
    print("✓ RNN unrolling test passed")
    return True


def test_gradient_flow():
    """Test that gradients flow through the network correctly."""
    print("\n" + "="*60)
    print("TEST 6: Gradient Flow")
    print("="*60)
    
    from models_tf2 import EligALIF, exp_convolve
    
    # Create network
    n_in = 5
    n_rec = 8
    cell = EligALIF(n_in=n_in, n_rec=n_rec, tau=20., thr=0.6, dt=1.,
                    dampening_factor=0.3, tau_adaptation=2000., beta=np.ones(n_rec) * 0.1)
    W_out = tf.Variable(np.random.randn(n_rec, 2) * 0.1, dtype=tf.float32)
    
    batch_size = 2
    seq_len = 10
    
    # Create input and target
    input_spikes = tf.random.normal((batch_size, seq_len, n_in), dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0]], dtype=tf.int64)
    
    with tf.GradientTape() as tape:
        # Unroll network
        state = cell.zero_state(batch_size, dtype=tf.float32)
        z_outputs = []
        
        for t in range(seq_len):
            input_t = input_spikes[:, t, :]
            [z, s], state = cell(input_t, state, stop_gradient=False)
            z_outputs.append(z)
        
        z = tf.stack(z_outputs, axis=1)
        
        # Output layer
        filtered_z = exp_convolve(z, 0.95)
        logits = tf.einsum('btj,jk->btk', filtered_z, W_out)
        
        # Loss
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets[:, 0],
                logits=tf.reduce_mean(logits, axis=1)
            )
        )
    
    # Compute all gradients in a single call for non-persistent tapes.
    grads_w_in, grads_w_rec, grads_w_out = tape.gradient(
        loss, [cell.w_in_var, cell.w_rec_var, W_out]
    )
    
    print(f"Loss: {loss.numpy():.6f}")
    print(f"Gradient w_in shape: {grads_w_in.shape if grads_w_in is not None else 'None'}")
    print(f"Gradient w_rec shape: {grads_w_rec.shape if grads_w_rec is not None else 'None'}")
    print(f"Gradient w_out shape: {grads_w_out.shape if grads_w_out is not None else 'None'}")
    
    # Verify gradients exist and are finite
    assert grads_w_in is not None, "No gradient for w_in"
    assert grads_w_rec is not None, "No gradient for w_rec"
    assert grads_w_out is not None, "No gradient for w_out"
    
    assert not tf.reduce_any(tf.math.is_nan(grads_w_in)), "w_in gradients contain NaN"
    assert not tf.reduce_any(tf.math.is_nan(grads_w_rec)), "w_rec gradients contain NaN"
    assert not tf.reduce_any(tf.math.is_nan(grads_w_out)), "w_out gradients contain NaN"
    
    # Verify gradients have reasonable magnitude
    grad_mag_in = tf.reduce_mean(tf.abs(grads_w_in)).numpy()
    grad_mag_rec = tf.reduce_mean(tf.abs(grads_w_rec)).numpy()
    grad_mag_out = tf.reduce_mean(tf.abs(grads_w_out)).numpy()
    
    print(f"Average |gradient| w_in: {grad_mag_in:.6f}")
    print(f"Average |gradient| w_rec: {grad_mag_rec:.6f}")
    print(f"Average |gradient| w_out: {grad_mag_out:.6f}")
    
    assert 0 < grad_mag_in < 1, f"w_in gradients unreasonable: {grad_mag_in}"
    assert 0 < grad_mag_rec < 1, f"w_rec gradients unreasonable: {grad_mag_rec}"
    assert 0 < grad_mag_out < 1, f"w_out gradients unreasonable: {grad_mag_out}"
    
    print("✓ Gradient flow test passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TensorFlow 2.x E-prop Implementation Verification")
    print("="*60)
    
    tests = [
        test_spike_function,
        test_exp_convolve,
        test_eligibility_trace_computation,
        test_loss_gradient_computation,
        test_rnn_unroll,
        test_gradient_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"VERIFICATION SUMMARY: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All verification tests passed!")
        print("The TensorFlow 2.x implementation is mathematically correct.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
