# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass
#
# TensorFlow 2.x compatible version - preserves exact mathematical operations

import datetime
import json
import os
import socket
from time import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from tools import update_plot, generate_click_task_data
from models_tf2 import EligALIF, exp_convolve, shift_by_one_time_step


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='E-prop training on click task')
    
    # Training parameters
    parser.add_argument('--n_batch', type=int, default=64, help='batch size')
    parser.add_argument('--n_iter', type=int, default=2000, help='total number of iterations')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Base learning rate')
    parser.add_argument('--stop_crit', type=float, default=0.07, 
                        help='Stopping criterion. Stops training if error goes below this value')
    parser.add_argument('--print_every', type=int, default=10, help='Print every')
    parser.add_argument('--validate_every', type=int, default=10, help='validate every')
    
    # Training algorithm
    parser.add_argument('--eprop', action='store_true', default=False, 
                        help='Use e-prop to train network (BPTT if false)')
    parser.add_argument('--eprop_impl', type=str, default='autodiff', 
                        choices=['autodiff', 'hardcoded'],
                        help='Use tensorflow for computing e-prop updates or implement equations directly')
    parser.add_argument('--feedback', type=str, default='symmetric', 
                        choices=['random', 'symmetric'],
                        help='Use random or symmetric e-prop')
    parser.add_argument('--f_regularization_type', type=str, default='simple',
                        choices=['simple', 'online'],
                        help='Two types of firing rate regularization')
    
    # Neuron model and simulation parameters
    parser.add_argument('--tau_a', type=float, default=2000, 
                        help='model alpha - threshold decay [ms]')
    parser.add_argument('--thr', type=float, default=0.6, 
                        help='threshold at which the LSNN neurons spike')
    parser.add_argument('--tau_v', type=float, default=20, 
                        help='tau for filtered_z decay in LSNN  neurons [ms]')
    parser.add_argument('--tau_out', type=float, default=20, 
                        help='tau for filtered_z decay in output neurons [ms]')
    parser.add_argument('--reg_f', type=float, default=1, 
                        help='regularization coefficient for firing rate')
    parser.add_argument('--reg_rate', type=int, default=10, 
                        help='target firing rate for regularization [Hz]')
    parser.add_argument('--n_ref', type=int, default=5, 
                        help='Number of refractory steps [ms]')
    parser.add_argument('--dt', type=int, default=1, 
                        help='Simulation time step [ms]')
    parser.add_argument('--dampening_factor', type=float, default=0.3, 
                        help='factor that controls amplitude of pseudoderivative')
    
    # Other settings
    parser.add_argument('--do_plot', action='store_true', default=True, 
                        help='Perform plots')
    parser.add_argument('--device_placement', action='store_true', default=False, 
                        help='Log device placement')
    parser.add_argument('--save_outputs', action='store_true', default=True, 
                        help='Save run outputs to disk')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory where run artifacts are saved')
    
    return parser.parse_args()


def build_model(FLAGS):
    """Build the network model."""
    # Experiment parameters
    t_cue_spacing = 150  # distance between two consecutive cues in ms
    
    # Frequencies
    input_f0 = 40. / 1000.  # Poisson firing rate of input neurons in kHz
    regularization_f0 = FLAGS.reg_rate / 1000.  # mean target network firing frequency
    
    # Network parameters
    tau_v = FLAGS.tau_v
    thr = FLAGS.thr
    n_adaptive = 50
    n_regular = 50
    n_neurons = n_adaptive + n_regular
    decay = np.exp(-FLAGS.dt / FLAGS.tau_out)  # output layer filtered_z decay
    
    n_in = 40
    
    # Generate the cell
    tau_a = FLAGS.tau_a
    rhos = np.exp(-FLAGS.dt / tau_a)  # decay factors for adaptive threshold
    beta_a = 1.7 * (1 - rhos) / (1 - np.exp(-1 / FLAGS.tau_v))  # heuristic value
    beta = np.concatenate([np.zeros(n_regular), beta_a * np.ones(n_adaptive)])
    
    cell = EligALIF(
        n_in=n_in, 
        n_rec=n_regular + n_adaptive, 
        tau=tau_v, 
        beta=beta, 
        thr=thr,
        dt=FLAGS.dt, 
        tau_adaptation=tau_a, 
        dampening_factor=FLAGS.dampening_factor,
        stop_z_gradients=FLAGS.eprop, 
        n_refractory=FLAGS.n_ref
    )
    
    # Output weight
    W_out = tf.Variable(
        np.random.randn(n_regular + n_adaptive, 2) * 0.1,
        dtype=tf.float32,
        trainable=True,
        name='OutputWeights'
    )
    
    # Regularization coefficient (non-trainable)
    regularization_coeff = tf.Variable(
        np.ones(n_neurons) * FLAGS.reg_f, 
        dtype=tf.float32, 
        trainable=False,
        name='RegularizationCoeff'
    )
    
    return {
        'cell': cell,
        'W_out': W_out,
        'regularization_coeff': regularization_coeff,
        't_cue_spacing': t_cue_spacing,
        'input_f0': input_f0,
        'regularization_f0': regularization_f0,
        'tau_v': tau_v,
        'thr': thr,
        'n_adaptive': n_adaptive,
        'n_regular': n_regular,
        'n_neurons': n_neurons,
        'decay': decay,
        'n_in': n_in,
    }


def run_network(cell, input_spikes, model_dict):
    """Run the network for all timesteps."""
    batch_size = tf.shape(input_spikes)[0]
    seq_len = tf.shape(input_spikes)[1]
    
    state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs_z = []
    outputs_s = []
    
    # Manual RNN loop
    for t in range(input_spikes.shape[1]):
        input_t = input_spikes[:, t, :]
        [z, s], state = cell(input_t, state, stop_gradient=None)
        outputs_z.append(z)
        outputs_s.append(s)
    
    z_tensor = tf.stack(outputs_z, axis=1)  # [batch, time, n_rec]
    s_tensor = tf.stack(outputs_s, axis=1)  # [batch, time, n_rec, 2]
    
    return z_tensor, s_tensor, state


def compute_loss(z, s, target_nums, model_dict, FLAGS):
    """Compute classification and regularization loss."""
    W_out = model_dict['W_out']
    t_cue_spacing = model_dict['t_cue_spacing']
    regularization_coeff = model_dict['regularization_coeff']
    decay = model_dict['decay']
    regularization_f0 = model_dict['regularization_f0']
    n_batch = FLAGS.n_batch
    
    v, b = s[..., 0], s[..., 1]
    
    # Output computation
    filtered_z = exp_convolve(z, decay)
    
    if FLAGS.eprop and FLAGS.feedback == 'random':
        @tf.custom_gradient
        def matmul_random_feedback(filtered_z, W_out_arg, B_out_arg):
            logits = tf.einsum('btj,jk->btk', filtered_z, W_out_arg)
            def grad(dy):
                dloss_dW_out = tf.einsum('bij,bik->jk', filtered_z, dy)
                dloss_dfiltered_z = tf.einsum('bik,jk->bij', dy, B_out_arg)
                dloss_db_out = tf.zeros_like(B_out_arg)
                return [dloss_dfiltered_z, dloss_dW_out, dloss_db_out]
            return logits, grad
        
        b_out_vals = rd.randn(model_dict['n_regular'] + model_dict['n_adaptive'], 2)
        B_out = tf.constant(b_out_vals, dtype=tf.float32, name='feedback_weights')
        out = matmul_random_feedback(filtered_z, W_out, B_out)
    else:
        out = tf.einsum('btj,jk->btk', filtered_z, W_out)
    
    # Use only the last t_cue_spacing outputs for classification
    output_logits = out[:, -t_cue_spacing:]
    
    # Classification loss
    tiled_targets = tf.tile(target_nums[:, np.newaxis, -1], (1, t_cue_spacing))
    loss_cls = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tiled_targets,
            logits=output_logits
        )
    )
    
    # Accuracy computation
    y_predict = tf.argmax(tf.reduce_mean(output_logits, axis=1), axis=1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(target_nums[:, -1], y_predict), dtype=tf.float32)
    )
    recall_errors = 1 - accuracy
    
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / FLAGS.dt
    
    if FLAGS.f_regularization_type == "simple":
        loss_reg_f = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)
    else:
        # Online regularization
        shp = tf.shape(z)
        z_single_agent = tf.concat(tf.unstack(z, axis=0), axis=0)
        spike_count_single_agent = tf.cumsum(z_single_agent, axis=0)
        timeline_single_agent = tf.cast(tf.range(shp[0] * shp[1]), tf.float32)
        running_av = spike_count_single_agent / (timeline_single_agent + 1)[:, None] / FLAGS.dt
        running_av = tf.stack(tf.split(running_av, FLAGS.n_batch), axis=0)
        loss_reg_f = tf.square(running_av - regularization_f0)
        loss_reg_f = tf.reduce_sum(tf.reduce_mean(loss_reg_f, axis=1) * regularization_coeff)
    
    total_loss = loss_reg_f + loss_cls
    
    return {
        'total_loss': total_loss,
        'loss_cls': loss_cls,
        'loss_reg_f': loss_reg_f,
        'recall_errors': recall_errors,
        'av': av,
        'accuracy': accuracy,
        'filtered_z': filtered_z,
        'output_logits': output_logits,
        'out': out,
    }


def unroll_network(cell, input_spikes, model_dict, FLAGS, tape=None):
    """Unroll network and compute loss within gradient tape context if provided."""
    z, s, _ = run_network(cell, input_spikes, model_dict)
    v, b = s[..., 0], s[..., 1]
    
    losses = compute_loss(z, s, model_dict.get('target_nums'), model_dict, FLAGS)
    
    return z, s, v, b, losses


def train_step(cell, W_out, regularization_coeff, input_spikes, target_nums, model_dict, FLAGS): 
    """Single training step."""
    model_dict['target_nums'] = target_nums
    cell_vars = cell.trainable_variables
    output_vars = [W_out]
    all_trainable_vars = cell_vars + output_vars
    
    with tf.GradientTape() as tape:
        z, s, v, b, losses = unroll_network(cell, input_spikes, model_dict, FLAGS, tape)
        total_loss = losses['total_loss']
    
    # Compute gradients
    grads = tape.gradient(total_loss, all_trainable_vars)
    
    # Apply gradients
    for var, grad in zip(all_trainable_vars, grads):
        if grad is not None:
            var.assign_sub(FLAGS.learning_rate * grad)
    
    return losses, z, s, v, b


def get_data_dict(batch_size, model_dict):
    """Generate a new batch of click task data."""
    t_cue_spacing = model_dict['t_cue_spacing']
    input_f0 = model_dict['input_f0']
    n_in = model_dict['n_in']
    
    seq_len = int(t_cue_spacing * 7 + 1200)
    spk_data, in_nums, target_data, _ = generate_click_task_data(
        batch_size=batch_size, 
        seq_len=seq_len, 
        n_neuron=n_in, 
        recall_duration=150,
        p_group=0.3, 
        t_cue=100, 
        n_cues=7, 
        t_interval=t_cue_spacing, 
        f0=input_f0,
        n_input_symbols=4
    )
    return spk_data, in_nums, target_data


def main():
    FLAGS = parse_args()
    
    print("\n" + "="*60)
    print("TensorFlow 2.x E-prop Training")
    print("="*60)
    
    start_time = datetime.datetime.now()
    
    # Build model
    model_dict = build_model(FLAGS)
    cell = model_dict['cell']
    W_out = model_dict['W_out']
    regularization_coeff = model_dict['regularization_coeff']
    
    # Setup for plotting
    if FLAGS.do_plot:
        plt.ion()
        if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':
            n_subplots = 7 - int(model_dict['n_regular'] == 0) - int(model_dict['n_adaptive'] == 0)
        else:
            n_subplots = 4
        fig, ax_list = plt.subplots(n_subplots, figsize=(5.9, 6))
        fig.canvas.set_window_title(socket.gethostname())
    
    # Create output directory
    if FLAGS.save_outputs:
        os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Training loop
    validation_loss_list = []
    validation_error_list = []
    training_time_list = []
    n_iter_list = []
    
    t_train = 0
    t_ref = time()
    
    print(f"Training for {FLAGS.n_iter} iterations...")
    print(f"Configuration: {'e-prop' if FLAGS.eprop else 'BPTT'}, "
          f"feedback={FLAGS.feedback}, reg_type={FLAGS.f_regularization_type}\n")
    
    for k_iter in range(FLAGS.n_iter):
        # Validation
        if np.mod(k_iter, FLAGS.validate_every) == 0:
            t0 = time()
            spk_data, in_nums, target_data = get_data_dict(FLAGS.n_batch, model_dict)
            
            # Run validation (without gradients)
            with tf.device('/CPU:0'):  # Force CPU for validation to save GPU memory
                z_val, s_val, _ = run_network(cell, spk_data, model_dict)
                model_dict['target_nums'] = target_data
                losses_val = compute_loss(z_val, s_val, target_data, model_dict, FLAGS)
            
            validation_loss_list.append(float(losses_val['total_loss']))
            validation_error_list.append(float(losses_val['recall_errors']))
            t_run = time() - t0
            
            if np.mod(k_iter, FLAGS.print_every) == 0:
                print(f"Iter {k_iter:5d}: loss = {losses_val['total_loss']:.6f}, "
                      f"error = {losses_val['recall_errors']:.6f}, "
                      f"val_time = {t_run:.3f}s")
            
            # Check stopping criterion
            if losses_val['total_loss'] < FLAGS.stop_crit:
                print(f"\nStopping criterion reached at iteration {k_iter}")
                break
        
        # Training step
        t0 = time()
        spk_data, in_nums, target_data = get_data_dict(FLAGS.n_batch, model_dict)
        losses_train, z_train, s_train, v_train, b_train = train_step(
            cell, W_out, regularization_coeff, spk_data, target_data, model_dict, FLAGS
        )
        t_run = time() - t0
        t_train += t_run
        
        n_iter_list.append(k_iter)
        training_time_list.append(t_train)
        
        # Plotting
        if FLAGS.do_plot and np.mod(k_iter, FLAGS.validate_every) == 0:
            try:
                plot_data = {
                    'z': z_train.numpy(),
                    'v': v_train.numpy(),
                    'b': b_train.numpy(),
                    'input_spikes': spk_data,
                    'target_nums': target_data,
                    'thr': float(model_dict['thr'].numpy()),
                }
                update_plot(fig, ax_list, model_dict, validation_loss_list, plot_data, 
                           k_iter, FLAGS.n_batch)
                plt.pause(0.001)
            except Exception as e:
                print(f"Plotting failed: {e}")
    
    # Save results
    if FLAGS.save_outputs:
        results = {
            'final_loss': float(validation_loss_list[-1]) if validation_loss_list else None,
            'final_error': float(validation_error_list[-1]) if validation_error_list else None,
            'validation_loss': [float(x) for x in validation_loss_list],
            'validation_error': [float(x) for x in validation_error_list],
            'training_time': float(t_train),
            'elapsed_time': str(datetime.datetime.now() - start_time),
            'FLAGS': {k: str(v) for k, v in vars(FLAGS).items()},
        }
        
        result_dir = os.path.join(FLAGS.output_dir, start_time.strftime('evidence_accumulation_%Y%m%d_%H%M%S'))
        os.makedirs(result_dir, exist_ok=True)
        
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {result_dir}")
    
    print(f"\nTraining completed in {datetime.datetime.now() - start_time}")


if __name__ == '__main__':
    main()
