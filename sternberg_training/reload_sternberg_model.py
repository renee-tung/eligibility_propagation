import argparse
import datetime
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf

from tools import generate_sternberg_task_data

tf.disable_v2_behavior()

from models import EligALIF, exp_convolve
from tools import generate_click_task_data


def load_model_spec(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_data_dict(
    batch_size,
    settings,
    inputs, targets, labels, loads
):
    # used for obtaining a new randomly generated batch of examples
    batch_inputs, batch_targets, batch_labels, batch_loads = generate_sternberg_task_data(batch_size=batch_size, settings=settings)
    batch_inputs = np.transpose(batch_inputs, (0, 2, 1))  # (batch, T, channels)
    return {inputs: batch_inputs, 
            targets: batch_targets, 
            labels: batch_labels, 
            loads: batch_loads}

def get_trial_scenarios(spec):
    """
    Define trial families here.
    Add/remove items in this list to run multiple trial types in one restore pass.
    """
    default_t_spacing = int(spec.get('t_cue_spacing', 150))
    default_input_f0 = float(spec.get('input_f0', 40.0 / 1000.0))

    return [
        {
            'name': 'training_task',
            'settings': {'T': 250, # trial duration (in steps)
                         'stim_on': 50, # input stim onset (in steps)
                         'stim_dur': 25, # input stim duration (in steps)
                         'delay': 10, # delay b/w sample and test (in steps)
                         'load': 1, # initialize with load 1, but will alternate with 3
                         'p_low': 0.5, # probability of low load trial (load 1) vs high load trial (load 3)
                        'jitter_onset': 0,
                        'jitter_delay': 0,
                        }
        },
        {
            'name': 'delay50',
            'settings': {'T': 300, # trial duration (in steps)
                         'stim_on': 50, # input stim onset (in steps)
                         'stim_dur': 25, # input stim duration (in steps)
                         'delay': 50, # delay b/w sample and test (in steps)
                         'load': 1, # initialize with load 1, but will alternate with 3
                         'p_low': 0.5, # probability of low load trial (load 1) vs high load trial (load 3)
                        'jitter_onset': 0,
                        'jitter_delay': 0,
                        }
        },
        {
            'name': 'delay200',
            'settings': {'T': 450, # trial duration (in steps)
                         'stim_on': 50, # input stim onset (in steps)
                         'stim_dur': 25, # input stim duration (in steps)
                         'delay': 200, # delay b/w sample and test (in steps)
                         'load': 1, # initialize with load 1, but will alternate with 3
                         'p_low': 0.5, # probability of low load trial (load 1) vs high load trial (load 3)
                        'jitter_onset': 0,
                        'jitter_delay': 0,
                        }
        },
    ]

# def get_data_dict(
#     batch_size,
#     input_spikes_ph,
#     input_nums_ph,
#     target_nums_ph,
#     n_in,
#     t_cue_spacing,
#     input_f0,
#     p_group,
#     n_cues,
#     t_cue,
#     recall_duration,
#     n_input_symbols,
#     seq_len_override,
# ):
#     if seq_len_override is None:
#         seq_len = int(t_cue_spacing * n_cues + 1200)
#     else:
#         seq_len = int(seq_len_override)

#     spk_data, in_nums, target_data, _ = generate_click_task_data(
#         batch_size=batch_size,
#         seq_len=seq_len,
#         n_neuron=n_in,
#         recall_duration=recall_duration,
#         p_group=p_group,
#         t_cue=t_cue,
#         n_cues=n_cues,
#         t_interval=t_cue_spacing,
#         f0=input_f0,
#         n_input_symbols=n_input_symbols,
#     )
#     return {
#         input_spikes_ph: spk_data,
#         input_nums_ph: in_nums,
#         target_nums_ph: target_data,
#     }


# def get_trial_scenarios(spec):
#     """
#     Define trial families here.
#     Add/remove items in this list to run multiple trial types in one restore pass.
#     """
#     default_t_spacing = int(spec.get('t_cue_spacing', 150))
#     default_input_f0 = float(spec.get('input_f0', 40.0 / 1000.0))

#     return [
#         {
#             'name': 'baseline',
#             'p_group': 0.3,
#             'n_cues': 7,
#             't_cue': 100,
#             't_cue_spacing': default_t_spacing,
#             'recall_duration': 150,
#             'n_input_symbols': 4,
#             'input_f0': default_input_f0,
#             'seq_len': None,
#         },
#         {
#             'name': 'ambiguous_balance',
#             'p_group': 0.5,
#             'n_cues': 7,
#             't_cue': 100,
#             't_cue_spacing': default_t_spacing,
#             'recall_duration': 150,
#             'n_input_symbols': 4,
#             'input_f0': default_input_f0,
#             'seq_len': None,
#         },
#         {
#             'name': 'long_memory',
#             'p_group': 0.3,
#             'n_cues': 11,
#             't_cue': 100,
#             't_cue_spacing': int(default_t_spacing * 1.5),
#             'recall_duration': 200,
#             'n_input_symbols': 4,
#             'input_f0': default_input_f0,
#             'seq_len': None,
#         },
#     ]


def build_graph(spec, settings, batch_size=None):
    n_in = int(spec['n_in'])
    n_regular = int(spec['n_regular'])
    n_adaptive = int(spec['n_adaptive'])
    n_neurons = n_regular + n_adaptive

    tau_v = float(spec['tau_v'])
    tau_a = float(spec['tau_a'])
    tau_out = float(spec.get('tau_out', tau_v))
    thr = float(spec['thr'])
    dt = int(spec['dt'])
    n_ref = int(spec['n_ref'])
    dampening_factor = float(spec['dampening_factor'])
    eprop = bool(spec.get('eprop', False))

    t_cue_spacing = int(spec.get('t_cue_spacing', 150))
    input_f0 = float(spec.get('input_f0', 40.0 / 1000.0))

    # Dynamic batch dimension allows feeding different batch sizes without rebuilding the graph.
    inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, n_in), name='Inputs')  # MAIN input placeholder
    targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, None), name='Targets')  # Target output placeholder
    labels = tf.placeholder(dtype=tf.int64, shape=(batch_size,), name='Labels') # Same/different label for each trial in batch
    loads = tf.placeholder(dtype=tf.int64, shape=(batch_size,), name='Loads') # Load (number of items to remember) for each trial in batch
    # input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, n_in), name='InputSpikes')
    # input_nums = tf.placeholder(dtype=tf.float32, shape=(None, None), name='InputNums')
    # target_nums = tf.placeholder(dtype=tf.int64, shape=(None, None), name='TargetNums')

    rhos = np.exp(-dt / tau_a)
    beta_a = 1.7 * (1 - rhos) / (1 - np.exp(-1 / tau_v))
    beta = np.concatenate([np.zeros(n_regular), beta_a * np.ones(n_adaptive)])

    with tf.variable_scope('CellDefinition'):
        cell = EligALIF(
            n_in=n_in,
            n_rec=n_neurons,
            tau=tau_v,
            beta=beta,
            thr=thr,
            dt=dt,
            tau_adaptation=tau_a,
            dampening_factor=dampening_factor,
            stop_z_gradients=eprop,
            n_refractory=n_ref,
        )

    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    # outputs, _ = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
    z, _ = outputs

    decay = np.exp(-dt / tau_out)
    filtered_z = exp_convolve(z, decay)

    with tf.name_scope('OutputComputation'):
        # w_out = tf.get_variable(name='out_weight', shape=[n_neurons, 2])
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, 1])
        out = tf.einsum('btj,jk->btk', filtered_z, w_out)
        # output_logits = out[:, -t_cue_spacing:]

    out_2d = tf.squeeze(out, axis=-1)
    resp_onsets = (settings['stim_on'] + loads * settings['stim_dur'] + \
        settings['stim_dur'] + settings['delay'] + 10) # calculate response onset time for each trial in batch based on load
    T = tf.shape(out_2d, out_type=tf.int64)[1]
    batch_size = tf.shape(out_2d)[0]
    
    time_idx = tf.range(T)[None, :] # Create time indices: shape (1, T)
    resp_onsets = (settings['stim_on'] + loads * settings['stim_dur'] + \
        settings['stim_dur'] + settings['delay'] + 10) # calculate response onset time for each trial in batch based on load
    resp_onsets_exp = resp_onsets[:, None] # Expand resp_onsets: shape (batch, 1)
    resp_offsets_exp = resp_onsets_exp + settings['stim_dur']
    time_mask = tf.logical_and(
        time_idx >= resp_onsets_exp,
        time_idx < resp_offsets_exp
    )  # shape: (batch, T)
    # pdb.set_trace()
    masked_out_max = tf.where(time_mask, out_2d, tf.fill(tf.shape(out_2d), -1e9)) # For max: mask out invalid with very small value
    masked_out_min = tf.where(time_mask, out_2d, tf.fill(tf.shape(out_2d), 1e9)) # For min: mask out invalid with very large value

    # Reduce over time
    max_vals = tf.reduce_max(masked_out_max, axis=1)  # (batch,)
    min_vals = tf.reduce_min(masked_out_min, axis=1)  # (batch,)

    # --- correct mask shape ---
    match_mask = tf.equal(labels, 1)  # (batch,)

    # --- compute performance ---
    perfs = tf.where(
        match_mask,
        tf.cast(max_vals > 0.7, tf.float32),
        tf.cast(min_vals < -0.7, tf.float32)
    )
    
    accuracy = tf.reduce_mean(perfs)
    
    # y_predict = tf.argmax(tf.reduce_mean(output_logits, axis=1), axis=1)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(target_nums[:, -1], y_predict), dtype=tf.float32))

    var_list = [cell.w_in_var, cell.w_rec_var, w_out]
    saver = tf.train.Saver(var_list=var_list)

    endpoints = {
        'inputs': inputs,
        'targets': targets,
        'loads': loads,
        'labels': labels,
        'z': z,
        'outs': out_2d,
        'accuracy': accuracy,
        # 'input_f0': input_f0,
        # 'n_in': n_in,
    }

    return saver, endpoints


def main():
    parser = argparse.ArgumentParser(description='Restore a trained evidence accumulation model and run fresh trials.')
    parser.add_argument('--model_spec', required=True, help='Path to model_spec.json from a saved run')
    parser.add_argument('--n_batch', type=int, default=64, help='Batch size for fresh trials')
    parser.add_argument('--n_eval_batches', type=int, default=4, help='Number of new random batches to run')
    parser.add_argument('--save_npz', action='store_true', help='Save trial inputs/spikes/predictions as NPZ')
    parser.add_argument('--output_dir', default='restored_runs', help='Output directory for restored trial artifacts')
    args = parser.parse_args()

    spec = load_model_spec(args.model_spec)
    trial_scenarios = get_trial_scenarios(spec)

    # saver, ep = build_graph(spec, settings, args.n_batch)

    ckpt_prefix = spec.get('checkpoint_prefix')
    if ckpt_prefix is None:
        raise ValueError('model_spec.json is missing checkpoint_prefix')

    save_root = os.path.abspath(args.output_dir)
    os.makedirs(save_root, exist_ok=True)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = 'restored_eprop_{}_batch{}_{}'.format('true' if bool(spec.get('eprop', False)) else 'false', args.n_batch, ts)
    run_dir = os.path.join(save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    scenario_summaries = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, ckpt_prefix)

        for scenario in trial_scenarios:
            all_acc = []
            all_outs = []
            all_targets = []
            all_spikes = []
            all_inputs = []
            
            settings = scenario['settings']
            saver, ep = build_graph(spec, scenario['settings'], args.n_batch)
            saver.restore(sess, ckpt_prefix)

            for _ in range(args.n_eval_batches):
                batch_size = int(scenario.get('batch_size', args.n_batch))
                feed = get_data_dict(
                    batch_size=batch_size,
                    inputs=ep['inputs'],
                    targets=ep['targets'],
                    labels=ep['labels'],
                    loads=ep['loads'],
                    settings=scenario['settings'],
                    # input_spikes_ph=ep['input_spikes'],
                    # input_nums_ph=ep['input_nums'],
                    # target_nums_ph=ep['target_nums'],
                    # n_in=ep['n_in'],
                    # t_cue_spacing=scenario['t_cue_spacing'],
                    # input_f0=scenario['input_f0'],
                    # p_group=scenario['p_group'],
                    # n_cues=scenario['n_cues'],
                    # t_cue=scenario['t_cue'],
                    # recall_duration=scenario['recall_duration'],
                    # n_input_symbols=scenario['n_input_symbols'],
                    # seq_len_override=scenario['seq_len'],
                )
                
                
                acc, outs, z_vals, tgt = sess.run(
                    [ep['accuracy'], ep['outs'], ep['z'], ep['targets']],
                    feed_dict=feed,
                )

                all_acc.append(float(acc))
                all_outs.append(outs)
                all_targets.append(tgt[:, -1])
                if args.save_npz:
                    all_spikes.append(z_vals)
                    all_inputs.append(feed[ep['input_spikes']])

            scenario_summary = {
                'name': scenario['name'],
                'trial_generation': scenario,
                'mean_accuracy': float(np.mean(all_acc)),
                'std_accuracy': float(np.std(all_acc)),
                'batch_accuracies': all_acc,
            }
            scenario_summaries.append(scenario_summary)

            if args.save_npz:
                npz_path = os.path.join(run_dir, 'restore_trials_{}.npz'.format(scenario['name']))
                np.savez_compressed(
                    npz_path,
                    outputs=np.array(all_outs),
                    targets=np.array(all_targets),
                    spikes=np.array(all_spikes),
                    input_spikes=np.array(all_inputs),
                )
                print('Saved restore trials NPZ to {}'.format(npz_path))

    summary = {
        'model_spec': os.path.abspath(args.model_spec),
        'checkpoint_prefix': ckpt_prefix,
        'eprop': bool(spec.get('eprop', False)),
        'n_batch': args.n_batch,
        'n_eval_batches': args.n_eval_batches,
        'scenario_summaries': scenario_summaries,
    }

    summary_path = os.path.join(run_dir, 'restore_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Restore summary saved to {}'.format(summary_path))
    for scenario_summary in scenario_summaries:
        print('[{}] mean accuracy over fresh trials: {:.4f} +- {:.4f}'.format(
            scenario_summary['name'],
            scenario_summary['mean_accuracy'],
            scenario_summary['std_accuracy'],
        ))


if __name__ == '__main__':
    main()
