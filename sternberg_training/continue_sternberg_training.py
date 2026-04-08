import argparse
import datetime
import json
import os
from time import time

import numpy as np
import tensorflow.compat.v1 as tf

from models import EligALIF, exp_convolve
from tools import generate_sternberg_task_data


tf.disable_v2_behavior()


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj).__name__))


def load_model_spec(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_checkpoint_prefix(spec):
    ckpt_prefix = spec.get("checkpoint_prefix")
    if not ckpt_prefix:
        raise ValueError("model_spec.json is missing checkpoint_prefix")

    if tf.io.gfile.exists(ckpt_prefix + ".index"):
        return ckpt_prefix

    ckpt_dir = os.path.dirname(ckpt_prefix)
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        return latest

    raise ValueError("No checkpoint found for prefix: {}".format(ckpt_prefix))


def compute_min_trial_length(stim_on, stim_dur, delay, max_load=3):
    # Need enough room for sample items, probe, response offset (+10), and full response window, then add extra 50.
    return int(stim_on + (max_load + 2) * stim_dur + delay + 10 + 1 + 50)


def build_task_settings(args):
    min_t = compute_min_trial_length(args.stim_on, args.stim_dur, args.delay, max_load=3)
    trial_T = int(args.trial_T) if args.trial_T is not None else min_t
    if trial_T < min_t:
        raise ValueError("trial_T={} is too short for task timing; need at least {}".format(trial_T, min_t))

    return {
        "T": trial_T,
        "stim_on": int(args.stim_on),
        "stim_dur": int(args.stim_dur),
        "delay": int(args.delay),
        "load": 1,
        "p_low": float(args.p_low),
        "jitter_onset": int(args.jitter_onset),
        "jitter_delay": int(args.jitter_delay),
    }


def get_data_dict(batch_size, settings, inputs, targets, labels, loads, jitter_onsets, jitter_delays):
    batch_inputs, batch_targets, batch_labels, batch_loads, batch_jitter_onsets, batch_jitter_delays = generate_sternberg_task_data(
        batch_size=batch_size,
        settings=settings,
    )
    batch_inputs = np.transpose(batch_inputs, (0, 2, 1))
    return {
        inputs: batch_inputs,
        targets: batch_targets,
        labels: batch_labels,
        loads: batch_loads,
        jitter_onsets: batch_jitter_onsets,
        jitter_delays: batch_jitter_delays,
    }


def build_graph(spec, settings, n_batch, learning_rate, reg_f, reg_rate_hz):
    n_in = int(spec["n_in"])
    n_regular = int(spec["n_regular"])
    n_adaptive = int(spec["n_adaptive"])
    n_neurons = n_regular + n_adaptive

    tau_v = float(spec["tau_v"])
    tau_a = float(spec["tau_a"])
    tau_out = float(spec.get("tau_out", tau_v))
    thr = float(spec["thr"])
    dt = int(spec["dt"])
    n_ref = int(spec["n_ref"])
    dampening_factor = float(spec["dampening_factor"])
    eprop = bool(spec.get("eprop", False))

    regularization_f0 = float(reg_rate_hz) / 1000.0

    inputs = tf.placeholder(dtype=tf.float32, shape=(n_batch, None, n_in), name="Inputs")
    targets = tf.placeholder(dtype=tf.float32, shape=(n_batch, None), name="Targets")
    labels = tf.placeholder(dtype=tf.int64, shape=(n_batch,), name="Labels")
    loads = tf.placeholder(dtype=tf.int64, shape=(n_batch,), name="Loads")
    jitter_onsets = tf.placeholder(dtype=tf.int64, shape=(n_batch,), name="JitterOnsets")
    jitter_delays = tf.placeholder(dtype=tf.int64, shape=(n_batch,), name="JitterDelays")

    rhos = np.exp(-dt / tau_a)
    beta_a = 1.7 * (1 - rhos) / (1 - np.exp(-1 / tau_v))
    beta = np.concatenate([np.zeros(n_regular), beta_a * np.ones(n_adaptive)])

    with tf.variable_scope("CellDefinition"):
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
    z, _ = outputs

    decay = np.exp(-dt / tau_out)
    filtered_z = exp_convolve(z, decay)

    with tf.name_scope("OutputComputation"):
        w_out = tf.get_variable(name="out_weight", shape=[n_neurons, 1])
        out = tf.einsum("btj,jk->btk", filtered_z, w_out)

    out_2d = tf.squeeze(out, axis=-1)

    with tf.name_scope("TaskLoss"):
        loss_cls = tf.reduce_mean(tf.square(targets - out_2d))

        T = tf.shape(out_2d, out_type=tf.int64)[1]
        time_idx = tf.range(T)[None, :]
        resp_onsets = (
            settings["stim_on"]
            + jitter_onsets
            + loads * settings["stim_dur"]
            + settings["stim_dur"]
            + settings["delay"]
            + jitter_delays
            + 10
        )
        resp_onsets_exp = resp_onsets[:, None]
        resp_offsets_exp = resp_onsets_exp + settings["stim_dur"]
        time_mask = tf.logical_and(time_idx >= resp_onsets_exp, time_idx < resp_offsets_exp)

        masked_out_max = tf.where(time_mask, out_2d, tf.fill(tf.shape(out_2d), -1e9))
        masked_out_min = tf.where(time_mask, out_2d, tf.fill(tf.shape(out_2d), 1e9))

        max_vals = tf.reduce_max(masked_out_max, axis=1)
        min_vals = tf.reduce_min(masked_out_min, axis=1)

        match_mask = tf.equal(labels, 1)
        perfs = tf.where(
            match_mask,
            tf.cast(max_vals > 0.7, tf.float32),
            tf.cast(min_vals < -0.7, tf.float32),
        )
        accuracy = tf.reduce_mean(perfs)
        recall_errors = 1.0 - accuracy

    with tf.name_scope("RegularizationLoss"):
        av = tf.reduce_mean(z, axis=(0, 1)) / dt
        regularization_coeff = tf.Variable(np.ones(n_neurons) * reg_f, dtype=tf.float32, trainable=False)
        loss_reg_f = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)

    with tf.name_scope("Optimization"):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        learning_rate_var = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)
        loss = loss_cls + loss_reg_f

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate_var)
        var_list = [cell.w_in_var, cell.w_rec_var, w_out]
        gradients = tf.gradients(loss, var_list)
        train_step = opt.apply_gradients(list(zip(gradients, var_list)), global_step=global_step)

    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

    tensors = {
        "inputs": inputs,
        "targets": targets,
        "labels": labels,
        "loads": loads,
        "jitter_onsets": jitter_onsets,
        "jitter_delays": jitter_delays,
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_reg": loss_reg_f,
        "accuracy": accuracy,
        "recall_errors": recall_errors,
        "global_step": global_step,
        "train_step": train_step,
        "z": z,
        "out": out_2d,
        "var_list": var_list,
    }

    return saver, tensors


def main():
    parser = argparse.ArgumentParser(
        description="Reload a trained Sternberg ALIF model and continue training on a modified Sternberg task."
    )
    parser.add_argument("--model_spec", required=True, help="Path to model_spec.json from previous run")
    parser.add_argument("--output_dir", default="results", help="Where to save continued-training runs")

    parser.add_argument("--n_batch", type=int, default=64, help="Batch size")
    parser.add_argument("--n_iter", type=int, default=1000, help="Continuation training iterations")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for continuation")
    parser.add_argument("--validate_every", type=int, default=10, help="Validation frequency")
    parser.add_argument("--print_every", type=int, default=50, help="Logging frequency")
    parser.add_argument("--stop_crit", type=float, default=0.05, help="Early-stop threshold on validation recall error")

    parser.add_argument("--delay", type=int, default=100, help="New task delay duration")
    parser.add_argument("--stim_on", type=int, default=50, help="Stimulus onset")
    parser.add_argument("--stim_dur", type=int, default=25, help="Stimulus duration")
    parser.add_argument("--trial_T", type=int, default=None, help="Total trial length; auto-computed if omitted")
    parser.add_argument("--p_low", type=float, default=0.5, help="Probability for low-load (load=1) trials")
    parser.add_argument("--jitter_onset", type=int, default=0, help="Random onset jitter")
    parser.add_argument("--jitter_delay", type=int, default=0, help="Random delay jitter")

    parser.add_argument("--reg_f", type=float, default=None, help="Override firing-rate regularization coefficient")
    parser.add_argument("--reg_rate_hz", type=float, default=None, help="Override target firing rate in Hz")

    args = parser.parse_args()

    if not (0.0 <= args.p_low <= 1.0):
        raise ValueError("p_low must be in [0, 1]")

    spec = load_model_spec(args.model_spec)
    checkpoint_prefix = resolve_checkpoint_prefix(spec)

    reg_f = float(args.reg_f) if args.reg_f is not None else float(spec.get("reg_f", 1.0))
    reg_rate_hz = float(args.reg_rate_hz) if args.reg_rate_hz is not None else float(spec.get("reg_rate_hz", 10.0))

    task_settings = build_task_settings(args)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eprop_tag = "eprop_{}".format("true" if bool(spec.get("eprop", False)) else "false")
    run_name = "sternberg_continue_{}_delay{}_jitter_{}_{}_{}".format(eprop_tag, task_settings["delay"], 
                                                                      task_settings["jitter_onset"], task_settings["jitter_delay"],
                                                                      run_ts)
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tf.reset_default_graph()
    saver, tensors = build_graph(
        spec=spec,
        settings=task_settings,
        n_batch=args.n_batch,
        learning_rate=args.learning_rate,
        reg_f=reg_f,
        reg_rate_hz=reg_rate_hz,
    )

    validation_loss_list = []
    validation_error_list = []
    validation_acc_list = []
    train_loss_list = []
    train_error_list = []
    training_time_list = []
    n_iter_list = []

    t_ref = time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_prefix)
        print("Restored model weights from {}".format(checkpoint_prefix))

        for k_iter in range(args.n_iter):
            if np.mod(k_iter, args.validate_every) == 0:
                val_feed = get_data_dict(
                    batch_size=args.n_batch,
                    settings=task_settings,
                    inputs=tensors["inputs"],
                    targets=tensors["targets"],
                    labels=tensors["labels"],
                    loads=tensors["loads"],
                    jitter_onsets=tensors["jitter_onsets"],
                    jitter_delays=tensors["jitter_delays"],
                )
                val_loss, val_err, val_acc = sess.run(
                    [tensors["loss"], tensors["recall_errors"], tensors["accuracy"]],
                    feed_dict=val_feed,
                )
                validation_loss_list.append(float(val_loss))
                validation_error_list.append(float(val_err))
                validation_acc_list.append(float(val_acc))

                if k_iter > 0 and val_err < args.stop_crit:
                    n_iter_list.append(int(k_iter))
                    print("Early stopping at iteration {} (validation recall error {:.4f})".format(k_iter, val_err))
                    break

            train_feed = get_data_dict(
                batch_size=args.n_batch,
                settings=task_settings,
                inputs=tensors["inputs"],
                targets=tensors["targets"],
                labels=tensors["labels"],
                loads=tensors["loads"],
                jitter_onsets=tensors["jitter_onsets"],
                jitter_delays=tensors["jitter_delays"],
            )
            t0 = time()
            _, train_loss, train_err = sess.run(
                [tensors["train_step"], tensors["loss"], tensors["recall_errors"]],
                feed_dict=train_feed,
            )
            t_train = time() - t0

            train_loss_list.append(float(train_loss))
            train_error_list.append(float(train_err))
            training_time_list.append(float(t_train))

            if np.mod(k_iter, args.print_every) == 0:
                print(
                    "Iter {:5d} | train loss {:.4f} | train err {:.4f} | val err {:.4f}".format(
                        k_iter,
                        train_loss,
                        train_err,
                        validation_error_list[-1] if validation_error_list else float("nan"),
                    )
                )

            if k_iter == args.n_iter - 1:
                n_iter_list.append(int(k_iter))

        print("Continuation training finished in {:.2f}s".format(time() - t_ref))

        final_ckpt_prefix = os.path.join(run_dir, "model.ckpt")
        checkpoint_path = saver.save(sess, final_ckpt_prefix)
        print("Saved continued model checkpoint to {}".format(checkpoint_path))

    results = {
        "continued_from_model_spec": os.path.abspath(args.model_spec),
        "continued_from_checkpoint": checkpoint_prefix,
        "iterations": n_iter_list,
        "validation_losses": validation_loss_list,
        "validation_errors": validation_error_list,
        "validation_accuracies": validation_acc_list,
        "train_losses": train_loss_list,
        "train_errors": train_error_list,
        "training_time": training_time_list,
        "task_settings": task_settings,
        "args": vars(args),
    }

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    updated_spec = dict(spec)
    updated_spec.update(
        {
            "checkpoint_prefix": os.path.join(run_dir, "model.ckpt"),
            "continued_from_model_spec": os.path.abspath(args.model_spec),
            "continued_from_checkpoint": checkpoint_prefix,
            "task_settings": task_settings,
            "learning_rate": args.learning_rate,
            "n_batch": args.n_batch,
            "n_iter": args.n_iter,
            "reg_f": reg_f,
            "reg_rate_hz": reg_rate_hz,
        }
    )

    model_spec_path = os.path.join(run_dir, "model_spec.json")
    with open(model_spec_path, "w") as f:
        json.dump(updated_spec, f, indent=2, default=_json_default)

    print("Saved continuation results to {}".format(results_path))
    print("Saved updated model spec to {}".format(model_spec_path))


if __name__ == "__main__":
    main()
