# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass
# 
# TensorFlow 2.x compatible version - preserves exact mathematical operations

import tensorflow as tf
import numpy as np
from collections import namedtuple

try:
    TFModuleBase = tf.Module
except AttributeError:
    TFModuleBase = object


def sum_of_sines_target(seq_len, n_sines=4, periods=[1000, 500, 333, 200], weights=None, phases=None, normalize=True):
    '''
    Generate a target signal as a weighted sum of sinusoids with random weights and phases.
    :param n_sines: number of sinusoids to combine
    :param periods: list of sinusoid periods
    :param weights: weight assigned the sinusoids
    :param phases: phases of the sinusoids
    :return: one dimensional vector of size seq_len contained the weighted sum of sinusoids
    '''
    if periods is None:
        periods = [np.random.uniform(low=100, high=1000) for i in range(n_sines)]
    assert n_sines == len(periods)
    sines = []
    weights = np.random.uniform(low=0.5, high=2, size=n_sines) if weights is None else weights
    phases = np.random.uniform(low=0., high=np.pi * 2, size=n_sines) if phases is None else phases
    for i in range(n_sines):
        sine = np.sin(np.linspace(0 + phases[i], np.pi * 2 * (seq_len // periods[i]) + phases[i], seq_len))
        sines.append(sine * weights[i])

    output = sum(sines)
    if normalize:
        output = output - output[0]
        scale = max(np.abs(np.min(output)), np.abs(np.max(output)))
        output = output / np.maximum(scale, 1e-6)
    return output


def pseudo_derivative(v_scaled, dampening_factor):
    '''
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    '''
    return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    '''
    The tensorflow function which is defined as a Heaviside function (to compute the spikes),
    but with a gradient defined with the pseudo derivative.
    :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
    :param dampening_factor: parameter to stabilize learning
    :return: the spike tensor
    '''
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


EligALIFStateTuple = namedtuple('EligALIFStateTuple', ('s', 'z', 'z_local', 'r'))


class EligALIF(TFModuleBase):
    """
    Eligibility Propagation-ready ALIF (Adaptive LIF) neuron model.
    Computes spikes and eligibility traces for learning with e-prop.
    
    In TensorFlow 2.x, this uses eager execution throughout.
    All mathematical operations are preserved exactly from the original implementation.
    """
    
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=tf.float32, 
                 dampening_factor=0.3, tau_adaptation=200., beta=1.6,
                 stop_z_gradients=False, n_refractory=1, name='EligALIF'):

        if TFModuleBase is object:
            super(EligALIF, self).__init__()
        else:
            super(EligALIF, self).__init__(name=name)
        
        if tau_adaptation is None: 
            raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: 
            raise ValueError("beta parameter for adaptive bias must be set")

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = tf.constant(beta, dtype=dtype)
        self.decay_b = np.exp(-dt / tau_adaptation)

        # Convert scalars to tensors
        if np.isscalar(tau): 
            tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): 
            thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor
        self.stop_z_gradients = stop_z_gradients
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype
        self.tau = tau
        self._decay = tf.exp(-dt / tau)
        self.thr = thr

        # Initialize weights with exactly the same distribution as original
        w_in_init = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
        w_rec_init = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)
        
        self.w_in_var = tf.Variable(w_in_init, dtype=dtype, trainable=True, name='InputWeights')
        self.w_rec_var = tf.Variable(w_rec_init, dtype=dtype, trainable=True, name='RecurrentWeights')
        
        # Disconnect self-connections (diagonal)
        self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
        
        self.variable_list = [self.w_in_var, self.w_rec_var]
        self.built = True

    def get_recurrent_weights(self):
        """Get recurrent weights with diagonal (self-connections) zeroed out."""
        return tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var), self.w_rec_var)

    def state_size(self):
        """Return the structure of the state tuple."""
        return EligALIFStateTuple(
            s=tf.TensorShape((self.n_rec, 2)), 
            z=self.n_rec, 
            r=self.n_rec, 
            z_local=self.n_rec
        )

    def output_size(self):
        """Return the structure of the output."""
        return [self.n_rec, tf.TensorShape((self.n_rec, 2))]

    def zero_state(self, batch_size, dtype, n_rec=None):
        """Create zero-initialized state."""
        if n_rec is None: 
            n_rec = self.n_rec

        s0 = tf.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z_local0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return EligALIFStateTuple(s=s0, z=z0, r=r0, z_local=z_local0)

    def compute_z(self, v, b):
        """Compute spikes from voltages."""
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def compute_v_relative_to_threshold_values(self, hidden_states):
        """Compute voltage relative to adaptive threshold."""
        v = hidden_states[..., 0]
        b = hidden_states[..., 1]

        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        return v_scaled

    def __call__(self, inputs, state, stop_gradient=None):
        """
        Single timestep of RNN computation.
        
        :param inputs: input spike tensor [batch_size, n_in]
        :param state: EligALIFStateTuple
        :param stop_gradient: whether to stop gradients on spike
        :return: [output, new_state]
        """
        decay = self._decay
        z = state.z
        z_local = state.z_local
        s = state.s
        r = state.r
        v, b = s[..., 0], s[..., 1]

        # This stop_gradient allows computing e-prop with auto-diff
        # needed for correct auto-diff computation of gradient for threshold adaptation
        # stop_gradient: forward pass unchanged, gradient is blocked in the backward pass
        use_stop_gradient = stop_gradient if stop_gradient is not None else self.stop_z_gradients
        if use_stop_gradient:
            z = tf.stop_gradient(z)

        # Threshold update does not have to depend on the stopped-gradient-z, it's local
        new_b = self.decay_b * b + z_local

        # Compute input current
        w_rec_val = self.get_recurrent_weights()
        i_t = tf.matmul(inputs, self.w_in_var) + tf.matmul(z, w_rec_val)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation with refractory period
        is_refractory = r > 0
        zeros_like_spikes = tf.zeros_like(z)
        new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_z_local = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        
        new_r = r + self.n_refractory * new_z - 1
        new_r = tf.stop_gradient(tf.clip_by_value(new_r, 0., float(self.n_refractory)))
        new_s = tf.stack((new_v, new_b), axis=-1)

        new_state = EligALIFStateTuple(s=new_s, z=new_z, r=new_r, z_local=new_z_local)
        return [new_z, new_s], new_state

    def compute_eligibility_traces(self, v_scaled, z_pre, z_post, is_rec):
        """
        Compute eligibility traces for gradient computation with e-prop.
        
        CRITICAL: This function must produce numerically identical results to the original.
        All mathematical operations are preserved exactly.
        
        :param v_scaled: scaled voltages relative to threshold [batch, time, n_rec]
        :param z_pre: presynaptic spikes [batch, time, n_pre]
        :param z_post: postsynaptic spikes [batch, time, n_rec]
        :param is_rec: whether to zero out diagonal (recurrent) connections
        :return: eligibility traces and components
        """
        n_neurons = tf.shape(z_post)[2]
        rho = self.decay_b
        beta = self.beta
        alpha = self._decay
        n_ref = self.n_refractory

        # Convert to time-major format
        z_pre = tf.transpose(z_pre, perm=[1, 0, 2])
        v_scaled = tf.transpose(v_scaled, perm=[1, 0, 2])
        z_post = tf.transpose(z_post, perm=[1, 0, 2])

        # Pseudo-derivative without refractory consideration
        psi_no_ref = self.dampening_factor / self.thr * tf.maximum(0., 1. - tf.abs(v_scaled))

        # Track refractory period
        def update_refractory(refractory_count, z_post_t):
            return tf.where(
                z_post_t > 0,
                tf.ones_like(refractory_count) * (n_ref - 1),
                tf.maximum(0, refractory_count - 1)
            )

        refractory_count_init = tf.zeros_like(z_post[0], dtype=tf.int32)
        refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init)
        refractory_count = tf.concat([[refractory_count_init], refractory_count], axis=0)

        is_refractory = refractory_count > 0
        psi = tf.where(is_refractory, tf.zeros_like(psi_no_ref), psi_no_ref)

        # Compute epsilon_v (eligibility for voltage)
        def update_epsilon_v(epsilon_v, z_pre_t):
            return alpha[None, None, :] * epsilon_v + z_pre_t[:, :, None]

        epsilon_v_zero = tf.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
        epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer=epsilon_v_zero)
        epsilon_v = tf.concat([[epsilon_v_zero], epsilon_v], axis=0)

        # Compute epsilon_a (eligibility for adaptation)
        def update_epsilon_a(epsilon_a, elems):
            return (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']

        epsilon_a_zero = tf.zeros_like(epsilon_v[0])
        epsilon_a = tf.scan(
            fn=update_epsilon_a,
            elems={
                'psi': psi[:-1], 
                'epsi': epsilon_v[:-1], 
                'previous_epsi': shift_by_one_time_step(epsilon_v[:-1])
            },
            initializer=epsilon_a_zero
        )

        epsilon_a = tf.concat([[epsilon_a_zero], epsilon_a], axis=0)

        # Compute final eligibility trace
        e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

        # Convert back to batch-major format
        e_trace = tf.transpose(e_trace, perm=[1, 0, 2, 3])
        epsilon_v = tf.transpose(epsilon_v, perm=[1, 0, 2, 3])
        epsilon_a = tf.transpose(epsilon_a, perm=[1, 0, 2, 3])
        psi = tf.transpose(psi, perm=[1, 0, 2])

        # Zero out diagonal for recurrent connections if requested
        if is_rec:
            identity_diag = tf.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi

    def compute_loss_gradient(self, learning_signal, z_pre, z_post, v_post, b_post,
                              decay_out=None, zero_on_diagonal=None):
        """
        Compute weight gradients using eligibility traces.
        
        :param learning_signal: gradient from loss wrt output [batch, time, n_rec]
        :param z_pre: presynaptic spikes [batch, time, n_pre]
        :param z_post: postsynaptic spikes [batch, time, n_rec]
        :param v_post: postsynaptic voltages [batch, time, n_rec]
        :param b_post: postsynaptic adaptation [batch, time, n_rec]
        :param decay_out: optional exponential decay for output filtering
        :param zero_on_diagonal: whether to zero recurrent diagonal
        :return: gradient, e_trace, epsilon_v, epsilon_a
        """
        thr_post = self.thr + self.beta * b_post
        v_scaled = (v_post - thr_post) / self.thr

        e_trace, epsilon_v, epsilon_a, _ = self.compute_eligibility_traces(
            v_scaled, z_pre, z_post, zero_on_diagonal
        )

        if decay_out is not None:
            # Apply exponential filtering to eligibility traces
            e_trace_time_major = tf.transpose(e_trace, perm=[1, 0, 2, 3])
            filtered_e_zero = tf.zeros_like(e_trace_time_major[0])
            
            def filtering(filtered_e, e):
                return filtered_e * decay_out + e * (1 - decay_out)
            
            filtered_e = tf.scan(filtering, e_trace_time_major, initializer=filtered_e_zero)
            filtered_e = tf.transpose(filtered_e, perm=[1, 0, 2, 3])
            e_trace = filtered_e

        # Compute weight gradient
        gradient = tf.einsum('btj,btij->ij', learning_signal, e_trace)
        return gradient, e_trace, epsilon_v, epsilon_a


def exp_convolve(tensor, decay):
    '''
    Filters a tensor with an exponential filter.
    :param tensor: a tensor of shape (batch, time, neuron)
    :param decay: a decay constant of the form exp(-dt/tau) with tau the time constant
    :return: the filtered tensor of shape (batch, time, neuron)
    '''
    r_shp = range(len(tensor.shape))
    transpose_perm = [1, 0] + list(r_shp)[2:]

    tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
    initializer = tf.zeros_like(tensor_time_major[0])
    
    def filter_step(a, x):
        return a * decay + (1 - decay) * x
    
    filtered_tensor = tf.scan(filter_step, tensor_time_major, initializer=initializer)
    filtered_tensor = tf.transpose(filtered_tensor, perm=transpose_perm)

    return filtered_tensor


def shift_by_one_time_step(tensor, initializer=None):
    '''
    Shift the input on the time dimension by one.
    :param tensor: a tensor of shape (batch, time, neuron)
    :param initializer: pre-prend this as the new first element on the time dimension
    :return: a shifted tensor of shape (batch, time, neuron)
    '''
    r_shp = range(len(tensor.shape))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    tensor_time_major = tf.transpose(tensor, perm=transpose_perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor_time_major[0])

    shifted_tensor = tf.concat([initializer[None, :, :], tensor_time_major[:-1]], axis=0)
    shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)
    return shifted_tensor


def check_gradients(var_list, eprop_grads_np, true_grads_np):
    '''
    Check the correctness of the gradients.
    A ValueError() is raised if the gradients are not almost identical.

    :param var_list: the list of trainable tensorflow variables
    :param eprop_grads_np: a list of numpy arrays containing the gradients obtained eprop
    :param true_grads_np: a list of numpy arrays containing the gradients obtained with bptt
    :return: 
    '''
    for k_v, v in enumerate(var_list):
        eprop_grad = eprop_grads_np[k_v]
        true_grad = true_grads_np[k_v]

        diff = eprop_grad - true_grad
        is_correct = np.abs(diff) < 1e-4

        if np.all(is_correct):
            print('\t' + v.name + ' is correct.')
        else:
            print('\t' + v.name + ' is wrong')
            ratio = np.abs(eprop_grad) / (1e-8 + np.abs(true_grad))
            print('E-prop')
            print(np.array_str(eprop_grad[:5, :5], precision=4))
            print('True gradients')
            print(np.array_str(true_grad[:5, :5], precision=4))
            print('Difference')
            print(np.array_str(diff[:5, :5], precision=4))
            print('Ratio')
            print(np.array_str(ratio[:5, :5], precision=4))

            mismatch_indices = np.where(1 - is_correct)
            mismatch_indices = list(zip(*mismatch_indices))
            print('mismatch indices', mismatch_indices[:5])
            print('diff. vals', [diff[i, j] for i, j in mismatch_indices[:5]])

            raise ValueError()
