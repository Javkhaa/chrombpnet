import numpy as np ;
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import multinomial_nll
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

"""
Multi-task ChromBPNet: a shared dilated-conv trunk feeding TWO BPNet-style
profile+count head pairs.

  Head 1 (accessibility): Tn5 cut-site insertion profile + total count
                          -- the standard ChromBPNet target.
  Head 2 (nucleosome):    mono-nucleosomal dyad/occupancy profile + total count
                          -- the novel target derived from fragment insert
                          lengths (centers of ~150-250 bp fragments).

Both heads branch off the SAME trunk tensor `x`; that shared representation is
the whole point of the multi-task formulation. The trunk is identical to
chrombpnet's stock `bpnet_model.py` so a pretrained trunk/head-1 can be loaded
by name if desired.

Bias note: chrombpnet's Tn5-bias correction (the frozen bias model added in
logit/log space) is a CUT-SITE phenomenon and applies to HEAD 1 only. Head 2
targets fragment centers, which carry no Tn5 insertion-site bias, so it branches
off the bias-free trunk and takes no bias term. This file builds the bias-free
shared-trunk model; bias correction for head 1 is layered on separately (same
Add/logsumexp pattern as chrombpnet_with_bias_model.py) once a bias model exists.
"""


def _profile_count_head(x, out_pred_len, profile_kernel_size, prefix, num_tasks=1):
    """One BPNet-style head: large-kernel profile conv (cropped) + GAP->Dense count."""
    prof_precrop = Conv1D(filters=num_tasks,
                          kernel_size=profile_kernel_size,
                          padding='valid',
                          name='%s_prof_out_precrop' % prefix)(x)
    cropsize = int(int_shape(prof_precrop)[1] / 2) - int(out_pred_len / 2)
    assert cropsize >= 0
    assert (int_shape(prof_precrop)[1] % 2 == 0)  # symmetric crop
    prof = Cropping1D(cropsize, name='%s_logits_profile_preflatten' % prefix)(prof_precrop)
    profile_out = Flatten(name='%s_logits_profile_predictions' % prefix)(prof)

    gap = GlobalAvgPool1D(name='%s_gap' % prefix)(x)
    count_out = Dense(num_tasks, name='%s_logcount_predictions' % prefix)(gap)
    return profile_out, count_out


def getModelGivenModelOptionsAndWeightInits(args, model_params):
    # trunk hyperparams (match chrombpnet defaults)
    conv1_kernel_size = 21
    profile_kernel_size = 75
    num_tasks = 1

    filters = int(model_params['filters'])
    n_dil_layers = int(model_params['n_dil_layers'])
    sequence_len = int(model_params['inputlen'])
    out_pred_len = int(model_params['outputlen'])

    # per-head loss weights. counts heads weighted like chrombpnet's
    # counts_loss_weight; the nucleosome profile is weighted by
    # nucleosome_profile_weight (start at 1.0 == equal to accessibility profile).
    acc_counts_w = float(model_params.get('counts_loss_weight', 1.0))
    nuc_profile_w = float(model_params.get('nucleosome_profile_weight', 1.0))
    nuc_counts_w = float(model_params.get('nucleosome_counts_weight', acc_counts_w))

    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)

    # ---- shared trunk (identical to bpnet_model.py) ----
    inp = Input(shape=(sequence_len, 4), name='sequence')
    x = Conv1D(filters, kernel_size=conv1_kernel_size, padding='valid',
               activation='relu', name='bpnet_1st_conv')(inp)
    for i in range(1, n_dil_layers + 1):
        conv_x = Conv1D(filters, kernel_size=3, padding='valid', activation='relu',
                        dilation_rate=2 ** i, name='bpnet_{}conv'.format(i))(x)
        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert ((x_len - conv_x_len) % 2 == 0)
        x = Cropping1D((x_len - conv_x_len) // 2, name='bpnet_{}crop'.format(i))(x)
        x = add([conv_x, x])

    # ---- two heads off the shared trunk ----
    acc_profile, acc_count = _profile_count_head(
        x, out_pred_len, profile_kernel_size, prefix='accessibility', num_tasks=num_tasks)
    nuc_profile, nuc_count = _profile_count_head(
        x, out_pred_len, profile_kernel_size, prefix='nucleosome', num_tasks=num_tasks)

    model = Model(inputs=[inp],
                  outputs=[acc_profile, acc_count, nuc_profile, nuc_count])

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=[multinomial_nll, 'mse', multinomial_nll, 'mse'],
        loss_weights=[1, acc_counts_w, nuc_profile_w, nuc_counts_w])

    return model


def save_model_without_bias(model, output_prefix):
    # No bias branch in this (bias-free) multi-task model; saved as-is.
    return
