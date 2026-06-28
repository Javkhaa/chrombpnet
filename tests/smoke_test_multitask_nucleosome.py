#!/usr/bin/env python3
"""CPU smoke test for the multi-task (accessibility + nucleosome) model.

Proves the novel two-head architecture builds, runs forward+backward, and that
ALL FOUR losses (2 profile multinomial_nll + 2 count mse) are finite and drop on
a tiny overfit -- all on CPU, before any H100 time.
"""
import sys, types, numpy as np, tensorflow as tf

sys.path.insert(0, ".")
from chrombpnet.training.models import multitask_nucleosome_model as mt

print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

INPUTLEN, OUTPUTLEN = 2114, 1000
args = types.SimpleNamespace(seed=0, learning_rate=1e-3)
model_params = {
    "filters": 64, "n_dil_layers": 8,
    "inputlen": INPUTLEN, "outputlen": OUTPUTLEN,
    "counts_loss_weight": 1.0,
    "nucleosome_profile_weight": 1.0,
    "nucleosome_counts_weight": 1.0,
}

model = mt.getModelGivenModelOptionsAndWeightInits(args, model_params)
print("outputs:", [o.name for o in model.outputs])
print("output shapes:", [tuple(o.shape) for o in model.outputs])
print("total params:", model.count_params())

B = 8
X = np.eye(4)[np.random.randint(0, 4, size=(B, INPUTLEN))].astype("float32")
def rand_profile_and_count():
    prof = np.random.randint(0, 5, size=(B, OUTPUTLEN)).astype("float32")
    return prof, np.log(1 + prof.sum(-1, keepdims=True)).astype("float32")
acc_p, acc_c = rand_profile_and_count()    # accessibility (cut sites)
nuc_p, nuc_c = rand_profile_and_count()    # nucleosome (dyads)
Y = [acc_p, acc_c, nuc_p, nuc_c]

print("\n[forward] predicting...")
preds = model.predict(X, verbose=0)
print("  pred shapes:", [p.shape for p in preds])

print("[train] 6 steps on a fixed batch (every loss should drop):")
first = last = None
for i in range(6):
    logs = model.train_on_batch(X, Y, return_dict=True)
    if i == 0: first = logs
    last = logs
    print(f"  step {i}: total={logs['loss']:.2f} | "
          f"acc_prof={logs['accessibility_logits_profile_predictions_loss']:.2f} "
          f"acc_cnt={logs['accessibility_logcount_predictions_loss']:.3f} "
          f"nuc_prof={logs['nucleosome_logits_profile_predictions_loss']:.2f} "
          f"nuc_cnt={logs['nucleosome_logcount_predictions_loss']:.3f}")

assert np.isfinite(last["loss"]), "non-finite loss"
assert last["loss"] < first["loss"], "total loss did not decrease"
for k in first:
    if k == "loss":
        continue
    assert last[k] <= first[k] + 1e-6, f"{k} increased ({first[k]} -> {last[k]})"
print("\nSMOKE_OK: multi-task model builds; all 4 heads train; every loss dropped.")
