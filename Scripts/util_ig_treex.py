import torch
from torch import nn

import numpy as np
import pandas as pd

def calculate_outputs_and_gradients(inputs, model, index,cuda=False):
    # do the pre-processing
    predict_idx = None
    gradients = []
    n_steps = len(inputs)
    for i in range(n_steps):
        input = inputs[i]
        input.requires_grad = True
        input.retain_grad()
        output = model(input)
        # clear grad
        model.zero_grad()
        output[index].backward(retain_graph=True)
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients

# integrated gradients
def integrated_gradients(inputs, model, predict_and_gradients, baseline, index,steps=50, cuda=False):
    if baseline is None:
        baseline = 0 * inputs
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads = predict_and_gradients(scaled_inputs, model, index, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.expand_dims(avg_grads, axis=0)
    inputs = inputs.cpu().numpy()
    baseline = baseline.cpu().numpy()
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, predict_and_gradients, index, steps, num_random_trials, cuda):
    all_intgrads = []
    length = inputs.shape[-1]        # input shape [1,4,length]
    mid = length // 2
    baseline = torch.cuda.FloatTensor(np.zeros(inputs.shape))
    # baseline[:,:,mid] = inputs[:,:,mid]
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, predict_and_gradients, \
                                                baseline=baseline, \
                                                index=index, steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        # print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads

# ==================== TreeX Batch Processing Functions ====================

def calculate_outputs_and_gradients_batch_treex(inputs, model, index, cuda=False):
    """
    Batch version for TreeX model gradient computation.
    Inputs: list of tensors, each with shape [4, length]
    Returns: numpy array with shape [n_steps, 4, length]

    Adapted for TreeX model which returns (logits, attention_weights) tuple.
    """
    gradients = []
    n_steps = len(inputs)

    for i in range(n_steps):
        input = inputs[i]
        # Ensure input has batch dimension: [1, 4, length]
        if input.dim() == 2:
            input = input.unsqueeze(0)

        input.requires_grad = True
        input.retain_grad()

        output = model(input)

        # TreeX model returns (logits, attn_weights), extract logits
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        model.zero_grad()
        logits[0, index].backward(retain_graph=True)
        gradient = input.grad.detach().cpu().numpy()[0]  # [4, length]
        gradients.append(gradient)

    return np.array(gradients)  # [n_steps, 4, length]


def random_baseline_integrated_gradients_batch_treex(inputs_batch, model, predict_and_gradients,
                                                      index, steps=50, num_random_trials=10, cuda=False):
    """
    Batch version of random baseline integrated gradients for TreeX.

    Args:
        inputs_batch: torch.Tensor with shape [batch_size, 4, length]
        model: TreeX model
        predict_and_gradients: gradient computation function
        index: target class index
        steps: number of interpolation steps (default: 50)
        num_random_trials: number of random trials (default: 10)
        cuda: cuda flag (unused, kept for compatibility)

    Returns:
        numpy array with shape [batch_size, 4, length]
    """
    all_intgrads = []
    batch_size = inputs_batch.shape[0]
    length = inputs_batch.shape[-1]

    # Create baseline zeros on the same device as inputs
    baseline = torch.zeros(inputs_batch.shape, device=inputs_batch.device)

    for trial in range(num_random_trials):
        batch_intgrads = []

        for b in range(batch_size):
            # Process each sample in the batch
            inputs = inputs_batch[b:b+1]  # [1, 4, length]
            intgrad = integrated_gradients(
                inputs, model, predict_and_gradients,
                baseline=baseline[b:b+1],
                index=index, steps=steps, cuda=cuda
            )
            batch_intgrads.append(intgrad)

        all_intgrads.append(np.array(batch_intgrads))  # [batch_size, 1, 4, length]

    # Average across random trials: [batch_size, 1, 4, length] -> [batch_size, 4, length]
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads.squeeze(1)  # [batch_size, 4, length]
