import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

logits = [1.5, 2.5, -0.9]

sm_output = [(np.exp(i) / sum(np.exp(logits))).item() for i in logits]
print(sm_output)

log_output = [np.log(x).item() for x in sm_output]
print(f'Log Output: {log_output}')

neg_log_likelihood = -1 * np.array(log_output)
print(neg_log_likelihood)

logits = torch.tensor([[2.0, 1.0, 0.1]])
log_probs = F.log_softmax(logits, dim=1)
print("log_probs: ", log_probs)
targets = torch.tensor([0])

loss_fn = nn.CrossEntropyLoss()
loss    = loss_fn(logits, targets)
print(loss.item())
