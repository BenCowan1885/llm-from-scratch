import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- Phase 3: attention ----

# NOTE: this section uses toy/dummy data (not the real dataset)
# purely to demonstrate the averaging mechanism before building real attention

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range (T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)
print(xbow[0])

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
print(torch.allclose(xbow, xbow2, atol=1e-6))

# Now to create real self-attention head:

torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias = False)
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
print(out.shape)
print(wei[0])