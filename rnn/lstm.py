import torch
import torch.autograd as autograd
import torch.nn as nn

torch.manual_seed(1)


# Args:
#   input_size: # of expected features in the input x (dim of input x)
#   hidden_size: # of features in the hidden state h
#   num_layers: Number of recurrent layers. (default = 1)
lstm = nn.LSTM(3, 3)

inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]

# initialize the hidden state.
# Q? why dose hidden state use (1, 1, 3) not (1, 3)
# A. pytorch's LSTM expects all of its inputs to be 3D tensors.
#    The 1st axis is the sequence itself, the 2nd indexes instances in 
#    the minibatch, and the 3rd indexes elements of the input.
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
        autograd.Variable(torch.randn((1, 1, 3))))

for i in inputs:
    # step through the sequence on element at a time.
    # after each step, hidden contains the hidden state.
    #
    # Inputs: input, (h_0, c_0)
    #   - **input** (seq_len, batch, input_size): tensor containing the
    #   features of the input sequence.
    #   The input can also be a packed variable length sequence.
    #   - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
    #   containing the initial hidden state for each element in the batch.
    #   - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
    #   containing the initial cell state for each element in the batch.
    #
    #   if (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# "out" will give you access to al hidden states in the sequece
# "hidden" will allow you to continue the sequence and backproagate,
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
        autograd.Variable(torch.randn((1, 1, 3)))) # clean out hidden state
out, hidden = lstm(inputs, hidden)

