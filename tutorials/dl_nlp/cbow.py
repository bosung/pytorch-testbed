import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        # this nn.Embedding takes role as V vector in cs224d
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        """
        e.g) word index like 
        {'I': 0, 'REALLY': 1, 'LOVE': 2, ..., 'DOG': 11, 'CUTE': 76}
        and if input is 'I really love cute dog' then context window is
        'i', 'really', 'cute', 'dog'. In this case, arguments like 

        Arguments:
            input (Varaible): [0, 1, 76, 11]

        get embeded vector with this index list thourgh embeddig function
        let embedding dimenstion 10, the shape of weight vector in linear1
        is 4x10 metrix which is called 'context voector'.

        """
        # embeds is context vector metrix.
        embeds = self.embeddings(inputs)
        # print(inputs, embeds, embeds.sum(dim=0).view((1, -1)))
        out = self.linear2(F.relu(self.linear1(embeds.sum(dim=0))))
        log_probs = F.log_softmax(out, dim=0).view((1, -1))
        return log_probs

# create your model and train. here are some functions 
# to help you make the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

make_context_vector(data[0][0], word_to_ix)  # example

embedding_dim = 5
model = CBOW(vocab_size, embedding_dim, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(3000):
    total_loss = torch.Tensor([0])
    for context, target in data:
        # print(context, target)
        # step 1. make data to vector
        context_vector = make_context_vector(context, word_to_ix)

        # step 2. initialize gradients
        model.zero_grad()

        # step 3. put data into model
        out = model(context_vector)

        # step 4. loss function
        loss = loss_function(out, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # step 5. bacward
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    print("total_loss: %s " % total_loss)

