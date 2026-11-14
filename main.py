import numpy as np
from tokenizer import enc



# ======================================================================
# Building blocks: embedding, softmax, cross-entropy
# ======================================================================
class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.2
        self.inputs = None

        # Gradients
        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, inputs):
        # inputs: (B, T) token ids -> (B, T, embedding_dim) vectors
        self.inputs = inputs
        return self.embeddings[inputs]

    def backward(self, grad_output):
        # grad_output: (B, T, embedding_dim)
        # Each token's gradient goes to its row; np.add.at accumulates repeated ids
        self.grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(self.grad_embeddings, self.inputs, grad_output)

        # Token ids are integers, so there is no gradient to pass further back
        return None


class Softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        # Softmax is shift-invariant so we subtract max_logit for numerical stability
        max_logit = np.max(inputs, axis=-1, keepdims=True)

        softmax_out = np.exp(inputs - max_logit) / np.sum(np.exp(inputs - max_logit), axis=-1, keepdims=True)
        self.softmax_out = softmax_out

        return softmax_out

    def backward(self, grad_output):
        # Jacobian-vector product: s * (g - sum(g * s))
        s = self.softmax_out
        return s * (grad_output - np.sum(grad_output * s, axis=-1, keepdims=True))


class CrossEntropyWithSoftmax:
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.labels_flat = None

    def forward(self, logits, labels):
        assert len(logits.shape) == 3
        B, T, V = logits.shape

        logits_flat = np.reshape(logits, (B * T, V))
        labels_flat = np.reshape(labels, (B * T))

        max_logit = np.max(logits_flat, axis=-1, keepdims=True)
        diff = logits_flat - max_logit

        lse = max_logit + np.log(np.sum(np.exp(diff), axis=-1, keepdims=True))
        log_softmax = np.expand_dims(logits_flat[np.arange(B * T), labels_flat], axis=-1) - lse
        self.lse = lse
        self.logits_flat = logits_flat
        self.labels_flat = labels_flat
        self.B = B
        self.T = T
        self.V = V

        return -np.sum(log_softmax) / logits_flat.shape[0]

    def backward(self):
        # d(loss)/d(logits) = (softmax - one_hot) / (B * T)
        probs = np.exp(self.logits_flat - self.lse)
        probs[np.arange(self.B * self.T), self.labels_flat] -= 1
        grad_logits = probs / (self.B * self.T)

        return np.reshape(grad_logits, (self.B, self.T, self.V))


# ======================================================================
# GPT forward pass: text -> token ids -> vectors -> blocks -> logits -> loss
# ======================================================================

softmax = Softmax().forward


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def feed_forward(x, W1, b1, W2, b2):
    # (B, T, d) -> (B, T, 4d) -> (B, T, d)
    return gelu(x @ W1 + b1) @ W2 + b2


def multi_head_attention(x, W_q, W_k, W_v, W_out, num_heads):
    B, T, d = x.shape
    head_dim = d // num_heads

    # Project, then split into heads: (B, T, d) -> (B, heads, T, head_dim)
    Q = (x @ W_q).reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = (x @ W_k).reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = (x @ W_v).reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scores, causal mask, softmax: (B, heads, T, T)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    scores = scores + np.triu(np.full((T, T), -np.inf), k=1)
    weights = softmax(scores)

    # Weighted sum of values, then glue heads back: (B, T, d)
    context = (weights @ V).transpose(0, 2, 1, 3).reshape(B, T, d)
    return context @ W_out


def transformer_block(x, p, num_heads):
    x = x + multi_head_attention(layer_norm(x, p["gamma1"], p["beta1"]),
                                 p["W_q"], p["W_k"], p["W_v"], p["W_out"], num_heads)
    x = x + feed_forward(layer_norm(x, p["gamma2"], p["beta2"]),
                         p["W1"], p["b1"], p["W2"], p["b2"])
    return x


def init_block(d, rng):
    w = lambda *shape: rng.normal(size=shape) * 0.02
    return {
        "W_q": w(d, d), "W_k": w(d, d), "W_v": w(d, d), "W_out": w(d, d),
        "gamma1": np.ones(d), "beta1": np.zeros(d),
        "gamma2": np.ones(d), "beta2": np.zeros(d),
        "W1": w(d, 4 * d), "b1": np.zeros(4 * d),
        "W2": w(4 * d, d), "b2": np.zeros(d),
    }


class GPT:
    def __init__(self, vocab_size, context_len, d, num_heads, num_layers, seed=0):
        rng = np.random.default_rng(seed)
        self.num_heads = num_heads
        self.token_embedding = EmbeddingLayer(vocab_size, d)     # one vector per token id
        self.position_embedding = EmbeddingLayer(context_len, d) # one vector per position
        self.blocks = [init_block(d, rng) for _ in range(num_layers)]
        self.gamma_f, self.beta_f = np.ones(d), np.zeros(d)      # final layer norm
        self.W_vocab = rng.normal(size=(d, vocab_size)) * 0.02   # vector -> score per token

    def forward(self, token_ids):
        B, T = token_ids.shape

        # 1: token ids -> vectors, plus position vectors: (B, T, d)
        x = self.token_embedding.forward(token_ids) + self.position_embedding.forward(np.arange(T))

        # 2: run through every transformer block: (B, T, d)
        for p in self.blocks:
            x = transformer_block(x, p, self.num_heads)

        # 3: final layer norm, then one score per vocabulary token: (B, T, vocab_size)
        x = layer_norm(x, self.gamma_f, self.beta_f)
        return x @ self.W_vocab




if __name__ == "__main__":
    ids = np.array([enc.encode("I love Brando Code Videos")])  # (1, T)

    inputs = ids[:, :-1]   # what the model sees
    targets = ids[:, 1:]   # the next token at each position

    model = GPT(vocab_size=enc.n_vocab, context_len=64, d=16, num_heads=2, num_layers=2)
    logits = model.forward(inputs)
    loss = CrossEntropyWithSoftmax().forward(logits, targets)

    print("token ids:   ", ids)
    print("logits shape:", logits.shape)
    print("loss:        ", loss, " (random guessing ≈", np.log(enc.n_vocab), ")")
