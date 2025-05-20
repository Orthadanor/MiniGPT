## Building and training a bigram language model
from functools import partial
import math

import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange

from config import BigramConfig, MiniGPTConfig

class BigramLanguageModel(nn.Module):
    """
    Class definition for a simple bigram language model.
    """

    def __init__(self, config):
        """
        Initialize the bigram language model with the given configuration.

        Args:
        config : BigramConfig (Defined in config.py)
            Configuration object containing the model parameters.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.embeddings)
           You can use the Embedding layer from PyTorch.
        2. A linear layer that maps embeddings to logits. (self.linear) **set bias to True**
        3. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        super().__init__()
        # ========= TODO : START ========= #

        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        # ========= TODO : END ========= #

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the bigram language model.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, 1) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, vocab_size) containing the logits.
        """

        # ========= TODO : START ========= #
        # Logits size (batch_size, seq_len, vocab_size), where seq_len=1 for bigram model
        embeds = self.embeddings(x).squeeze(1) # (batch_size, embed_dim)
        output = self.dropout(self.linear(embeds)) # (batch_siz, vocab_size)
        return output 
    
        # ========= TODO : END ========= #

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.
        We will perform multinomial sampling which is very similar to greedy sampling,
        but instead of taking the token with the highest probability, we sample the next token from a multinomial distribution.

        Remember in Bigram Language Model, we are only using the last token to predict the next token.
        You should sample the next token x_t from the distribution p(x_t | x_{t-1}).

        Args:
        context : List[int]
            A list of integers (tokens) representing the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Output:
        List[int]
            A list of integers (tokens) representing the generated tokens.
        """

        ### ========= TODO : START ========= ###
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        curr = context[:, -1:] # Get last token
        for _ in range(max_new_tokens):
            logits = self.forward(curr)
            probs = nn.Softmax(dim=-1)(logits)
            curr = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, curr], dim=1)
        return context.squeeze(0)
        ### ========= TODO : END ========= ###


class SingleHeadAttention(nn.Module):
    """
    Class definition for Single Head Causal Self Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(
        self,
        input_dim,
        output_key_query_dim=None,
        output_value_dim=None,
        dropout=0.1,
        max_len=512,
    ):
        """
        Initialize the Single Head Attention Layer.

        The model should have the following layers:
        1. A linear layer for key. (self.key) **set bias to False**
        2. A linear layer for query. (self.query) **set bias to False**
        3. A linear layer for value. (self.value) # **set bias to False**
        4. A dropout layer. (self.dropout)
        5. A causal mask. (self.causal_mask) This should be registered as a buffer.
           - You can use the torch.tril function to create a lower triangular matrix.
           - In the skeleton we use register_buffer to register the causal mask as a buffer.
             This is typically used to register a buffer that should not to be considered a model parameter.

        NOTE : Please make sure that the causal mask is upper triangular and not lower triangular (this helps in setting up the test cases, )
        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        if output_key_query_dim:
            self.output_key_query_dim = output_key_query_dim
        else:
            self.output_key_query_dim = input_dim

        if output_value_dim:
            self.output_value_dim = output_value_dim
        else:
            self.output_value_dim = input_dim

        causal_mask = None  # You have to implement this, currently just a placeholder

        # ========= TODO : START ========= #

        self.key = nn.Linear(self.input_dim, self.output_key_query_dim, bias=False)
        self.query = nn.Linear(self.input_dim, self.output_key_query_dim, bias=False)
        self.value = nn.Linear(self.input_dim, self.output_value_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        causal_mask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
        
        # print("Initialized causal_mask (sanity check):")
        # print(causal_mask[:5, :5])
        
        # ========= TODO : END ========= #

        self.register_buffer(
            "causal_mask", causal_mask
        )  # Registering as buffer to avoid backpropagation

    def forward(self, x):
        """
        Forward pass of the Single Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, output_value_dim) containing the output tokens.

        Hint:
        - You need to 'trim' the causal mask to the size of the input tensor.
        """

        # ========= TODO : START ========= #
        B, N, _ = x.size() # (batch_size, num_tokens, token_dim)
        
        k, q = self.key(x), self.query(x) # (B, N, output_key_query_dim)
        v = self.value(x) # (B, N, output_value_dim)

        # Calculate the attention scores (B, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.output_key_query_dim)
        mask = self.causal_mask[:N, :N]  # (1, N, N)?
        mask[torch.triu(torch.ones_like(mask), diagonal=1) == 1] = float('-inf')
        
        # print("causal_mask loaded from test case:")
        # print(mask[:5, :5])
        
        attn_scores = torch.softmax(scores + mask, dim=-1) # (B, N, N)
        attn_scores = self.dropout(attn_scores) # (B, N, N)
        
        # Calculate the attention output
        attn_output = torch.matmul(attn_scores, v) # (B, N, output_value_dim)
        return attn_output
        # ========= TODO : END ========= #
        
class ParallelMHA(nn.Module):
    def __init__(self, input_dim, num_heads, output_key_query_dim=None, output_value_dim=None, max_len=512, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        self.output_key_query_dim = output_key_query_dim or input_dim
        self.output_value_dim = output_value_dim or input_dim

        assert self.output_key_query_dim % num_heads == 0, "key/query dim must be divisible by num_heads"
        assert self.output_value_dim % num_heads == 0, "value dim must be divisible by num_heads"

        self.head_dim_kq = self.output_key_query_dim // num_heads
        self.head_dim_v = self.output_value_dim // num_heads

        self.q_proj = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.output_value_dim, bias=False)

        self.out_proj = nn.Linear(self.output_value_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)  # Registering as buffer to avoid backpropagation

    def forward(self, x):
        B, N, _ = x.shape

        # Project inputs to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim_kq).transpose(1, 2)  # (B, H, N, head_dim_kq)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim_kq).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim_v).transpose(1, 2)    # (B, H, N, head_dim_v)

        # Attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim_kq)  # (B, H, N, N)
        # mask[torch.triu(torch.ones_like(mask), diagonal=1) == 1] = float('-inf') # for the testing bug
        attn_weights += self.causal_mask[:N, :N]  # (N, N) → broadcasted to (B, H, N, N)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, v)  # (B, H, N, head_dim_v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.output_value_dim)  # (B, N, output_value_dim)

        return self.out_proj(attn_output)  # Project back to input_dim


class MultiHeadAttention(nn.Module):
    """
    Class definition for Multi Head Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        """
        Initialize the Multi Head Attention Layer.

        The model should have the following layers:
        1. Multiple SingleHeadAttention layers. (self.head_{i}) Use setattr to dynamically set the layers.
        2. A linear layer for output. (self.out) **set bias to True**
        3. A dropout layer. (self.dropout) Apply dropout to the output of the out layer.

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        # ========= TODO : START ========= #
        self.head_dim = input_dim // num_heads
        
        for i in range(num_heads):
            setattr(self, f"head_{i}", SingleHeadAttention(input_dim, self.head_dim, self.head_dim, MiniGPTConfig.attention_dropout)) # attention_dropout

        self.out = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(MiniGPTConfig.out_dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Multi Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #
        
        head_output = []
        for i in range(self.num_heads):
            head_output.append(getattr(self, f"head_{i}")(x))
        attn_out = self.out(torch.cat(head_output, dim=-1))
        
        return self.dropout(attn_out)
    
        # ========= TODO : END ========= #


class FeedForwardLayer(nn.Module):
    """
    Class definition for Feed Forward Layer.
    """

    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        """
        Initialize the Feed Forward Layer.

        The model should have the following layers:
        1. A linear layer for the feedforward network. (self.fc1) **set bias to True**
        2. A GELU activation function. (self.activation)
        3. A linear layer for the feedforward network. (self.fc2) ** set bias to True**
        4. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = input_dim * 4

        # ========= TODO : START ========= #

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Feed Forward Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        ### ========= TODO : START ========= ###

        out1 = self.fc1(x)
        out2 = self.fc2(self.activation(out1))
        return self.dropout(out2)

        ### ========= TODO : END ========= ###


class LayerNorm(nn.Module):
    """
    LayerNorm module as in the paper https://arxiv.org/abs/1607.06450

    Note : Variance computation is done with biased variance.

    Hint :
    - You can use torch.var and specify whether to use biased variance or not.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        """
        Forward pass of the LayerNorm Layer.

        Args:
        input : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        mean = None
        var = None
        # ========= TODO : START ========= #
        
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, unbiased=False, keepdim=True)

        # ========= TODO : END ========= #

        if self.elementwise_affine:
            return (
                self.gamma * (input - mean) / torch.sqrt((var + self.eps)) + self.beta
            )
        else:
            return (input - mean) / torch.sqrt((var + self.eps))


class TransformerLayer(nn.Module):
    """
    Class definition for a single transformer layer.
    """

    def __init__(self, input_dim, num_heads, feedforward_dim=None, parallelHeads=False):
        super().__init__()
        """
        Initialize the Transformer Layer.
        We will use prenorm layer where we normalize the input before applying the attention and feedforward layers.

        The model should have the following layers:
        1. A LayerNorm layer. (self.norm1)
        2. A MultiHeadAttention layer. (self.attention)
        3. A LayerNorm layer. (self.norm2)
        4. A FeedForwardLayer layer. (self.feedforward)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        # ========= TODO : START ========= #

        self.norm1 = LayerNorm(input_dim)
        
        if parallelHeads:
            self.attention = ParallelMHA(input_dim, num_heads)
        else:
            self.attention = MultiHeadAttention(input_dim, num_heads)
        # self.attention = ParallelMHA(input_dim, num_heads)
        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim, MiniGPTConfig.feedforward_dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Transformer Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        norm1 = self.norm1(x)
        sub1 = x + self.attention(norm1)
        norm2 = self.norm2(sub1)
        sub2 = self.feedforward(norm2)
        
        return sub2 + sub1

        # ========= TODO : END ========= #


class MiniGPT(nn.Module):
    """
    Putting it all together: GPT model
    """

    def __init__(self, config) -> None:
        super().__init__()
        """
        Putting it all together: our own GPT model!

        Initialize the MiniGPT model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.vocab_embedding)
        2. A positional embedding layer. (self.positional_embedding) We will use learnt positional embeddings. 
        3. A dropout layer for embeddings. (self.embed_dropout)
        4. Multiple TransformerLayer layers. (self.transformer_layers)
        5. A LayerNorm layer before the final layer. (self.prehead_norm)
        6. Final language Modelling head layer. (self.head) We will use weight tying (https://paperswithcode.com/method/weight-tying) and set the weights of the head layer to be the same as the vocab_embedding layer.

        NOTE: You do not need to modify anything here.
        """

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size, config.parallelHeads
                ) ############### Parallel Multihead Attn configuration added ###############
                for _ in range(config.num_layers)
            ]
        )

        # prehead layer norm
        self.prehead_norm = LayerNorm(config.embed_dim)

        self.head = nn.Linear(
            config.embed_dim, config.vocab_size
        )  # Language modelling head

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the MiniGPT model.

        Remember to add the positional embeddings to your input token!!

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, seq_len, vocab_size) containing the logits.

        Hint:
        - You may need to 'trim' the positional embedding to match the input sequence length
        """

        ### ========= TODO : START ========= ###
        B, seq_len = x.shape
        
        input = self.vocab_embedding(x) + self.positional_embedding(self.pos[:seq_len]) 
        input = self.embed_dropout(input) # (B, seq_len, embed_dim)
        
        for layer in self.transformer_layers:
            input = layer(input)
        prehead = input

        prehead = self.prehead_norm(prehead)
        out = self.head(prehead)

        return out

        ### ========= TODO : END ========= ###

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.

        Hint:
        - This should be similar to the Bigram Language Model, but you will use the entire context to predict the next token.
          Instead of sampling from the distribution p(x_t | x_{t-1}), 
            you will sample from the distribution p(x_t | x_{t-1}, x_{t-2}, ..., x_{t-n}),
            where n is the context length.
        - When decoding for the next token, you should use the logits of the last token in the input sequence.

        """

        ### ========= TODO : START ========= ###
        
        context_len = len(context)
        
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        for _ in range(max_new_tokens):
            last_context = context[:, -context_len:] # (batch_size, seq_len)
            logits = self.forward(last_context) # (batch_size, seq_len, vocab_size) 
            probs = nn.Softmax(dim=-1)(logits[:, -1, :]) # Perform softmax on last token in the sequence
            curr = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, curr], dim=1)
        return context.squeeze(0)

        ### ========= TODO : END ========= ###

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim=None, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(input_dim)
        #self.attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout, causal=False)
        self.attention = ParallelMHA_Modified(input_dim, num_heads, dropout=dropout, causal=False)

        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim, dropout=dropout)
        

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.embed_dim, config.num_heads, config.feedforward_size)
            for _ in range(config.num_layers)
        ])

        self.norm = LayerNorm(config.embed_dim)

        # Precomputed positions
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

    def forward(self, x):
        batch_size, seq_len = x.shape

        x = self.token_embedding(x)
        pos_embed = self.position_embedding(self.pos[:seq_len]).unsqueeze(0)
        x = x + pos_embed
        x = self.embed_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        return x  # Output shape: (batch_size, seq_len, embed_dim)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim=None, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(input_dim)
        #self.self_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout, causal=True)
        
        self.self_attention = ParallelMHA_Modified(input_dim, num_heads, dropout=dropout, causal=True)
        self.cross_attention = ParallelMHA_Modified(input_dim, num_heads, dropout=dropout, causal=False)

        self.norm2 = LayerNorm(input_dim)
        #self.cross_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout, causal=False)

        self.norm3 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim, dropout=dropout)

    def forward(self, x, encoder_out):
        x = x + self.self_attention(self.norm1(x))  # masked self-attention
        x = x + self.cross_attention(self.norm2(x), encoder_out)  # encoder-decoder attention
        x = x + self.feedforward(self.norm3(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoder
        self.src_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.src_pos_embed = nn.Embedding(config.context_length, config.embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.embed_dim, config.num_heads, config.feedforward_size)
            for _ in range(config.num_layers)
        ])
        self.encoder_norm = LayerNorm(config.embed_dim)

        # Decoder
        self.tgt_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.tgt_pos_embed = nn.Embedding(config.context_length, config.embed_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.embed_dim, config.num_heads, config.feedforward_size)
            for _ in range(config.num_layers)
        ])
        self.decoder_norm = LayerNorm(config.embed_dim)

        # Final head
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        if getattr(config, "weight_tie", False):
            self.head.weight = self.tgt_embed.weight

        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.dropout = nn.Dropout(config.embed_dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt):
        """
        src: (B, T_src)
        tgt: (B, T_tgt)
        """
        B, T_src = src.shape
        B, T_tgt = tgt.shape

        src = self.src_embed(src) + self.src_pos_embed(self.pos[:T_src]).unsqueeze(0)
        src = self.dropout(src)
        for layer in self.encoder_layers:
            src = layer(src)
        src = self.encoder_norm(src)

        tgt = self.tgt_embed(tgt) + self.tgt_pos_embed(self.pos[:T_tgt]).unsqueeze(0)
        tgt = self.dropout(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, encoder_out=src)
        tgt = self.decoder_norm(tgt)

        logits = self.head(tgt)
        return logits

    def generate(self, src, max_new_tokens=100, start_token_id=0, temperature=1.0, top_k=None):
        """
        Autoregressively generate tokens from the encoder-decoder model.

        Args:
            src: (B, T_src) input source sequence
            max_new_tokens: int, number of tokens to generate
            start_token_id: int, the token ID to start decoding with
            temperature: float, softmax temperature
            top_k: int or None, top-k sampling

        Returns:
            generated: (B, max_new_tokens) the generated sequence (excluding the start token)
        """
        self.eval()
        with torch.no_grad():
            B, T_src = src.shape
            device = src.device

            # Encode the source sequence
            src_emb = self.src_embed(src) + self.src_pos_embed(self.pos[:T_src]).unsqueeze(0).to(device)
            src_emb = self.dropout(src_emb)
            for layer in self.encoder_layers:
                src_emb = layer(src_emb)
            memory = self.encoder_norm(src_emb)

            # Initialize decoder input with <BOS> token
            generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

            for _ in range(max_new_tokens):
                T_tgt = generated.shape[1]
                tgt_emb = self.tgt_embed(generated) + self.tgt_pos_embed(self.pos[:T_tgt]).unsqueeze(0).to(device)
                tgt_emb = self.dropout(tgt_emb)

                for layer in self.decoder_layers:
                    tgt_emb = layer(tgt_emb, encoder_out=memory)
                tgt_emb = self.decoder_norm(tgt_emb)

                logits = self.head(tgt_emb)  # (B, T_tgt, vocab_size)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    topk_vals, _ = torch.topk(next_token_logits, top_k)
                    threshold = topk_vals[:, -1].unsqueeze(1)
                    next_token_logits[next_token_logits < threshold] = -float("inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

                generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  

## Modified the MHA module to include the following features:
# 1. Option to use causal masking for autoregressive models, with a registered buffer for the causal mask.
# 2. Incorporate cross-attention implementation (allowing the use of a separate key-value input (kv) for attention).


class ParallelMHA_Modified(nn.Module):
    def __init__(self, input_dim, num_heads, output_key_query_dim=None, output_value_dim=None, max_len=512, dropout=0.1, causal = False):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.causal = causal

        self.output_key_query_dim = output_key_query_dim or input_dim
        self.output_value_dim = output_value_dim or input_dim

        assert self.output_key_query_dim % num_heads == 0, "key/query dim must be divisible by num_heads"
        assert self.output_value_dim % num_heads == 0, "value dim must be divisible by num_heads"

        self.head_dim_kq = self.output_key_query_dim // num_heads
        self.head_dim_v = self.output_value_dim // num_heads

        self.q_proj = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.output_key_query_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.output_value_dim, bias=False)

        self.out_proj = nn.Linear(self.output_value_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x, kv=None):
        """
        Args:
            x:  (B, T_q, D) — query input
            kv: (B, T_kv, D) — key/value source (optional, if None: self-attention)
        Returns:
            output: (B, T_q, D)
        """
        B, T_q, _ = x.shape
        if kv is None:
            kv = x
        T_kv = kv.size(1)

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T_q, self.num_heads, self.head_dim_kq).transpose(1, 2)  # (B, H, T_q, d_k)
        k = self.k_proj(kv).view(B, T_kv, self.num_heads, self.head_dim_kq).transpose(1, 2)
        v = self.v_proj(kv).view(B, T_kv, self.num_heads, self.head_dim_v).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim_kq)  # (B, H, T_q, T_kv)

        if self.causal and T_q == T_kv:
            causal_mask = self.causal_mask[:T_q, :T_kv]  # (T_q, T_kv)
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_kv)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, v)  # (B, H, T_q, d_v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.output_value_dim)

        return self.out_proj(attn_output)

