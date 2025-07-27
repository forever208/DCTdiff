import torch
import torch.nn as nn
import torch.nn.functional as F


def img2seq(x):
    """
    (B x C x H x W) -> (B x H*W x C)
    """
    batch, channels, height, width = x.shape

    x = x.reshape(batch, channels, height * width).transpose(-1, -2)

    seq_len = height * width

    return x, seq_len


def seq2img(x, img_dim=None):
    """
    (B x H*W x C) -> (B x C x H x W)
    """
    batch, seq_len, channels = x.shape

    ### Assume Square Image if no img_dim is provided ###
    if img_dim is None:
        h = w = int(seq_len ** 0.5)

    else:
        h, w = img_dim

    x = x.transpose(-1, -2).reshape(batch, channels, h, w)

    return x


class Attention(nn.Module):
    """

    Implementation of Self and Cross Attention in One Module

    By default, self-attention will be computed on src (our images). If tgt is provided, then we are doing cross
    attention. In cross attention, an attention_mask can be used (padding mask for our embedded text), and
    src is our text and tgt is the images.

    Self-Attention:
        - Compute Self Attention on the src Tensor
            - One new step to include though is reshaping our src (image)
                from (B x C x H x W) -> (B x H*W x C) before doing attention

    Cross Attention
        - src: Our text Context (B x L x E)
        - tgt: What we want to weight against our src and output
            - One new step to include though is reshaping our tgt (image)
                from (B x C x H x W) -> (B x H*W x C) before doing attention
        - attention_mask: Padding mask for the text embeddings

    Args:
        - embedding_dimension: Number of channels in Image (the channels in BCHW act as our embedding)
        - cross_attn_dim: The embedding dimension of the text context (None for self-attention)
        - head_dim: What embedding dimension do you want to use per head?
        - attn_dropout: What dropout probability do you want to use in attention
        - groupnorm_groups: Number of groups in GroupNormalization (None if we dont need it)
        - attention_residual_connections: Do you want to add the input of attention to the output?

    """

    def __init__(self,
                 embedding_dimension=768,
                 cross_attn_dim=None,
                 head_dim=1,
                 attn_dropout=0.0,
                 groupnorm_groups=None,
                 attention_residual_connection=True,
                 bias=True,
                 return_shape="1D"):

        super(Attention, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.cross_attn_dim = cross_attn_dim
        self.attn_dropout = attn_dropout
        self.attn_residual = attention_residual_connection
        self.groupnorm_groups = groupnorm_groups

        if return_shape not in ["1D", "2D"]:
            raise Exception("Attention can output '1D' or '2D'")
        self.return_shape = return_shape

        ### Attention Head Dim ###
        self.head_dim = head_dim
        assert embedding_dimension % head_dim == 0
        self.num_heads = embedding_dimension // head_dim

        ### GroupNorm ###
        if self.groupnorm_groups is not None:
            self.groupnorm = nn.GroupNorm(num_channels=embedding_dimension,
                                          num_groups=groupnorm_groups, eps=1e-6)

        ### Attention Projections ###
        kv_input_dim = embedding_dimension if cross_attn_dim is None else cross_attn_dim
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.k_proj = nn.Linear(kv_input_dim, embedding_dimension, bias=bias)
        self.v_proj = nn.Linear(kv_input_dim, embedding_dimension, bias=bias)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def _check_for_reshape(self, images):

        ### Reshape from Img Dim to Seq Dim if in shape (B,C,H,W) ###
        if len(images.shape) == 4:
            images, num_patches = img2seq(images)
        else:
            num_patches = images.shape[1]

        return images, num_patches

    def forward_self_attn(self, images):

        batch_size = images.shape[0]

        images, num_patches = self._check_for_reshape(images)

        ### QKV Projection ###
        q = self.q_proj(images).reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,
                                                                                                          2).contiguous()
        k = self.k_proj(images).reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,
                                                                                                          2).contiguous()
        v = self.v_proj(images).reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,
                                                                                                          2).contiguous()

        attention_out = F.scaled_dot_product_attention(q, k, v,
                                                       dropout_p=self.attn_dropout if self.training else 0.0)

        ### Reshape back to (B, num_patches, head_dim) and Project with Linear ###
        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

    def forward_cross_attn(self,
                           images,
                           context,
                           attention_mask=None):

        images, num_patches = self._check_for_reshape(images)

        ### Query Projection on Images ###
        q = self.q_proj(images).reshape(-1, num_patches, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        ### Key/Value Projections on Text ###
        batch, context_len, embed_dim = context.shape
        k = self.k_proj(context).reshape(batch, context_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(context).reshape(batch, context_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        ### This is our text attention mask ###
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, num_patches, 1)

        attention_out = F.scaled_dot_product_attention(q, k, v,
                                                       attn_mask=attention_mask,
                                                       dropout_p=self.attn_dropout if self.training else 0.0)

        ### Reshape back to (B, num_patches, head_dim) and Project with Linear ###
        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

    def forward(self,
                x,
                context=None,
                attention_mask=None):

        ### x can be 1D or 2D ###
        residual = x

        if self.groupnorm_groups is not None:
            x = self.groupnorm(x)

        if context is None:
            attention_out = self.forward_self_attn(x)
        else:
            attention_out = self.forward_cross_attn(x, context, attention_mask)

        ### attention_out is always 1d ###
        if self.attn_residual:

            ### If residual shape doesnt match attention_out
            ### then the residuals must be (B,C,H,W), so we can
            ### reshape based on the output shape we want
            if len(attention_out.shape) != len(residual.shape):

                ### If we want a 1D output, flatten the residual before adding
                if self.return_shape == "1D":
                    residual, _ = img2seq(residual)

                ### If we want a 2D output, reshape the attention_out before adding
                elif self.return_shape == "2D":
                    attention_out = seq2img(attention_out)

            attention_out = attention_out + residual

        else:

            if self.return_shape == "2D":
                attention_out = seq2img(attention_out)

        return attention_out


class GEGLU(nn.Module):
    """
    GEGLU Activation as outlined here:
    https://paperswithcode.com/method/geglu
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 bias):
        super(GEGLU, self).__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x):
        hidden_states = self.proj(x)

        hidden_states, gate = hidden_states.chunk(2, dim=-1)

        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Standard MLP Module found in Transformers w/ GeGLU Activation
    """

    def __init__(self,
                 dim,
                 dim_mult,
                 mlp_dropout=0.0,
                 bias=True):
        super(FeedForward, self).__init__()

        hidden_dim = dim * dim_mult
        self.fc1_geglu = GEGLU(dim, hidden_dim, bias=bias)
        self.drop = nn.Dropout(mlp_dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1_geglu(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class BasicTransformerBlock1D(nn.Module):
    """
    Standard Transformer Block as Described in Attention is All You Need

    This applies to 1D signals in the shape (B,L,E)

    Args:
        - embed_dim: Embedding dimension of images (number of channels)
        - cross_attn_dim: Embedding dimension in for Text (for cross attention)
        - dim_mult: FeedForward hidden layer projection
        - attention_head_dim: Embedding dimension for each head of attention
        - norm_eps: Layer Norm eps
        - dropout_p: Dropout probability on Attention and FeedForward
        - attn_bias: Do you want to use bias in QKV Projections
    """

    def __init__(self,
                 embed_dim,
                 cross_attn_dim=None,
                 dim_mult=4,
                 attention_head_dim=1,
                 norm_eps=1e-6,
                 dropout_p=0.0,
                 attn_bias=True):

        super(BasicTransformerBlock1D, self).__init__()

        self.cross_attn_dim = cross_attn_dim

        ### Self-Attention ###
        self.norm1 = nn.LayerNorm(embed_dim, norm_eps)
        self.attn1 = Attention(embedding_dimension=embed_dim,
                               cross_attn_dim=None,
                               head_dim=attention_head_dim,
                               attn_dropout=dropout_p,
                               groupnorm_groups=None,
                               attention_residual_connection=False,
                               bias=attn_bias)

        ### Cross Attention ###
        if cross_attn_dim is not None:
            self.norm2 = nn.LayerNorm(embed_dim, norm_eps)
            self.attn2 = Attention(embedding_dimension=embed_dim,
                                   cross_attn_dim=cross_attn_dim,
                                   head_dim=attention_head_dim,
                                   attn_dropout=dropout_p,
                                   groupnorm_groups=None,
                                   attention_residual_connection=False,
                                   bias=attn_bias)

        ### FeedForward ###
        self.norm3 = nn.LayerNorm(embed_dim, norm_eps)
        self.feedforward = FeedForward(embed_dim,
                                       dim_mult=dim_mult,
                                       mlp_dropout=dropout_p)

    def forward(self, x, context=None, attention_mask=None):

        ### Normalize, Self-Attention, Residual Connection ###
        norm_hidden_states = self.norm1(x)
        attention_output = self.attn1(norm_hidden_states)
        x = x + attention_output

        ### Normalize, Cross-Attention, Residual Connection ###
        if self.cross_attn_dim is not None:

            if context is None:
                raise Exception("Not Passing in Context to a Block W Cross Attention")

            norm_hidden_states = self.norm2(x)
            attention_output = self.attn2(norm_hidden_states,
                                          context,
                                          attention_mask)
            x = x + attention_output

        ### Normalize and FeedForward ###
        norm_hidden_states = self.norm3(x)
        mlp_out = self.feedforward(norm_hidden_states)
        x = x + mlp_out

        return x


class TransformerBlock2D(nn.Module):
    """
    Wrapper on the TransformerBlock1D

    This applies to 2D signals in the shape (B,C,H,W) and reshapes/projects
    it to (B,H*W,embed_dim)

    Args:
        - num_attention_heads: How many heads of attention do you want?
        - attention_head_dim: What embed dim do you want per head?
        - in_channels: Number of input channels to the block
        - cross_attn_dim: Embedding dimension in for Text (for cross attention)
        - num_layers: How many transformer blocks do you want to stack?
        - dim_mult: FeedForward hidden layer projection
        - groupnorm_groups: How many groups in our groupnorm?
        - attention_bias: Do you want a bias on attention projections?
        - norm_eps: Layer Norm eps
        - dropout_p: Dropout probability on Attention and FeedForward
    """

    def __init__(self,
                 num_attention_heads,
                 attention_head_dim,
                 in_channels,
                 cross_attn_dim=None,
                 num_layers=1,
                 dim_mult=4,
                 groupnorm_groups=32,
                 attn_bias=False,
                 norm_eps=1e-6,
                 dropout_p=0.0):
        super(TransformerBlock2D, self).__init__()

        self.proj_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=groupnorm_groups,
                                 num_channels=in_channels,
                                 eps=norm_eps)

        ### Project to map (B,C,H,W) -> (B,proj_dim,H,W)
        self.proj_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=self.proj_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        ### TransformerBlock ###
        self.transformer = nn.ModuleList(

            [
                BasicTransformerBlock1D(self.proj_dim,
                                        cross_attn_dim=cross_attn_dim,
                                        dim_mult=dim_mult,
                                        attention_head_dim=attention_head_dim,
                                        norm_eps=norm_eps,
                                        dropout_p=dropout_p,
                                        attn_bias=attn_bias)

                for _ in range(num_layers)
            ]
        )

        ### Project Back to Input Channels ###
        self.proj_out = nn.Conv2d(in_channels=self.proj_dim,
                                  out_channels=in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x, context=None, attention_mask=None):
        ### Normalize and Project to Wanted Embed Dim ###
        x = self.norm(x)
        x = self.proj_in(x)

        ### Flatten to a 1D Signal (B,E,H,W) -> (B,H*W,E) ###
        x, _ = img2seq(x)

        ### Pass through Transformer ###
        for block in self.transformer:
            x = block(x, context, attention_mask)

        ### Return Back To Image Tensor (B,H*W,E) -> (B,E,H,W) ###
        x = seq2img(x)

        ### Project Back to Input Channels ###
        x = self.proj_out(x)

        return x