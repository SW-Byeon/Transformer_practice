from ast import Mult
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import math
class Transformer(nn.Module):
    def __init__(self, src_embed, trg_embed, encoder, decoder, fc_layer):
        super().__init__()
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.fc_layer = fc_layer
    
    def forward(self, src, trg, src_mask, trg_mask):
        encoder_output = self.encoder(self.src_embed(src), src_mask)#encoder output: context
        out = self.decoder(self.trg_embed(trg), trg_mask, encoder_output, src_mask)
        out = self.fc_layer(out)
        out = F.log_softmax(out, dim=-1)
        return out

class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer):#n_layer: 레이어 개수
        super().__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))
        
    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out 

class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
        super().__init__()
        self.multi_head_attention_layer = multi_head_attention_layer
        self.position_wise_feed_forward_layer = position_wise_feed_forward_layer
        self.residual_connection_layers = [ResidualConnectionLayer(copy.deepcopy(norm_layer)) for i in range(2)]
    
    def forward(self, x, mask):
        out = self.residual_connection_layers[0](x, lambda x: self.multi_head_attention_layer(x, x, x, mask))
        out = self.residual_connection_layers[1](x, lambda x: self.position_wise_feed_forward_layer(x))
        return out
    
class Decoder(nn.Module):
    def __init__(self, sub_layer, n_layer):
        super().__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(sub_layer))
            
    def forward(self, x, mask, encoder_output, encoder_mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask, encoder_output, encoder_mask)
        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, masked_multi_head_attention_layer, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
        super().__init__()
        self.masked_multi_head_attention_layer = ResidualConnectionLayer(masked_multi_head_attention_layer, copy.deepcopy(norm_layer))
        self.multi_head_attention_layer = ResidualConnectionLayer(multi_head_attention_layer, copy.deepcopy(norm_layer))
        self.position_wise_feed_forward_layer = ResidualConnectionLayer(position_wise_feed_forward_layer, copy.deepcopy(norm_layer))
    
    def forward(self, x, mask, encoder_output, encoder_mask):
        out = self.masked_multi_head_attention_layer(query=x, key=x, value=x, mask=mask)
        out = self.multi_head_attention_layer(query=out, key=encoder_output, value=encoder_output, mask=encoder_mask)
        out = self.position_wise_feed_forward_layer(x=out)
        return out
    

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
        # qkv_fc_layer's shape: (d_embed, d_model)
        #d_model = h * d_k
        # fc_layer's shape: (d_model, d_embed)
        super().__init__()
        self.d_model = d_model
        self.h = h
        #3개의 fc_layer가 독립적으로 학습되게 하기 위해 deepcopy사용
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer#attention 계산 이후 출력 전에 거쳐가는 layer
        
    def calculate_attention(self, query, key, value, mask):
        #mask: masking tensor(element '0' will be masked)
        #q, k, v's shape: (n_batch, seq_len, d_k)
        #word embedding, fcl 거쳐서 입력된다
        d_k = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))# Q x K^T
        attention_score = attention_score / math.sqrt(d_k) #scaling
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9) #masking
        attention_prob = F.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_prob, value)
        return out
        
    def forward(self, query, key, value, mask=None):
        #q, k , v's shape: (n_batch, seq_len, d_embed)
        #mask's shape: (n_batch, seq_len, seq_len)
        n_batch = query.shape[0]
        
        def transform(x, fc_layer):
            #reshape (n_batch, seq_len, d_embed) to (n_batch, h, seq_len, d_k)
            out = fc_layer(x) # out's shape: (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h) # out's shape: (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # out's shape: (n_batch, h, seq_len, d_k)
            return out
        
        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)
        
        if mask is not None:
            mask = mask.unsqueeze(1) #mask's shape: (n_batch, 1, seq_len, seq_len)
        
        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2) # out's shape: (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.fc_layer(out) # (n_batch, seq_len, d_embed)
        return out
    

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, first_fc_layer, second_fc_layer):
        self.first_fc_layer = first_fc_layer
        self.second_fc_layer = second_fc_layer
        
    def forward(self, x):
        out = self.first_fc_layer(x)
        out = F.relu(out)
        out = F.dropout(out)
        out = self.second_fc_layer(out)
        return out
    
class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm_layer):
        super().__init__()
        self.norm_layer = norm_layer
        
    def forward(self, x, sub_layer):
        out = sub_layer(x) + x
        out = self.norm_layer(out)
        return out
    

def subsequent_mask(size):
    atten_shape = (1, size, size)
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')#masking with upper triangle matrix
    return torch.from_numpy(mask)==0 # reverse (masking=False, non-masking=True)

def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad) #pad masking
    tgt_mask = tgt_mask.unsqueeze(-2) #reshape (n_batch, seq_len) -> (n_batch, 1, seq_len)
    # pad_masking & subsequent_masking
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) 
    return tgt_mask


class TransformerEmbedding(nn.Module):
    def __init__(self, embedding, positional_encoding):
        super().__init__()
        self.embedding = nn.Sequential(embedding, positional_encoding)
    
    def forward(self, x):
        out = self.embedding(x)
        return out

class Embedding(nn.Module):
    def __init__(self, d_embed, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), d_embed)
        self.vocab = vocab
        self.d_embed = d_embed
    
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed) # embedding 결과에 scaling
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=5000):
        super().__init__()
        encoding = torch.zeros(max_seq_len, d_embed) #Embedding Size
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding
    
    def forward(self, x):
        out = x + Variable(self.encoding[:, :x.size(1)], requires_grad=False)#학습되지않도록 requires_grad = False
        out = self.dropout(out)
        return out
    
def make_model(
    src_vocab,
    trg_vocab,
    d_embed = 512,
    n_layer = 6,
    d_model = 512,
    h = 8,
    d_ff = 2048):
    
    cp = lambda x: copy.deepcopy(x)
    
    # multi_head_attention_layer 생성한 뒤 copy해서 사용 
    multi_head_attention_layer = MultiHeadAttentionLayer(d_model=d_model,
                                                         h=h,
                                                         qkv_fc_layer=nn.Linear(d_embed, d_model),
                                                         fc_layer=nn.Linear(d_model, d_embed))
    
    # position_wise_feed_forward_layer 생성한 뒤 copy해서 사용 
    position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(first_fc_layer=nn.Linear(d_embed, d_ff),
                                                                    second_fc_layer=nn.Linear(d_ff, d_embed))
    
    # norm_layer 생성한 뒤 copy해서 사용 
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    
    # 실제 model 생성 
    model = Transformer(src_embed=TransformerEmbedding(embedding=Embedding(d_embed=d_embed,
                                                                           vocab=src_vocab),
                                                       positional_encoding=PositionalEncoding(d_embed=d_embed)),
                        trg_embed=TransformerEmbedding(embedding=Embedding(d_embed=d_embed,
                                                                           vocab=trg_vocab),
                                                       positional_encoding=PositionalEncoding(d_embed=d_embed)),
                        encoder=Encoder(sub_layer=EncoderLayer(multi_head_attention_layer=cp(multi_head_attention_layer),
                                                               position_wise_feed_forward_layer=cp(position_wise_feed_forward_layer),
                                                               norm_layer=cp(norm_layer)),
                                        n_layer=n_layer),
                        decoder=Decoder(sub_layer=DecoderLayer(masked_multi_head_attention_layer=cp(multi_head_attention_layer),
                                                               multi_head_attention_layer=cp(multi_head_attention_layer),
                                                               position_wise_feed_forward_layer=cp(position_wise_feed_forward_layer),
                                                               norm_layer=cp(norm_layer)),
                                        n_layer=n_layer),
                        fc_layer=nn.Linear(d_model, len(trg_vocab)))
    return model
