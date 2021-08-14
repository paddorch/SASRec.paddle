import numpy as np
# import torch
import paddle


class PointWiseFeedForward(paddle.nn.Layer):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = paddle.nn.Conv1D(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = paddle.nn.Dropout(p=dropout_rate)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv1D(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = paddle.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose((0, 2, 1)))))))
        outputs = outputs.transpose((0, 2, 1)) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(paddle.nn.Layer):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = paddle.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = paddle.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = paddle.nn.LayerList() # to be Q for self-attention
        self.attention_layers = paddle.nn.LayerList()
        self.forward_layernorms = paddle.nn.LayerList()
        self.forward_layers = paddle.nn.LayerList()

        self.last_layernorm = paddle.nn.LayerNorm(args.hidden_units, epsilon=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = paddle.nn.LayerNorm(args.hidden_units, epsilon=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  paddle.nn.MultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = paddle.nn.LayerNorm(args.hidden_units, epsilon=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(paddle.to_tensor(log_seqs, dtype='int64')) # (bs, sl, hs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(paddle.to_tensor(positions, dtype='int64'))
        seqs = self.emb_dropout(seqs)

        timeline_mask = paddle.to_tensor(log_seqs == 0, dtype='bool')
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~paddle.tril(paddle.ones((tl, tl), dtype='bool'))

        for i in range(len(self.attention_layers)):
            seqs = paddle.transpose(seqs, (1, 0, 2))  # TODO why
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = paddle.transpose(seqs, (1, 0, 2))

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(paddle.to_tensor(pos_seqs, dtype='int64'))
        neg_embs = self.item_emb(paddle.to_tensor(neg_seqs, dtype='int64'))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(paddle.to_tensor(item_indices, dtype='int64')) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
