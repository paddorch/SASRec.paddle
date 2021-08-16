import time
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn


class MyBCEWithLogitLoss(paddle.nn.Layer):
    def __init__(self):
        super(MyBCEWithLogitLoss, self).__init__()

    def forward(self, pos_logits, neg_logits, labels):
        return paddle.sum(
            - paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels -
            paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels,
            axis=(0, 1)
        ) / paddle.sum(labels, axis=(0, 1))


class SASRec(paddle.nn.Layer):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units)  # [pad] is 0
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout)
        # self.pos_embed = nn.Parameter(torch.normal(0, 1, (max_len, embed_size)))
        self.subsequent_mask = (paddle.triu(paddle.ones((args.maxlen, args.maxlen))) == 0)
        # TODO dropout
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_units,
                                                        nhead=args.num_heads,
                                                        dim_feedforward=args.hidden_units,
                                                        dropout=args.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args.num_blocks)

    def pe(self, seqs):
        seqs = self.item_emb(seqs)   # shape - batch_size, max_len, embed_size
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        position_embed = self.pos_emb(paddle.to_tensor(positions, dtype='int64'))
        return self.emb_dropout(seqs + position_embed)
        # return seqs + position_embed

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        # all input seqs: (batch_size, seq_len)
        seqs_embed = self.pe(log_seqs)   # (batch_size, seq_len, embed_size)
        log_feats = self.encoder(seqs_embed, self.subsequent_mask)  # (batch_size, seq_len, embed_size)

        pos_embed = self.item_emb(pos_seqs) # (batch_size, seq_len, embed_size)
        neg_embed = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embed).sum(axis=-1)
        neg_logits = (log_feats * neg_embed).sum(axis=-1)

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):  # for inference
        seqs = self.pe(log_seqs)
        log_feats = self.encoder(seqs, self.subsequent_mask)  # (batch_size, seq_len, embed_size)

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        item_embs = self.item_emb(paddle.to_tensor(item_indices, dtype='int64'))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits  # preds # (U, I)


def _test():
    from train_transformer import parser
    args = parser.parse_args()
    model = SASRec(100, args)
    print(model)


if __name__ == '__main__':
    _test()
