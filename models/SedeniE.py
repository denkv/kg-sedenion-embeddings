import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState
import itertools


class SedeniE(Model):
    VAL_DIM = 16

    def __init__(self, config):
        super(SedeniE, self).__init__(config)
        for i in range(1, self.VAL_DIM + 1):
            setattr(self, 'emb_%d' % i, nn.Embedding(self.config.entTotal, self.config.hidden_size))
            setattr(self, 'rel_%d' % i, nn.Embedding(self.config.relTotal, self.config.hidden_size))
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        for i in range(1, self.VAL_DIM + 1):
            nn.init.xavier_uniform_(getattr(self, 'emb_%d' % i).weight.data)
            nn.init.xavier_uniform_(getattr(self, 'rel_%d' % i).weight.data)

    @classmethod
    def _qmult(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        return A, B, C, D

    @classmethod
    def _qstar(self, a, b, c, d):
        return a, -b, -c, -d

    @classmethod
    def _omult(self, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c_1, c_2, c_3, c_4, d_1, d_2, d_3, d_4):

        d_1_star, d_2_star, d_3_star, d_4_star = self._qstar(d_1, d_2, d_3, d_4)
        c_1_star, c_2_star, c_3_star, c_4_star = self._qstar(c_1, c_2, c_3, c_4)

        o_1, o_2, o_3, o_4 = self._qmult(a_1, a_2, a_3, a_4, c_1, c_2, c_3, c_4 )
        o_1s, o_2s, o_3s, o_4s = self._qmult(d_1_star, d_2_star, d_3_star, d_4_star,  b_1, b_2, b_3, b_4)

        o_5, o_6, o_7, o_8 = self._qmult(d_1, d_2, d_3, d_4, a_1, a_2, a_3, a_4 )
        o_5s, o_6s, o_7s, o_8s = self._qmult( b_1, b_2, b_3, b_4, c_1_star, c_2_star, c_3_star, c_4_star)

        return  o_1 - o_1s, o_2 - o_2s, o_3 - o_3s, o_4 - o_4s, \
                o_5 + o_5s, o_6 + o_6s, o_7 + o_7s, o_8 + o_8s

    @classmethod
    def _ostar(self, arg):
        assert arg.shape[0] == 8
        c_0, c_others = arg.split((1, 7))
        return torch.vstack((c_0, c_others.neg()))
        return arg.mul(torch.tensor([1, -1, -1, -1, -1, -1, -1, -1]))

    def _onorm(self, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 ):
        denominator = torch.sqrt(r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
                                 + r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        r_1 = r_1 / denominator
        r_2 = r_2 / denominator
        r_3 = r_3 / denominator
        r_4 = r_4 / denominator
        r_5 = r_5 / denominator
        r_6 = r_6 / denominator
        r_7 = r_7 / denominator
        r_8 = r_8 / denominator

        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    @classmethod
    def _smult(cls, arg1, arg2):
        assert len(arg1.shape) == 2
        assert len(arg2.shape) == 2
        assert arg1.shape[0] == cls.VAL_DIM
        assert arg2.shape[0] == cls.VAL_DIM
        a, b = torch.split(arg1, cls.VAL_DIM//2)
        c, d = torch.split(arg2, cls.VAL_DIM//2)
        x = torch.stack(cls._omult(*a, *c)).sub(torch.stack(cls._omult(*cls._ostar(d), *b)))
        y = torch.stack(cls._omult(*d, *a)).add(torch.stack(cls._omult(*b, *cls._ostar(c))))
        return torch.cat((x, y))

    @classmethod
    def _snorm(cls, arg):
        assert arg.shape[0] == cls.VAL_DIM
        return arg.div(arg.pow(2).sum(0).sqrt())

    def _calc(self, h, t, r):
        r = self._snorm(r)
        s = self._smult(h, r)
        return -s.mul(t).sum(-1)

    def loss(self, score, regul, regul2):
        return (
            torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + self.config.lmbda_two * regul2
        )

    def embed_entities(self, x):
        return torch.stack([getattr(self, 'emb_%d' % (i + 1))(x) for i in range(0, self.VAL_DIM)])

    def embed_relations(self, x):
        return torch.stack([getattr(self, 'rel_%d' % (i + 1))(x) for i in range(0, self.VAL_DIM)])

    def forward(self):
        e_h = self.embed_entities(self.batch_h)
        e_t = self.embed_entities(self.batch_t)
        e_r = self.embed_relations(self.batch_r)
        score = self._calc(e_h, e_t, e_r)
        regul = torch.stack((e_h, e_t)).pow(2).mean(1).sum()
        regul2 = e_r.pow(2).mean(1).sum()
        return self.loss(score, regul, regul2)

    def predict(self):
        e_h = self.embed_entities(self.batch_h)
        e_t = self.embed_entities(self.batch_t)
        e_r = self.embed_relations(self.batch_r)
        score = self._calc(e_h, e_t, e_r)
        return score.cpu().data.numpy()
