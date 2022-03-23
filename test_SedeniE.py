from math import sqrt
from models import SedeniE as S
import torch
import unittest


zero = torch.zeros(16)
e = [torch.cat((torch.zeros(i), torch.ones(1), torch.zeros(15 - i))).unsqueeze(1).unsqueeze(2) for i in range(0, 16)]
r1 = \
    torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
i1 = \
    torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
i2 = \
    torch.tensor([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
j1 = \
    torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
k1 = \
    torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
i1j1 = \
    torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
isqrt2jsqrt2 = \
    torch.tensor([0, 1 / sqrt(2), 1 / sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
all1 = \
    torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float64)
all2 = \
    torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.float64)
allsqrt16 = \
    torch.tensor([1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16), 1 / sqrt(16)], dtype=torch.float64)


class TestSedeniE(unittest.TestCase):
    def assertEqual(self, a, b, msg=None):
        result = torch.equal(a, b)
        if not result:
            print("got: {}, expected: {}".format(a, b))
        assert result, msg if msg else "%s == %s" % (a, b)

    def test_ostar(self):
        self.assertEqual(S._ostar(torch.tensor([
            [[1], [11]], [[2], [12]], [[3], [13]], [[4], [14]], [[5], [15]], [[6], [16]], [[7], [17]], [[8], [18]]
        ])), torch.tensor([
            [[1], [11]], [[-2], [-12]], [[-3], [-13]], [[-4], [-14]], [[-5], [-15]], [[-6], [-16]], [[-7], [-17]], [[-8], [-18]]
        ]))

    def test_snorm(self):
        self.assertEqual(S._snorm(torch.stack((i1, i2, i1j1, all1, all2), 1)).unsqueeze(2), torch.stack((i1, i1, isqrt2jsqrt2, allsqrt16, allsqrt16), 1).unsqueeze(2))

    def test_smult(self):
        for i in range(0, 16):
            self.assertEqual(S._smult(e[0], e[i]), e[i])
            self.assertEqual(S._smult(e[i], e[0]), e[i], "e{0} × e0 = e{0}".format(i))
        for i in range(1, 16):
            self.assertEqual(S._smult(e[i], e[i]), -e[0])
            for j in range(1, 16):
                if i != j:
                    self.assertEqual(S._smult(e[i], e[j]), -S._smult(e[j], e[i]), "e{0} × e{1} = - e{1} × e{0}".format(i, j))
        self.assertEqual(S._smult(e[1], e[2]), e[3])
        self.assertEqual(S._smult(S._smult(e[1], e[2]), e[3]), -e[0])


if __name__ == '__main__':
    unittest.main()
