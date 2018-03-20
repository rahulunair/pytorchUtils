"""A KNN based on cosine similarity"""

import torch
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.functional import normalize
from typing import Tuple


def knn(query: Variable, var_2: Variable, k: int = 5) -> Tuple[Variable, LongTensor]:
    """query batch x key-size, variable batch x key-size"""
    assert query.shape[1] == var_2.shape[1]
    assert query.shape[0] <= var_2.shape[0]
    query = normalize(query)
    var_2 = normalize(var_2)
    similarity_score = torch.mm(query, torch.t(var_2))
    sims, indices = torch.topk(similarity_score, k)
    return sims, indices.detach().data


def test_knn():
    query = Variable(torch.Tensor(10, 64).uniform_(-0.01, 0.01))
    ps = Variable(torch.Tensor(10000, 64).uniform_(-100, 100))
    sims, indices = knn(query, ps)
    print("Similarity and indices: ", sims, indices)


if __name__ == "__main__":
    test_knn()
