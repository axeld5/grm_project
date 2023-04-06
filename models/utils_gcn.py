import torch
from torch_geometric.utils import to_torch_coo_tensor


def calculate_laplacian_with_self_loop(edge_index):
    matrix = to_torch_coo_tensor(edge_index).to("cuda:0")
    id = torch.eye(matrix.size(0)).to_sparse().to("cuda:0")
    matrix = matrix + id
    row_sum = torch.sparse.sum(matrix, dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).to_dense().flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    id_tensor = torch.arange(d_inv_sqrt.shape[0])[None, :].expand((2, d_inv_sqrt.shape[0])).to("cuda:0")
    d_mat_inv_sqrt = torch.sparse_coo_tensor(id_tensor, d_inv_sqrt).to("cuda:0")
    prod = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return prod
