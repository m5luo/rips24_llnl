import torch
from constants import device


def torch_row_to_matrix(row:torch.Tensor) -> torch.Tensor:
    """
    Transforms stencil to tri-diagonal matrix
    """
    n = torch.Tensor.size(row, 0)
    k = n // 2
    M = torch.zeros((n, n))
    for i in range(n):
        M[i,max(0,i - k):min(n, k + 1 + i)] = row[max(0, n - (k + i + 1)):min(n + k - i, n)]
    return M.to(device)

def torch_row_to_matrix_with_numrows(row:torch.Tensor, numrows:torch.Tensor) -> torch.Tensor:
    n = torch.Tensor.size(row, 0)
    answer = torch.zeros((numrows, numrows))
    for i in range(0,n//2+1):
        answer[i,0:n//2+i+1] = row[n//2-i:n]
        answer[numrows-i-1,numrows-1-i-n//2:numrows] = row[0:n//2+i+1]
    for i in range(0,numrows-n):
        answer[n//2+i,i:i+n] = row
    return answer.to(device)