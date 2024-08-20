import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import torch_row_to_matrix, torch_row_to_matrix_with_numrows
from constants import (
    device, 
    input_size,
    vecsize, 
    veccntr,
    testing_dim
)

## Loss Function 1 using random vectors
class NonGalerkinLoss1Rand(nn.Module):
    """
    Loss function 1 with random vectors and constant vector.
    """
    def __init__(self) -> None:
        super(NonGalerkinLoss1Rand, self).__init__()
        ones = torch.ones((1, vecsize)).to(device)
        rnds = torch.rand(32, vecsize).to(device)
        self.ykinput = torch.cat((ones, rnds))

    def forward(self, input:torch.Tensor, output:torch.Tensor, eps:float=1e-2) -> torch.Tensor:
        bsize = input.shape[0]          # Batch size
        ysize = self.ykinput.shape[0]   # Number of yk vectors
        loss = 0
        for i in range(bsize):
            # Get phi and m from input
            phi = input[i,0:input_size-1]
            m   = input[i,input_size-1].int()
            
            # Compute phi^m and pad with zeros -> phmz
            phmz = torch.zeros(vecsize).to(device)
            phmz[veccntr-1:veccntr+2] = phi
            tmp_phmz = torch.zeros(vecsize).to(device)
            
            for _ in range(m-1):
                for j in range(1,vecsize-1):
                    tmp_phmz[j] = torch.inner(phi, phmz[j-1:j+2])
                phmz[:] = tmp_phmz[:]
            
            # Get psi from output, pad with zeros -> psiz
            psi = output[i,:]
            psiz = torch.zeros(vecsize).to(device)
            psiz[veccntr-1:veccntr+2] = psi
            
            # For each yk, compute loss += (<phim, yk> - <psi, yk>)^2 / (1 - <phim, yk>)^2
            for k in range(ysize):
                yk = self.ykinput[k,:]
                phim_yk = torch.inner(phmz, yk)
                psi_yk  = torch.inner(psiz, yk)
                loss += (phim_yk - psi_yk)**2 / ((1 - phim_yk)**2 + eps)
        return loss


## Loss Function 1 using eigenvectors
class NonGalerkinLoss1Eig(nn.Module):
    """
    Loss function 1 with eigenvectors.
    """
    def __init__(self) -> None:
        super(NonGalerkinLoss1Eig, self).__init__()
        ones = torch.ones((1, vecsize)).to(device)
        rnds = torch.rand(32, vecsize).to(device)
        self.ykinput = torch.cat((ones, rnds))

    def forward(self, input:torch.Tensor, output:torch.Tensor, eps:float=1e-2, 
                with_const:bool=False, periodic:bool=False) -> torch.Tensor:
        bsize = input.shape[0]          # batch size
        ysize = self.ykinput.shape[0]   # number of yk vectors
        loss = 0
        for i in range(bsize):
            # Get phi and m from input
            phi = input[i,0:input_size-1]
            m   = input[i,input_size-1].int()
            
            # Compute phi^m and pad with zeros -> phmz
            phmz = torch.zeros(vecsize).to(device)
            phmz[veccntr-1:veccntr+2] = phi
            Phi = torch_row_to_matrix(phmz)
            
            # Add periodic Boundary Condition
            if periodic:
                Phi[0][vecsize - 1] = 1
                Phi[vecsize - 1][0] = 1
            
            Phi_m = torch.matrix_power(Phi, m)
            phmz = Phi_m[veccntr]
            _, evecs = torch.linalg.eigh(Phi_m)
            
            # Add constant vector
            if with_const:
                evecs = torch.cat((evecs, torch.ones((1, vecsize)).to(device)))
            
            # Get psi from output, pad with zeros -> psiz
            psi = output[i,:]
            psiz = torch.zeros(vecsize).to(device)
            psiz[veccntr-1:veccntr+2] = psi
            
            # For each yk, compute loss += (<phim, yk> - <psi, yk>)^2 / (1 - <phim, yk>)^2
            for k in range(20):
                evec = evecs[-k]
                phim_yk = torch.inner(phmz, evec)
                psi_yk  = torch.inner(psiz, evec)
                loss += (phim_yk - psi_yk)**2 / ((1 - phim_yk)**2 + eps)
        return loss
  
    
# Loss Function 2 using Eigenvectors
class NonGalerkinLoss2Eig(nn.Module):
    """
    Loss function 2 with eigenvectors.
    """
    def __init__(self) -> None:
        super(NonGalerkinLoss2Eig, self).__init__()
        rnds = torch.normal(0, 1, (16, vecsize))
        self.ykinput = rnds.to(device)

    def forward(self, input:torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
        first = True
        bsize = input.shape[0]          # batch size
        loss = 0
        for i in range(bsize):
            phi = input[i,0:input_size-1]
            m   = int(input[i,input_size-1].int())
            psi = outputs[i]
            # sum = 0
            
            for dim in torch.randint(m * 2 + 50, m * 4+ 100, size = [5]):
                padding = (dim - input_size + 1) // 2
                phi_pad = F.pad(phi, (padding, padding))
                psi_pad = F.pad(psi, (padding, padding))
                Phi = torch_row_to_matrix(phi_pad)
                Phi_m = torch.matrix_power(Phi, m)
                Psi = torch_row_to_matrix(psi_pad)
                if first:
                    # print(torch.linalg.matrix_norm(Psi))
                    first = False
                
                _, evecs = torch.linalg.eigh(Phi_m)
                k = 25
                A = Psi - Phi_m
                for i in range(k):
                    evec = evecs[-i]
                    evec2 = evecs[i]
                    loss += torch.linalg.vector_norm(torch.matmul(A, evec)) ** 2 / (k)

        return loss


# Loss Function 2 with Eigenvectors + constant vector
class NonGalerkinLoss2EigConst(nn.Module):
    """
    Loss function 2 with eigenvectors and constant vector
    """
    def __init__(self) -> None:
        super(NonGalerkinLoss2EigConst, self).__init__()
        # ones = torch.ones((1, vecsize))
        # rnds = torch.rand(16, vecsize)
        rnds = torch.normal(0, 1, (16, vecsize))
        self.ykinput = rnds.to(device)
        
    def forward(self, input:torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
        first = True
        bsize = input.shape[0]          # batch size
        # ysize = self.ykinput.shape[0]
        loss = 0
        torch.manual_seed(0)
        for i in range(bsize):
            phi = input[i,0:input_size-1]
            m   = int(input[i,input_size-1].int())
            psi = outputs[i]
            # sum = 0
            
            for dim in torch.randint(m * 2 + 50, m * 4+ 100, size = [5]):
                padding = (dim - input_size + 1) // 2
                phi_pad = F.pad(phi, (padding, padding))
                psi_pad = F.pad(psi, (padding, padding))
                Phi = torch_row_to_matrix(phi_pad)
                Phi_m = torch.matrix_power(Phi, m)
                Psi = torch_row_to_matrix(psi_pad)
                if first:
                    # print(torch.linalg.matrix_norm(Psi))
                    first = False
                
                _, evecs = torch.linalg.eigh(Phi_m)
                k = 25
                # evecs_ = evecs.gather(dim = 1, index = indices)
                # id = torch.eye(phi_pad.size(0)).to(device)
                A = Psi - Phi_m
                for i in range(k):
                    evec = evecs[-i]
                    # scale = 1.5 if i == 0 else 1
                    loss += torch.linalg.vector_norm(torch.matmul(A, evec)) ** 2 / k
                ones = torch.ones([(dim - 1) // 2 * 2 + 1]).to(device)
                loss += torch.linalg.vector_norm(torch.matmul(A, ones)) ** 2 / k
        return loss
    

class NonGalerkinLoss3Eig(nn.Module):
    """
    Loss function 3. Change of M is required (Currently M = 20)
    """
    def __init__(self) -> None:
        super(NonGalerkinLoss3Eig, self).__init__()
        # rnds = torch.rand(16, vecsize)
        # rnds = torch.normal(0, 1, (16, vecsize))
        # self.ykinput = rnds.to(device)

    def forward(self, input:torch.Tensor, output:torch.Tensor) -> torch.Tensor:
        bsize = input.shape[0]
        loss = 0
        for i in range(bsize):
            phi = input[i,0:input_size-1]
            m = input[i,input_size-1].int()
            psi = output[i]

            for dim in testing_dim:
                dim = dim.item()
                matrix_phi = torch_row_to_matrix_with_numrows(phi, dim).to(device)
                matrix_psi = torch_row_to_matrix_with_numrows(psi, dim).to(device)
                phi_m = torch.matrix_power(matrix_phi, m)
                M = 20
                eigens, evecs = torch.linalg.eigh(matrix_phi)
                pair_eigens_evecs = []
                for i in range(eigens.shape[0]):
                    pair_eigens_evecs.append((eigens[i], evecs[:,i]))
                pair_eigens_evecs.sort(key = lambda x: torch.abs(x[0]), reverse = True)

                ones = torch.ones((1, dim)).to(device)
                ones = torch.transpose(ones,0,1)
                # print(evecs)
                k = min(dim,25)
                A = matrix_psi - phi_m
                for e in range(k):
                    if e == k-1:
                        l = pair_eigens_evecs[e][0].item()
                        v = pair_eigens_evecs[e][1]
                        # print(torch.matmul(A, v))
                        numerator = (M-1)*torch.linalg.norm(torch.matmul(A, v))**2
                        denominator = ((1+(M-1)*abs(1-l**m))**2)*(k+1)
                        loss += numerator/denominator
                        numerator = (M-1)*torch.linalg.norm(torch.matmul(A, ones))**2
                        denominator = (torch.linalg.norm(ones)**2+(M-1)*torch.linalg.norm(ones-torch.matmul(phi_m, ones))**2)*(k+1)
                        loss += numerator/denominator
                    else:
                        l = pair_eigens_evecs[e][0].item()
                        v = pair_eigens_evecs[e][1]
                        # print(torch.matmul(A, v))
                        numerator = (M-1)*torch.linalg.norm(torch.matmul(A, v))**2
                        denominator = ((1+(M-1)*abs(1-l**m))**2)*(k+1)
                        loss += numerator/denominator
                
        return loss


class NonGalerkinLoss3EigConst(nn.Module):
    """t
    Loss function 3 with constant. Change of M is required (Currently M = 20)
    """
    def __init__(self) -> None:
        super(NonGalerkinLoss3EigConst, self).__init__()

    def forward(self, input:torch.Tensor, output:torch.Tensor) -> torch.Tensor:
        bsize = input.shape[0]
        loss = 0
        for i in range(bsize):
            phi = input[i,0:input_size-1]
            m = input[i,input_size-1].int()
            psi = output[i]

            for dim in testing_dim:
                dim = dim.item()
                matrix_phi = torch_row_to_matrix_with_numrows(phi, dim).to(device)
                matrix_psi = torch_row_to_matrix_with_numrows(psi, dim).to(device)
                phi_m = torch.matrix_power(matrix_phi, m)
                M = 20
                eigens, evecs = torch.linalg.eigh(matrix_phi)
                pair_eigens_evecs = []
                for i in range(eigens.shape[0]):
                    pair_eigens_evecs.append((eigens[i], evecs[:,i]))
                
                #Sort by decreasing order of |eigens[i]|
                pair_eigens_evecs.sort(key = lambda x: torch.abs(x[0]), reverse = True)
                ones = torch.ones((1, dim)).to(device)
                ones = torch.transpose(ones,0,1)
                # print(evecs)
                k = min(dim,25)
                A = matrix_psi - phi_m
                for e in range(k):
                    if e == k-1:
                        l = pair_eigens_evecs[e][0].item()
                        v = pair_eigens_evecs[e][1]
                        numerator = (M-1)*torch.linalg.norm(torch.matmul(A, v))**2
                        denominator = ((1+(M-1)*abs(1-l**m))**2)*(k+1)
                        loss += numerator/denominator
                        numerator = (M-1)*torch.linalg.norm(torch.matmul(A, ones))**2
                        denominator = (torch.linalg.norm(ones)**2+(M-1)*torch.linalg.norm(ones-torch.matmul(phi_m, ones))**2)*(k+1)
                        loss += numerator/denominator
                    else:
                        l = pair_eigens_evecs[e][0].item()
                        v = pair_eigens_evecs[e][1]
                        # print(torch.matmul(A, v))
                        numerator = (M-1)*torch.linalg.norm(torch.matmul(A, v))**2
                        denominator = ((1+(M-1)*abs(1-l**m))**2)*(k+1)
                        loss += numerator/denominator   
        return loss