import torch
from torch import nn
import torch.nn.functional as F
import torch.func as func
from lib.utils import AverageMeter


class TriFlowPICNN(nn.Module):

    """
    The second component of the monotone block triangular transport map g
    parameterized by Partially Input Convex Neural Networks (Amos et al., 2017)
    Inverse Map g2inv: maps the target conditional distribution to the reference marginal distribution
    Direct Map g2: generates samples from the target conditional distribution using samples
    from the reference marginal distribution and samples (conditional inputs/data) from the target marginal
    distribution
    """

    def __init__(self, prior, picnn):
        """
        :param prior: reference marginal distribution
        :param picnn: PICNN network
        """
        super(TriFlowPICNN, self).__init__()
        self.prior = prior
        self.picnn = picnn
        # trainable non-negative for quadratic term
        self.w1_picnn = nn.Parameter(torch.zeros(1))
        self.w2_picnn = nn.Parameter(torch.tensor(0.1))
        self.w3_picnn = nn.Parameter(torch.tensor(0.1))

    def get_picnn(self, x, y):
        """
        :param x: parameter/state component of the samples from the target joint distribution
        :param y: observation/data component of the samples from the target joint distribution
        :return: output of the PICNN potential
        """
        # quadratic term for strong convexity
        quad = x*x / 2
        return F.softplus(self.w1_picnn) * F.softplus(self.picnn(x, y)) + (F.relu(self.w2_picnn)+F.softplus(self.w3_picnn)) * quad

    def compute_sum(self, x, y):
        return self.get_picnn(x, y).sum()

    def g2inv(self, x, y):
        """
        :param x: parameter/state component of the samples from the target joint distribution
        :param y: observation/data component of the samples from the target joint distribution
        :return: output of inverse map
        """
        out = self.get_picnn(x, y)
        zx = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        return zx

    def g2inv_grad(self, x, y):
        """
        :param x: parameter/state component of the samples from the target joint distribution
        :param y: observation/data component of the samples from the target joint distribution
        :return: gradient of the inverse map w.r.t. x
        """
        hessian = func.vmap(func.hessian(self.compute_sum, argnums=0), in_dims=(0, 0))(x, y)
        return hessian

    def loglik_picnn(self, x, y):
        """
        :param x: parameter/state component of the samples from the target joint distribution
        :param y: observation/data component of the samples from the target joint distribution
        :return: log-likelihood
        """
        zx = self.g2inv(x, y)
        hessian = self.g2inv_grad(x, y)
        logprob = self.prior.log_prob(zx)
        if x.shape[1] > 1:
            # computing the log det of Hessian using eigenvalue decomposition
            eigen_val = torch.linalg.eigvalsh(hessian)
            logdet = torch.sum(torch.log(eigen_val), dim=1)
        else:
            logdet = torch.log(hessian)
        return logprob + logdet

    def g2(self, zx, y, tol, max_iter=1000000):
        """
        Generate samples from the target conditional distribution by solving a cvx optim problem
        using L-BFGS algorithm. Method borrowed from the CP-Flow paper (Huang et al., 2021)
        :param zx: samples from the reference marginal distribution
        :param y: conditional inputs
        :param tol: LBFGS tolerance
        :param max_iter: maximal number of iterations per optimization step
        :return: generated samples from the target conditional distribution
        """
        count = AverageMeter()    # count number of loss calculations
        yc = y.detach().requires_grad_(True)
        inv = zx.clone().detach().requires_grad_(True)

        def closure():
            out = self.get_picnn(inv, yc)
            if zx.shape[1] > 1:
                in_prod = torch.matmul(inv.unsqueeze(1), zx.unsqueeze(2)).squeeze(1)
            else:
                in_prod = inv * zx
            loss = torch.sum(out) - torch.sum(in_prod)
            count.update(1)
            inv.grad = torch.autograd.grad(loss, inv)[0].detach()
            return loss

        optimizer = torch.optim.LBFGS([inv], line_search_fn="strong_wolfe", max_iter=max_iter,
                                      tolerance_grad=tol, tolerance_change=tol)
        optimizer.step(closure)

        return inv, count.sum


if __name__ == "__main__":
    from functorch import vmap, grad
    # checking the inverse
    from src.icnn import PICNN
    picnn1 = PICNN(1, 1, 128, 1, 6)
    picnn2 = PICNN(3, 2, 128, 1, 6)
    flow1 = TriFlowPICNN(prior=None, picnn=picnn1)
    flow2 = TriFlowPICNN(prior=None, picnn=picnn2)
    x1 = torch.randn(100, 1).view(-1, 1).requires_grad_(True)
    y1 = torch.randn(100, 1).view(-1, 1).requires_grad_(True)
    x2 = torch.randn(100, 3).requires_grad_(True)
    y2 = torch.randn(100, 2).requires_grad_(True)
    zx_1 = flow1.g2inv(x1, y1)
    zx_2 = flow2.g2inv(x2, y2)
    x1_gen, num_evals1 = flow1.g2(zx_1, y1, tol=1e-12)
    x2_gen, num_evals2 = flow2.g2(zx_2, y2, tol=1e-12)
    err1 = torch.norm(x1 - x1_gen) / torch.norm(x1)
    err2 = torch.norm(x2 - x2_gen) / torch.norm(x2)
    print("Number of closure evaluations for x1: " + str(num_evals1))
    print("Number of closure evaluations for x2: " + str(num_evals2))
    print("Inversion Relative Error: " + str(err1.item()))
    print("Block Inversion Relative Error: " + str(err2.item()))

    # grad check
    fv = flow2.compute_sum(x2, y2)
    zxx_2 = flow2.g2inv_grad(x2, y2)
    dx2 = torch.rand_like(x2)
    dzx2dx2 = torch.sum(zx_2*dx2)
    dzxx2dx2 = torch.matmul(torch.matmul(dx2.unsqueeze(-1).transpose(2, 1), zxx_2), dx2.unsqueeze(-1)).sum()
    h, E0, E1, E2 = [1.0], [], [], []
    print("h\t E0 \t E1 \t E2".expandtabs(20))
    for i in range(11):
        h.append(h[i]/2)
        fvt = flow2.compute_sum(x2+h[i+1]*dx2, y2)
        fdiff = fvt-fv
        fdiff1 = fdiff - h[i+1]*dzx2dx2
        fdiff2 = fdiff1 - 0.5*(h[i+1]**2)*dzxx2dx2
        E0.append(torch.abs(fdiff).item())
        E1.append(torch.abs(fdiff1).item())
        E2.append(torch.abs(fdiff2).item())
        print(f"{h[i + 1]}\t {E0[i]}\t {E1[i]}\t {E2[i]}".expandtabs(20))
