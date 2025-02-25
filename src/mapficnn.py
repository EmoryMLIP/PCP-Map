import torch
from torch import nn
import torch.nn.functional as F
import torch.func as func
from lib.utils import AverageMeter


class MapFICNN(nn.Module):

    """
    Transport map parameterized by Fully Input Convex Neural Networks (Amos et al., 2017)
    Inverse Map gyinv: pushes the target marginal distribution to the reference distribution
    Direct Map gy: pushes the reference distribution to the target marginal distribution
    """

    def __init__(self, prior, ficnn):
        """
        :param prior: reference (Gaussian) distribution
        :param ficnn: FICNN network
        """
        super(MapFICNN, self).__init__()
        self.prior = prior
        self.ficnn = ficnn
        # trainable non-negative weight for quadratic term
        self.w1_ficnn = nn.Parameter(torch.zeros(1))
        self.w2_ficnn = nn.Parameter(torch.tensor(0.1))
        self.w3_ficnn = nn.Parameter(torch.tensor(0.1))

    def get_ficnn(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: output of the FICNN potential
        """
        # quadratic term for strong convexity
        quad = y*y / 2
        return F.softplus(self.w1_ficnn) * F.softplus(self.ficnn(y)) + (F.relu(self.w2_ficnn)+F.softplus(self.w3_ficnn)) * quad

    def compute_sum(self, y):
        return self.get_ficnn(y).sum()

    def gyinv(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: output of inverse map
        """
        out = self.get_ficnn(y)
        zy = torch.autograd.grad(out.sum(), y, create_graph=True)[0]
        return zy

    def gyinv_grad(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: gradient of the inverse map
        """
        hessian = func.vmap(func.hessian(self.compute_sum, argnums=0), in_dims=0)(y)
        return hessian

    def loglik_ficnn(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: log-likelihood
        """
        zy = self.gyinv(y)
        hessian = self.gyinv_grad(y)
        logprob = self.prior.log_prob(zy)
        if y.shape[1] > 1:
            # computing the log det of Hessian using eigenvalue decomposition
            eigen_val = torch.linalg.eigvalsh(hessian)
            logdet = torch.sum(torch.log(eigen_val), dim=1)
        else:
            logdet = torch.log(hessian)
        return logprob + logdet

    def gy(self, zy, tol, max_iter=1000000):
        """
        Generate samples from the target marginal distribution by solving a cvx optim problem
        using L-BFGS algorithm. Method borrowed from the CP-Flow paper (Huang et al., 2021)
        :param zy: samples from the reference (Gaussian) distribution
        :param tol: L-BFGS tolerance
        :param max_iter: maximal number of iterations per optimization step
        :return: generated samples from the target marginal distribution
        """
        count = AverageMeter()  # count number of loss calculations
        inv = zy.clone().detach().requires_grad_(True)

        def closure():
            out = self.get_ficnn(inv)
            if zy.shape[1] > 1:
                in_prod = torch.matmul(inv.unsqueeze(1), zy.unsqueeze(2)).squeeze(1)
            else:
                in_prod = inv * zy
            loss = torch.sum(out) - torch.sum(in_prod)
            count.update(1)
            inv.grad = torch.autograd.grad(loss, inv)[0].detach()
            return loss

        optimizer = torch.optim.LBFGS([inv], line_search_fn="strong_wolfe", max_iter=max_iter,
                                      tolerance_grad=tol, tolerance_change=tol)
        optimizer.step(closure)

        return inv, count.sum


if __name__ == "__main__":

    # checking the inverse

    from src.icnn import FICNN
    y1 = torch.randn(100, 1).view(-1, 1).requires_grad_(True)
    y2 = torch.randn(100, 3).requires_grad_(True)
    ficnn1 = FICNN(1, 128, 1, 6)
    ficnn2 = FICNN(3, 128, 1, 6)
    map1 = MapFICNN(prior=None, ficnn=ficnn1)
    map2 = MapFICNN(prior=None, ficnn=ficnn2)
    zy_1 = map1.gyinv(y1)
    zy_2 = map2.gyinv(y2)
    y1_gen, num_evals1 = map1.gy(zy_1, tol=1e-12)
    y2_gen, num_evals2 = map2.gy(zy_2, tol=1e-12)
    err1 = torch.norm(y1 - y1_gen) / torch.norm(y1)
    err2 = torch.norm(y2 - y2_gen) / torch.norm(y2)
    print("Number of closure evaluations for y1: " + str(num_evals1))
    print("Number of closure evaluations for y2: " + str(num_evals2))
    print("Inversion Relative Error: " + str(err1.item()))
    print("Block Inversion Relative Error: " + str(err2.item()))

    # grad check
    fv = map2.compute_sum(y2)
    zyy_2 = map2.gyinv_grad(y2)
    dy2 = torch.rand_like(y2)
    dzy2dy2 = torch.sum(zy_2 * dy2)
    dzyy2dy2 = torch.matmul(torch.matmul(dy2.unsqueeze(-1).transpose(2, 1), zyy_2), dy2.unsqueeze(-1)).sum()
    h, E0, E1, E2 = [1.0], [], [], []
    print("h\t E0 \t E1 \t E2".expandtabs(20))
    for i in range(11):
        h.append(h[i] / 2)
        fvt = map2.compute_sum(y2 + h[i + 1] * dy2)
        fdiff = fvt - fv
        fdiff1 = fdiff - h[i + 1] * dzy2dy2
        fdiff2 = fdiff1 - 0.5 * (h[i + 1] ** 2) * dzyy2dy2
        E0.append(torch.abs(fdiff).item())
        E1.append(torch.abs(fdiff1).item())
        E2.append(torch.abs(fdiff2).item())
        print(f"{h[i + 1]}\t {E0[i]}\t {E1[i]}\t {E2[i]}".expandtabs(20))
