import torch
from torch import nn
import torch.nn.functional as F
from lib.utils import AverageMeter


class TriFlowFICNN(nn.Module):

    """
    The first component of the monotone block triangular transport map g
    parameterized by Fully Input Convex Neural Networks (Amos et al., 2017)
    Inverse Map g1inv: maps the target marginal distribution (over the conditional input) to
    the reference marginal distribution
    Direct Map g1: generates samples from the target marginal distribution using samples
    from the reference marginal distribution
    """

    def __init__(self, prior, ficnn):
        """
        :param prior: reference marginal distribution
        :param ficnn: FICNN network
        """
        super(TriFlowFICNN, self).__init__()
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
        quad = torch.norm(y, dim=1, keepdim=True) ** 2 / 2
        return F.softplus(self.w1_ficnn) * F.softplus(self.ficnn(y)) + (F.relu(self.w2_ficnn)+F.softplus(self.w3_ficnn)) * quad

    def g1inv(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: output of inverse map
        """
        out = self.get_ficnn(y)
        zy = torch.autograd.grad(out.sum(), y, create_graph=True)[0]
        return zy

    def g1inv_grad(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: gradient of the inverse map
        """
        if y.shape[1] > 1:
            y_grad = self.g1inv(y)
            hessian = []
            for i in range(y_grad.shape[1]):
                hessian.append(torch.autograd.grad(y_grad[:, i].sum(), y, create_graph=True)[0])
            hessian = torch.stack(hessian, dim=1)
        else:
            g1inv = self.g1inv(y)
            hessian = torch.autograd.grad(g1inv.sum(), y, create_graph=True)[0]
        return hessian

    def loglik_ficnn(self, y):
        """
        :param y: samples from the target marginal distribution
        :return: log-likelihood
        """
        zy = self.g1inv(y)
        hessian = self.g1inv_grad(y)
        logprob = self.prior.log_prob(zy)
        if y.shape[1] > 1:
            # computing the log det of Hessian using eigenvalue decomposition
            eigen_val = torch.linalg.eigvalsh(hessian)
            logdet = torch.sum(torch.log(eigen_val), dim=1)
        else:
            logdet = torch.log(hessian)
        return logprob + logdet

    def g1(self, zy, tol, max_iter=1000000):
        """
        Generate samples from the target marginal distribution by solving a cvx optim problem
        using L-BFGS algorithm. Method borrowed from the CP-Flow paper (Huang et al., 2021)
        :param zy: samples from the reference marginal distribution
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
    flow1 = TriFlowFICNN(prior=None, ficnn=ficnn1)
    flow2 = TriFlowFICNN(prior=None, ficnn=ficnn2)
    zy_1 = flow1.g1inv(y1)
    zy_2 = flow2.g1inv(y2)
    y1_gen, num_evals1 = flow1.g1(zy_1, tol=1e-12)
    y2_gen, num_evals2 = flow2.g1(zy_2, tol=1e-12)
    err1 = torch.norm(y1 - y1_gen) / torch.norm(y1)
    err2 = torch.norm(y2 - y2_gen) / torch.norm(y2)
    print("Number of closure evaluations for y1: " + str(num_evals1))
    print("Number of closure evaluations for y2: " + str(num_evals2))
    print("Inversion Relative Error: " + str(err1.item()))
    print("Block Inversion Relative Error: " + str(err2.item()))
