import torch
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN


class GradTestTrimap:

    """
    A check on Jacobian and Hessian computation using Taylor expansion.
    """

    def __init__(self, icnn):
        """
        :param icnn: input convex neural network (FICNN or PICNN)
        """
        self.icnn = icnn

    def grad_check_ficnn(self, x, num_test, tol, base=2, Hessian=True):
        """
        :param x: convex input
        :param num_test: number of down-factoring
        :param tol: error tolerance
        :param base: base for perturbation
        :param Hessian: boolean for Hessian calculation
        :return: Error and test results
        """
        import math
        E0, E1, E2 = [], [], []
        grad_check, hess_check = None, None
        flow = TriFlowFICNN(None, self.icnn)
        f0, df0, d2f0 = flow.get_ficnn(x), flow.g1inv(x), flow.g1inv_grad(x)  # initial evaluation

        dx = torch.randn_like(x)
        dx = dx / torch.norm(x)
        curvx = None

        if x.shape[1] > 1:
            if Hessian is not False:
                dx_sqz = dx.unsqueeze(1)
                left = torch.matmul(dx_sqz, d2f0)
                right = torch.matmul(left, torch.permute(dx_sqz, (0, 2, 1)))
                curvx = right.squeeze(1)
            dfdx = torch.sum(torch.mul(df0, dx), dim=1, keepdim=True)
        elif x.shape[1] == 1:
            if Hessian is not False:
                curvx = torch.sum(dx.unsqueeze(0) * d2f0 * dx.unsqueeze(0), dim=0)
            dfdx = df0 * dx
        else:
            raise Exception("Dimension Incorrect")

        for k in range(num_test):
            h = base ** (-k)
            ft = flow.get_ficnn(x + h * dx)
            E0.append(torch.norm(f0 - ft).item())
            E1.append(torch.norm(f0 + h * dfdx - ft).item())

            if curvx is not None:
                E2.append(torch.norm(f0 + h * dfdx + 0.5 * (h ** 2) * curvx - ft).item())

        E0, E1, E2 = torch.tensor(E0), torch.tensor(E1), torch.tensor(E2)

        grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / math.log2(base)) > (
                base - tol)) > num_test // 3).item()

        if curvx is not None:
            hess_check = (sum((torch.log2(E2[:-1] / E2[1:]) / math.log2(base)) > (
                    3 - tol)) > num_test // 3).item()

        if curvx is None:
            E = torch.cat((E0.view(-1, 1), E1.view(-1, 1)), 1)
        else:
            E = torch.cat((E0.view(-1, 1), E1.view(-1, 1), E2.view(-1, 1)), 1)

        return E, grad_check, hess_check

    def grad_check_picnn(self, x, y, num_test, tol, base=2, Hessian=True):
        """
        :param x: non-convex input
        :param y: convex input
        :param num_test: number of down-factoring
        :param tol: error tolerance
        :param base: base for perturbation
        :param Hessian: boolean for Hessian calculation
        :return: Error and test results
        """
        import math
        E0, E1, E2 = [], [], []
        grad_check, hess_check = None, None
        flow = TriFlowPICNN(None, self.icnn)
        f0, df0, d2f0 = flow.get_picnn(y, x), flow.g2inv(y, x), flow.g2inv_grad(y, x)  # initial evaluation

        dx2 = torch.randn_like(y)
        dx2 = dx2 / torch.norm(y)
        curvx = None

        if y.shape[1] > 1:
            if Hessian is not False:
                dx2_sqz = dx2.unsqueeze(1)
                left = torch.matmul(dx2_sqz, d2f0)
                right = torch.matmul(left, torch.permute(dx2_sqz, (0, 2, 1)))
                curvx = right.squeeze(1)
            dfdx2 = torch.sum(torch.mul(df0, dx2), dim=1, keepdim=True)
        elif y.shape[1] == 1:
            if Hessian is not False:
                curvx = torch.sum(dx2.unsqueeze(0) * d2f0 * dx2.unsqueeze(0), dim=0)
            dfdx2 = df0 * dx2
        else:
            raise Exception("Dimension Incorrect")

        for k in range(num_test):
            h = base ** (-k)
            ft = flow.get_picnn(y + h * dx2, x)
            E0.append(torch.norm(f0 - ft).item())
            E1.append(torch.norm(f0 + h * dfdx2 - ft).item())

            if curvx is not None:
                E2.append(torch.norm(f0 + h * dfdx2 + 0.5 * (h ** 2) * curvx - ft).item())

        E0, E1, E2 = torch.tensor(E0), torch.tensor(E1), torch.tensor(E2)

        grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / math.log2(base)) > (
                base - tol)) > num_test // 3).item()

        if curvx is not None:
            hess_check = (sum((torch.log2(E2[:-1] / E2[1:]) / math.log2(base)) > (
                    3 - tol)) > num_test // 3).item()

        if curvx is None:
            E = torch.cat((E0.view(-1, 1), E1.view(-1, 1)), 1)
        else:
            E = torch.cat((E0.view(-1, 1), E1.view(-1, 1), E2.view(-1, 1)), 1)

        return E, grad_check, hess_check


if __name__ == "__main__":

    from src.icnn import FICNN, PICNN

    torch.set_default_dtype(torch.float64)
    x = torch.randn(10, 2).requires_grad_(True)
    y = torch.randn(10, 3).requires_grad_(True)
    picnn = PICNN(3, 2, 256, 1, 3, act_v=torch.nn.ELU())
    ficnn = FICNN(2, 256, 1, 3)
    test_ficnn = GradTestTrimap(icnn=ficnn)
    test_picnn = GradTestTrimap(icnn=picnn)

    E_ficnn, grad_ficnn, hess_ficnn = test_ficnn.grad_check_ficnn(x, num_test=10, tol=0.01, Hessian=True)
    E_picnn, grad_picnn, hess_picnn = test_picnn.grad_check_picnn(x, y, num_test=10, tol=0.01, Hessian=True)

    print("--------------------------------------------------")
    print("FICNN Test Result:")
    print(E_ficnn)
    print(grad_ficnn, hess_ficnn)
    print("--------------------------------------------------")
    print("PICNN Test Result:")
    print(E_picnn)
    print(grad_picnn, hess_picnn)
    print("--------------------------------------------------")
