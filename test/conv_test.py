import torch
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN


class ConvTest:
    """
    Checks function convexity by minimizing the minimum of
    hessian / minimum of eigenvalues of hessian. If the value
    is positive then we claim convexity
    """
    def __init__(self, icnn):
        """
        :param icnn: neural network
        """
        self.icnn = icnn

    def conv_test_ficnn(self, x):
        """
        Convexity test for FICNN network
        :param x: the first i-1 / first block component of target sample x
        :return: minimum of hessian / eigenvalues of hessian
        """
        flow = TriFlowFICNN(None, self.icnn)

        def closure():
            optimizer.zero_grad()
            hessian = flow.g1inv_grad(x)
            if x.shape[1] > 1:
                loss = torch.min(torch.linalg.eigvalsh(hessian))
            else:
                loss = torch.min(hessian)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS(flow.parameters(), line_search_fn="strong_wolfe",
                                      max_iter=1000000)
        optimizer.step(closure)
        eig_min = closure()
        return eig_min

    def conv_test_picnn(self, x, y):
        """
        Convexity test for PICNN network
        :param x: non-input-convex component
        :param y: input-convex component
        :return: minimum of hessian / eigenvalues of hessian
        """
        flow = TriFlowPICNN(None, self.icnn)

        def closure():
            optimizer.zero_grad()
            hessian = flow.g2inv_grad(y, x)
            if y.shape[1] > 1:
                loss = torch.min(torch.linalg.eigvalsh(hessian))
            else:
                loss = torch.min(hessian)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS(flow.parameters(), line_search_fn="strong_wolfe",
                                      max_iter=1000000)
        optimizer.step(closure)
        eig_min = closure()
        return eig_min


if __name__ == "__main__":

    from src.icnn import PICNN
    x = torch.randn(100, 2).requires_grad_(True)
    y = torch.randn(100, 3).requires_grad_(True)
    picnn = PICNN(3, 2, 128, 128, 1, 6)
    ConvTest1 = ConvTest(icnn=picnn)
    eig = ConvTest1.conv_test_picnn(x, y)
    print("Minimum Eigenvalue:", eig)
