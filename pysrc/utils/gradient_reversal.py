# Optimized from @janfreyberg, version 0.1.1

from torch.autograd import Function
from torch.nn import Module


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, sign, lambda_):
        '''

        :param ctx:
        :param input_:
        :param sign: -1 for gradient reversal layer, 1 for just
            an additionnal binary loss
        :param lambda_: absolute weight of this additionnal loss
        :return:
        '''
        # Saving input in context, if necessary for gradient computation
        # Here, no need
        # ctx.save_for_backward(input_)
        # Saving lambda
        ctx.lambda_ = sign * lambda_
        # Also, optimizing: no need to materialize gradients here
        ctx.set_materialize_grads(False)
        # Output is input
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        # Checking if we need to compute the gradient wrt input of layer
        if ctx.needs_input_grad[0]:
            grad_input = ctx.lambda_ * grad_output

        # Return as many gradients as there were inputs
        return grad_input, None, None


class GradientReversalLayer(Module):
    def __init__(self, lambda_=1, sign=-1):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__()
        self.lambda_ = lambda_
        self.sign = sign

    def forward(self, input_):
        return RevGrad.apply(input_, self.sign, self.lambda_)
