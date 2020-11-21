# Optimized from @janfreyberg, version 0.1.1

from torch.autograd import Function
from torch.nn import Module


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, lambda_):
        '''

        :param ctx:
        :param input_:
        :param lambda_: float, weight of this additionnal loss
        :return:
        '''
        # Saving input in context, if necessary for gradient computation
        # Here, no need
        # ctx.save_for_backward(input_)
        # Saving lambda
        ctx.lambda_ = lambda_
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
