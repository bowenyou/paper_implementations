import torch
import torch.optim as optim


class COCOB_Backprop(optim.Optimizer):

    def __init__(self, params, device, alpha=100, epsilon=1e-8):

        self.device = device
        self.alpha = alpha
        self.epsilon = epsilon
        defaults = dict(alpha=alpha, epsilon=epsilon)
        super(COCOB_Backprop, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['theta'] = torch.zeros_like(
                        p.data, device=self.device)
                    state['G'] = torch.zeros_like(p.data, device=self.device)
                    state['L'] = self.epsilon * \
                        torch.ones_like(p.data, device=self.device)
                    state['w'] = torch.zeros_like(p.data, device=self.device)
                    state['reward'] = torch.zeros_like(
                        p.data, device=self.device)

                state['L'] = state['L'].max(grad.abs())
                state['G'] = state['G'] + grad.abs()
                state['reward'] -= state['w'].mul(grad)
                state['reward'] = state['reward'].max(
                    torch.zeros(1, device=self.device))
                state['theta'] += grad

                temp = (self.alpha * state['L']).max(state['G'] + state['L'])
                beta = -state['theta'].div(state['L']).div(temp)

                p.data -= state['w'] - beta.mul(state['L'] + state['reward'])
                state['w'] = p.data

        return loss
