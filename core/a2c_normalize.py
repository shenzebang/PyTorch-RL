import torch
import numpy as np

def flat(grads):
    flat_grads = []
    for grad in grads:
        flat_grads.append(grad.view(-1))
    flat_grads = torch.cat(flat_grads)
    return flat_grads


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def a2c_normalize_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):

    """update critic"""
    values_pred = value_net(states)
    value_loss = (values_pred - returns).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    probs = torch.exp(log_probs)
    dice = probs/probs.detach()
    policy_loss = -(dice * advantages).mean()
    grad = torch.autograd.grad(policy_loss, policy_net.parameters(), retain_graph=True)
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    grad_flat = flat(grad)
    prev_params = get_flat_params_from(policy_net)
    direction = grad_flat/torch.norm(grad_flat)
    updated_params = prev_params - 3e-2 * direction
    set_flat_params_to(policy_net, updated_params)
    d_theta = updated_params - prev_params

    return grad_flat.detach(), d_theta.detach()
