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


def get_update_direction_with_lo(grad_flat, current_net, lo):
    directions = []
    prev_ind = 0
    for param in current_net.parameters():
        flat_size = int(np.prod(list(param.size())))
        ndarray = grad_flat[prev_ind:prev_ind + flat_size].view(param.size()).detach()
        if ndarray.dim() > 1:  # inter-layer parameters
            ndarray = ndarray.numpy()
            direction_layer = lo.lo_oracle(-ndarray)
            direction_layer = torch.from_numpy(direction_layer).view(-1)
            direction_layer = param.view(-1) - direction_layer.double()
        else:  # parameters of activation functions
            direction_layer = -ndarray
        directions.append(direction_layer)
        prev_ind += flat_size
    direction = torch.cat(directions)
    # print(torch.norm(-grad_flat - direction))
    return direction


def a2c_storm_fw_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg, prev_grad, d_theta, i_iter, lo):

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
    learning_rate = 3e-2
    alpha = 1/(i_iter+2)**(2/3)
    # alpha = 1
    log_probs = policy_net.get_log_prob(states, actions)
    probs = torch.exp(log_probs)
    dice = probs/probs.detach()
    policy_loss = -(dice * advantages).mean()
    grad = torch.autograd.grad(policy_loss, policy_net.parameters(), retain_graph=True, create_graph=True)
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    grad_flat_current = flat(grad)

    jacob_d_theta = torch.autograd.grad(torch.dot(grad_flat_current, d_theta), policy_net.parameters(), retain_graph=True)
    jacob_d_theta = flat(jacob_d_theta)
    grad_flat = (1-alpha) * prev_grad + (1-alpha) * jacob_d_theta + alpha * grad_flat_current
    # direction = grad_flat/torch.norm(grad_flat)
    direction = get_update_direction_with_lo(grad_flat, policy_net, lo)
    # direction = grad_flat
    prev_params = get_flat_params_from(policy_net)
    updated_params = prev_params - learning_rate * direction
    set_flat_params_to(policy_net, updated_params)
    d_theta = updated_params - prev_params

    return grad_flat.detach(), d_theta.detach()

