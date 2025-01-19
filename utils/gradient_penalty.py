import torch


# def calc_gradients_input(x, y_pred):
#     gradients = torch.autograd.grad(
#         outputs=y_pred,
#         inputs=x,
#         grad_outputs=torch.ones_like(y_pred),
#         create_graph=True,
#         #allow_unused=True  # Add this line
#     )[0]
#
#     gradients = gradients.flatten(start_dim=1)
#
#     return gradients
#
#
# def calc_gradient_penalty(x, y_pred):
#
#     gradients = calc_gradients_input(x, y_pred)
#
#     # L2 norm
#     grad_norm = gradients.norm(2, dim=1)
#
#     # Two sided penalty
#     gradient_penalty = ((grad_norm - 1) ** 2).mean()
#
#     return gradient_penalty

def calc_gradient_penalty(x, y_pred_sum):
    gradients = torch.autograd.grad(
        outputs=y_pred_sum,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred_sum),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty