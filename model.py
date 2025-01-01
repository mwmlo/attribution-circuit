import torch
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, FactoredMatrix

def compute_integrated_gradients_layer(model: HookedTransformer, input_x, input_baseline, hook_name):
    # Run model until hook point using input x to get activation a
    # Run model until hook point using baseline x' to get activation a' (baseline at layer)
    # Calculate mean gradient over small intervals between a and a'
    # Calculate integrated gradients between a and a'

    _, cache_x = model.run_with_cache(input_x)
    _, cache_b = model.run_with_cache(input_baseline)

    act_x = cache_x[hook_name]
    act_x.requires_grad = True
    act_b = cache_b[hook_name]
    act_b.requires_grad = True

    mean_grad = 0
    n = 100

    for i in range(1, n + 1):

        def ig_hook_fn(value, hook):
            assert torch.allclose(value, act_x)
            # value.requires_grad = True
            new_act = act_b + i / n * (value - act_b)
            return new_act
        
        logits = model.run_with_hooks(input_x, return_type="logits", fwd_hooks=[(hook_name, ig_hook_fn)])
        y = logits[:, 0].softmax(-1)[:, 1]
        (grad,) = torch.autograd.grad(y, act_x)
        mean_grad += grad / n

    integrated_gradients = (act_x - act_b) * mean_grad

    return integrated_gradients, mean_grad