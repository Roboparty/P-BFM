"""Read-only local-rank Actor loss-gradient diagnostics (before clipping/Adam)."""
import itertools
import torch


def actor_gradient_metrics(terms, parameters):
    params = tuple(p for p in parameters if p.requires_grad)
    grads = {}
    metrics = {}
    for name, loss in terms.items():
        if not loss.requires_grad:
            continue
        grads[name] = tuple(None if g is None else g.detach() for g in
                            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True))
        sq = sum((g.float().square().sum() for g in grads[name] if g is not None), loss.detach().new_zeros(()))
        metrics[f'actor_gradient_local/{name}_norm'] = torch.sqrt(sq).detach()
    for a, b in itertools.combinations(grads, 2):
        dot = sum((x.float()*y.float()).sum() for x,y in zip(grads[a],grads[b])
                  if x is not None and y is not None)
        denom = metrics[f'actor_gradient_local/{a}_norm']*metrics[f'actor_gradient_local/{b}_norm']
        metrics[f'actor_gradient_local/{a}_{b}_cosine'] = (dot/denom.clamp_min(1e-20)).detach()
    for name in grads:
        if name != 'FB':
            metrics[f'actor_gradient_local/{name}_ratio_FB'] = (
                metrics[f'actor_gradient_local/{name}_norm']/metrics['actor_gradient_local/FB_norm'].clamp_min(1e-20))
    return metrics
