"""Global probability temperature; fitting uses development data only."""
import numpy as np


def apply_temperature(probabilities,temperature):
    if not np.isfinite(temperature) or temperature<=0: raise ValueError('Temperature must be positive')
    p=np.asarray(probabilities,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid probabilities')
    p=np.clip(p,1e-6,1-1e-6)
    logit=(np.log(p)-np.log1p(-p))/temperature
    return (1/(1+np.exp(-np.clip(logit,-80,80)))).astype(np.float32)


def fit_temperature(probabilities,labels):
    from scipy.optimize import minimize_scalar
    p,y=np.asarray(probabilities),np.asarray(labels)
    if p.shape!=y.shape or not np.isin(y,[0,1]).all(): raise ValueError('Calibration shapes/labels invalid')
    def objective(log_t):
        adjusted=np.clip(apply_temperature(p,np.exp(log_t)).astype(float),1e-6,1-1e-6)
        return float(-(y*np.log(adjusted)+(1-y)*np.log1p(-adjusted)).mean())
    result=minimize_scalar(objective,bounds=(-3,3),method='bounded')
    return float(np.exp(result.x))


def apply_affine(probabilities, *, temperature=1., bias=0.):
    """sigmoid((logit(p) + bias) / T); scalar or one bias per attribute."""
    p = np.asarray(probabilities, dtype=float)
    b = np.asarray(bias, dtype=float)
    if p.ndim != 2 or not all(p.shape) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be a nonempty finite matrix in [0,1]')
    if np.ndim(temperature) != 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be a finite positive scalar')
    if b.shape not in [(), (p.shape[1],)] or not np.isfinite(b).all():
        raise ValueError('Bias must be finite, scalar or one value per attribute')
    if temperature == 1 and np.all(b == 0):
        return p.astype(np.float32)
    clipped = np.clip(p, 1e-6, 1-1e-6)
    logits = (np.log(clipped)-np.log1p(-clipped)+b)/temperature
    return (1/(1+np.exp(-np.clip(logits, -80, 80)))).astype(np.float32)


def fit_domain_temperature(probabilities, labels, domains):
    """Fit one T by mean domain BCE, not by source image frequency."""
    from scipy.optimize import minimize_scalar
    p, y, d = np.asarray(probabilities), np.asarray(labels), np.asarray(domains)
    apply_affine(p)  # Validate before the optimizer, even for a degenerate fold.
    if p.shape != y.shape or not np.isin(y, [0, 1]).all() or d.shape != (len(y),):
        raise ValueError('Calibration labels/domains must match probabilities')
    groups = [d == domain for domain in np.unique(d)]
    def objective(log_t):
        adjusted = np.clip(apply_affine(p, temperature=np.exp(log_t)).astype(float), 1e-6, 1-1e-6)
        loss = -(y*np.log(adjusted)+(1-y)*np.log1p(-adjusted))
        return float(np.mean([loss[group].mean() for group in groups]))
    result = minimize_scalar(objective, bounds=(-3, 3), method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Domain temperature optimization failed')
    # Identity is the reference even if numerical optimization returns a worse fit.
    return float(np.exp(result.x)) if result.fun < objective(0) else 1.
