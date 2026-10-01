import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, average_precision_score
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize_scalar

def fit_temperature(logits, y):
    def nll(t):
        lp = logits / t - logsumexp(logits / t, axis=1, keepdims=True); return -lp[np.arange(len(y)), y].mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x

def ece(p, y, bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float); e = 0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def cls_metrics(logits, y, labels=None, T=1.0):
    p = softmax(logits / T, axis=1); pred = p.argmax(1)
    labels = labels if labels is not None else sorted(set(y))
    oh = np.eye(p.shape[1])[y]
    return dict(acc=(pred == y).mean(), f1=f1_score(y, pred, labels=labels, average="macro", zero_division=0),
                bacc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred), ece=ece(p, y),
                nll=-np.log(p[np.arange(len(y)), y] + 1e-12).mean(), brier=((p - oh) ** 2).sum(1).mean())

def ood_metrics(s_in, s_out):
    """scores: larger = more likely unknown"""
    y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]; s = np.r_[s_in, s_out]
    auroc = roc_auc_score(y, s); aupr = average_precision_score(y, s)
    thr = np.quantile(s_out, 0.05)   # 95% of unknowns flagged
    fpr = (s_in >= thr).mean()
    return dict(auroc=auroc, aupr=aupr, fpr95=fpr)

def conformal_quantile(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]

def pvalues(cal_scores, test_scores):
    """conformal p-values: (1 + #{cal >= s}) / (n+1); test_scores any shape"""
    c = np.sort(cal_scores); n = len(c)
    ge = n - np.searchsorted(c, test_scores, side="left")
    return (1 + ge) / (n + 1)
