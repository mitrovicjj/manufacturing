"""
ANFIS Layers Module
===================
Implementacija svih 5 slojeva ANFIS arhitekture.
Svaki layer je nezavisna funkcija koja prima ANFIS model i relevantne inpute.
"""

import numpy as np


def layer1_fuzzification(anfis_model, X):
    """
    LAYER 1: FUZZIFICATION
    Pretvara crisp input vrednosti u fuzzy membership degrees.
    
    Args:
        anfis_model: Instance ANFISAdvanced klase
        X: Input matrix shape (n_samples, n_inputs)
    
    Returns:
        mu: List of arrays, mu[i] has shape (n_samples, n_mfs_per_input[i])
    """
    n_samples = X.shape[0]
    mu = []
    
    for i in range(anfis_model.n_inputs):
        n_mfs = anfis_model.n_mfs_per_input[i]
        mu_input = np.zeros((n_samples, n_mfs))
        
        for j in range(n_mfs):
            params = anfis_model.mf_params[i][j]
            
            if anfis_model.mf_type == 'gaussian':
                center, sigma = params
                mu_input[:, j] = _gaussian_mf(X[:, i], center, sigma)
            
            elif anfis_model.mf_type == 'bell':
                center, width, slope = params
                mu_input[:, j] = _bell_mf(X[:, i], center, width, slope)
            
            elif anfis_model.mf_type == 'trapezoid':
                a, b, c, d = params
                mu_input[:, j] = _trapezoid_mf(X[:, i], a, b, c, d)
        
        mu.append(mu_input)
    
    return mu


def layer2_rule_firing(anfis_model, mu):
    """
    LAYER 2: RULE FIRING
    Izračunava firing strength svakog pravila pomoću T-norm operatora (product).
    
    Args:
        anfis_model: Instance ANFISAdvanced klase
        mu: List of membership degree arrays from Layer 1
            mu[i] shape: (n_samples, n_mfs_per_input[i])
    
    Returns:
        w: Firing strengths shape (n_samples, n_rules)
            w[sample, rule] = firing strength pravila 'rule' za 'sample'
    """
    n_samples = mu[0].shape[0]
    w = np.ones((n_samples, anfis_model.n_rules))  # Inicijalizuj sa 1 (product identity)
    
    # Za svako pravilo, izračunaj product svih relevantnih MF-ova
    for rule_idx, rule in enumerate(anfis_model.rule_base):
        # rule = (temp_idx, vib_idx, cycle_idx, wear_idx)
        for input_idx, mf_idx in enumerate(rule):
            # Pomnožimo sa odgovarajućim membership degree-om
            w[:, rule_idx] *= mu[input_idx][:, mf_idx]
    
    return w


def layer3_normalization(w):
    """
    LAYER 3: NORMALIZATION
    Normalizuje firing strengths tako da suma = 1.0.
    
    Args:
        w: Firing strengths shape (n_samples, n_rules)
    
    Returns:
        w_bar: Normalized weights shape (n_samples, n_rules)
    """
    # Suma svih firing strengths po sample-u
    w_sum = w.sum(axis=1, keepdims=True)  # shape: (n_samples, 1)
    
    # Izbegni deljenje sa nulom
    w_sum = np.where(w_sum == 0, 1.0, w_sum)
    
    # Normalizacija
    w_bar = w / w_sum
    
    return w_bar


def layer4_consequent(anfis_model, X, w_bar):
    """
    LAYER 4: CONSEQUENT PARAMETERS
    Izračunava output svake rule koristeći linearnu funkciju (Takagi-Sugeno).
    
    fi = pi·x1 + qi·x2 + ri·x3 + si·x4 + ti
    
    Args:
        anfis_model: Instance ANFISAdvanced klase
        X: Input matrix shape (n_samples, n_inputs)
        w_bar: Normalized weights shape (n_samples, n_rules)
    
    Returns:
        f: Rule outputs shape (n_samples, n_rules)
    """
    n_samples = X.shape[0]
    f = np.zeros((n_samples, anfis_model.n_rules))
    
    # Za svako pravilo, primeni linearnu funkciju
    for i in range(anfis_model.n_rules):
        # consequent_params[i] = [p, q, r, s, t]
        # X·[p,q,r,s]ᵀ + t
        f[:, i] = np.dot(X, anfis_model.consequent_params[i, :-1]) + anfis_model.consequent_params[i, -1]
    
    return f


def layer5_output(w_bar, f):
    """
    LAYER 5: WEIGHTED SUM (Final output)
    Kombinuje sve rule outputs pomoću normalizovanih težina.
    
    output = Σ(w̄i · fi)
    
    Args:
        w_bar: Normalized weights shape (n_samples, n_rules)
        f: Rule outputs shape (n_samples, n_rules)
    
    Returns:
        output: Final output shape (n_samples,)
    """
    # Element-wise product, pa suma po pravilima
    output = (w_bar * f).sum(axis=1)
    
    return output


# ==========================================
# HELPER FUNCTIONS - Membership Functions
# ==========================================

def _gaussian_mf(x, center, sigma):
    """Gaussian membership function"""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _bell_mf(x, center, width, slope):
    """Generalized Bell membership function"""
    return 1 / (1 + np.abs((x - center) / width) ** (2 * slope))


def _trapezoid_mf(x, a, b, c, d):
    """Trapezoidal membership function"""
    return np.maximum(0, np.minimum(
        np.minimum((x - a) / (b - a + 1e-10), 1),
        (d - x) / (d - c + 1e-10)
    ))


# ==========================================
# VERBOSE LAYER FUNCTIONS (za debugging)
# ==========================================

def layer2_rule_firing_verbose(anfis_model, mu, sample_idx=0, top_k=5):
    """
    Verbose verzija Layer 2 - prikazuje top K pravila za debug.
    
    Args:
        anfis_model: Instance ANFISAdvanced
        mu: Membership degrees
        sample_idx: Koji sample da prikaže (default: prvi)
        top_k: Broj top pravila za prikaz
    """
    w = layer2_rule_firing(anfis_model, mu)
    
    print(f"\n🔥 RULE FIRING - Sample {sample_idx}:")
    print(f"   Total rules: {anfis_model.n_rules}")
    print(f"   Non-zero rules: {(w[sample_idx] > 1e-6).sum()}")
    
    # Sortiraj pravila po firing strength
    rule_strengths = [(i, w[sample_idx, i]) for i in range(anfis_model.n_rules)]
    rule_strengths.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   🔝 Top {top_k} firing rules:")
    for rank, (rule_idx, strength) in enumerate(rule_strengths[:top_k], 1):
        from utils import rule_to_string
        rule = anfis_model.rule_base[rule_idx]
        rule_str = rule_to_string(anfis_model, rule)
        bar = '█' * int(strength * 50)  # Visual bar
        print(f"   {rank}. w={strength:.4f} {bar}")
        print(f"      {rule_str}")
    
    return w