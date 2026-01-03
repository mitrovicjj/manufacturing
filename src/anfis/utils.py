"""
ANFIS Utility Functions
=======================
Helper funkcije: rule base generisanje, rule string, lingvistički termini,
i model save/load.
"""

import itertools
import pickle
import os


def setup_linguistic_terms(user_terms, n_mfs_per_input):
    """
    Automatski generiše lingvističke termine (refaktor _setup_linguistic_terms).

    Args:
        user_terms: lista termina ili None
        n_mfs_per_input: lista broja MF-ova po inputu

    Returns:
        lista termina (globalna, npr. ['LOW','MEDIUM','HIGH'])
    """
    if user_terms:
        return user_terms

    term_maps = {
        2: ['LOW', 'HIGH'],
        3: ['LOW', 'MEDIUM', 'HIGH'],
        5: ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
    }

    max_mfs = max(n_mfs_per_input)
    return term_maps.get(max_mfs, [f'MF_{i+1}' for i in range(max_mfs)])


def generate_rule_base(n_mfs_per_input):
    """
    Generiše sve moguće kombinacije pravila (refaktor _generate_rule_base).

    Za 36 pravila (3×3×2×2):
    - temp: 3 MF-a (LOW, MED, HIGH)
    - vib: 3 MF-a
    - cycle: 2 MF-a (LOW, HIGH)
    - wear: 2 MF-a

    Args:
        n_mfs_per_input: lista broja MF-ova po inputu

    Returns:
        rule_base: list of tuples, npr. [(0,0,0,0), ..., (2,2,1,1)]
    """
    ranges = [range(n) for n in n_mfs_per_input]
    rule_base = list(itertools.product(*ranges))

    print(f"\n🔧 RULE BASE GENERATION:")
    print(f"   Total rules: {len(rule_base)}")
    print(f"   MFs per input: {n_mfs_per_input}")
    print(f"\n   Sample rules:")

    from itertools import islice
    for idx, rule in enumerate(islice(rule_base, 5)):
        print(f"   Rule {idx+1:2d}: {rule}")

    if len(rule_base) > 5:
        print(f"   ... ({len(rule_base)-5} more rules)")

    return rule_base


def rule_to_string(anfis_model, rule_indices):
    """
    Konvertuje rule indices u čitljiv string.
    Refaktor _rule_to_string.

    Args:
        anfis_model: instance ANFISAdvanced (za feature names i linguistic terms)
        rule_indices: tuple indeksa (npr. (2, 1, 0, 1))

    Returns:
        String "IF temp=HIGH AND vib=MEDIUM AND cycle=LOW AND wear=HIGH"
    """
    feature_names = [
        d['name'] for d in anfis_model.domain_knowledge[:anfis_model.n_inputs]
    ]
    conditions = []

    for i, idx in enumerate(rule_indices):
        feature = feature_names[i] if i < len(feature_names) else f"Input_{i}"
        term = (
            anfis_model.linguistic_terms[idx]
            if idx < len(anfis_model.linguistic_terms)
            else f"MF_{idx}"
        )
        conditions.append(f"{feature}={term}")

    return "IF " + " AND ".join(conditions)


def save_model(anfis_model, filepath):
    """
    Sačuvaj trenirani model (mf_params, consequent_params, osnovni config).
    Minimalni wrapper oko pickle.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        'n_inputs': anfis_model.n_inputs,
        'n_mfs_per_input': anfis_model.n_mfs_per_input,
        'mf_type': anfis_model.mf_type,
        'input_ranges': anfis_model.input_ranges,
        'domain_knowledge': anfis_model.domain_knowledge,
        'linguistic_terms': anfis_model.linguistic_terms,
        'mf_params': anfis_model.mf_params,
        'consequent_params': anfis_model.consequent_params,
    }

    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

    print(f"\n✅ Model saved to: {filepath}")


def load_model(anfis_class, filepath):
    """
    Učitaj trenirani model i vrati instancu anfis_class (npr. ANFISAdvanced).

    Args:
        anfis_class: klasa koju instanciramo (ANFISAdvanced)
        filepath: path do .pkl fajla

    Returns:
        instanca anfis_class sa učitanim parametrima
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    # Kreiraj minimalni config-like objekat za inicijalizaciju
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        N_INPUTS=state['n_inputs'],
        N_MFS_PER_INPUT=state['n_mfs_per_input'],
        MF_TYPE=state['mf_type'],
        USE_DOMAIN_KNOWLEDGE=True,
        FEATURE_RANGES=state['input_ranges'],
        DOMAIN_KNOWLEDGE=state['domain_knowledge'],
        LINGUISTIC_TERMS=state['linguistic_terms'],
    )

    model = anfis_class(config=cfg)
    model.mf_params = state['mf_params']
    model.consequent_params = state['consequent_params']

    print(f"\n✅ Model loaded from: {filepath}")
    return model