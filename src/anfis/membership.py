"""
ANFIS Membership Functions
==========================
Membership funkcije i DOMAIN-AWARE inicijalizacija MF parametara,
ekstrahovano iz ANFISAdvanced._initialize_membership_functions i MF metoda.
"""

import numpy as np


def gaussian_mf(x, center, sigma):
    """Gaussian membership function"""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def bell_mf(x, center, width, slope):
    """Generalized Bell membership function"""
    return 1 / (1 + np.abs((x - center) / width) ** (2 * slope))


def trapezoid_mf(x, a, b, c, d):
    """Trapezoidal membership function"""
    return np.maximum(0, np.minimum(
        np.minimum((x - a) / (b - a + 1e-10), 1),
        (d - x) / (d - c + 1e-10)
    ))


def initialize_mf_params(
    n_inputs,
    n_mfs_per_input,
    mf_type,
    input_ranges,
    domain_knowledge,
    use_domain_knowledge=True
):
    """
    DOMAIN-AWARE inicijalizacija MF parametara.
    Ovo je refaktorisana verzija _initialize_membership_functions iz ANFISAdvanced.

    Args:
        n_inputs: broj input-a
        n_mfs_per_input: lista broja MF-ova po inputu
        mf_type: 'gaussian' | 'bell' | 'trapezoid'
        input_ranges: lista (min, max) po inputu
        domain_knowledge: lista dict-ova sa threshold-ima
        use_domain_knowledge: da li koristiti domain-aware init

    Returns:
        mf_params: list, dužine n_inputs, svaki element je (n_mfs, param_dim)
    """
    mf_params = []

    print("\n" + "="*70)
    print("🔧 MEMBERSHIP FUNCTION INITIALIZATION (Domain-Aware)")
    print("="*70)

    for i in range(n_inputs):
        n_mfs = n_mfs_per_input[i]
        input_min, input_max = input_ranges[i]
        domain = domain_knowledge[i] if i < len(domain_knowledge) else None

        if mf_type == 'gaussian' and n_mfs == 3 and use_domain_knowledge and domain:
            # DOMAIN-AWARE INICIJALIZACIJA (gaussian, 3 MF-a)
            safe = domain['safe_max']
            warn = domain['warning']
            crit = domain['critical']

            # Centers bazirani na threshold-ima
            centers = np.array([
                (input_min + safe) / 2,       # LOW: centar u sigurnoj zoni
                (warn + crit) / 2,           # MEDIUM: između warning i critical
                (crit + input_max) / 2       # HIGH: iznad critical
            ])

            # Adaptive widths - kritična zona je UŽA!
            sigmas = np.array([
                (safe - input_min) * 0.6,    # LOW: široka (normalna operacija)
                (crit - warn) * 0.5,         # MEDIUM: uža (kritična zona!)
                (input_max - crit) * 0.6     # HIGH: široka (jasno loše)
            ])

            params = np.column_stack([centers, sigmas])

            # Debug info
            print(f"\n📊 {domain['name']} ({domain['unit']}):")
            print(f"   Standard: {domain['standard']}")
            print(f"   Range: [{input_min:.3f}, {input_max:.3f}]")
            print(f"   Thresholds: safe≤{safe:.3f}, warn={warn:.3f}, crit={crit:.3f}")
            print(f"\n   MF Parameters:")
            for j, term in enumerate(['LOW', 'MEDIUM', 'HIGH']):
                coverage_min = max(input_min, centers[j] - 2*sigmas[j])
                coverage_max = min(input_max, centers[j] + 2*sigmas[j])
                print(
                    f"   {term:8s}: center={centers[j]:6.3f}, σ={sigmas[j]:5.3f} "
                    f"(covers: {coverage_min:.2f}-{coverage_max:.2f})"
                )

        elif mf_type == 'bell' and n_mfs == 3 and use_domain_knowledge and domain:
            # DOMAIN-AWARE INICIJALIZACIJA (bell, 3 MF-a)
            safe = domain['safe_max']
            warn = domain['warning']
            crit = domain['critical']

            centers = np.array([
                (input_min + safe) / 2,
                (warn + crit) / 2,
                (crit + input_max) / 2
            ])

            widths = np.array([
                (safe - input_min) * 0.5,
                (crit - warn) * 0.4,
                (input_max - crit) * 0.5
            ])

            slopes = np.array([2.0, 3.0, 2.0])  # MEDIUM je strmiji!

            params = np.column_stack([centers, widths, slopes])

            print(f"\n📊 {domain['name']} ({domain['unit']}) - Bell MFs:")
            print(f"   Thresholds: safe≤{safe:.3f}, warn={warn:.3f}, crit={crit:.3f}")

        else:
            # FALLBACK: uniformna raspodela (bez domain knowledge)
            input_span = input_max - input_min

            if mf_type == 'gaussian':
                centers = np.linspace(
                    input_min + 0.1*input_span,
                    input_max - 0.1*input_span,
                    n_mfs
                )
                sigmas = np.ones(n_mfs) * (input_span / (2 * n_mfs))
                params = np.column_stack([centers, sigmas])

            elif mf_type == 'bell':
                centers = np.linspace(
                    input_min + 0.1*input_span,
                    input_max - 0.1*input_span,
                    n_mfs
                )
                widths = np.ones(n_mfs) * (input_span / (2 * n_mfs))
                slopes = np.ones(n_mfs) * 2.0
                params = np.column_stack([centers, widths, slopes])

            elif mf_type == 'trapezoid':
                params = np.zeros((n_mfs, 4))
                step = input_span / (n_mfs - 1) if n_mfs > 1 else input_span
                for j in range(n_mfs):
                    center = input_min + j * step
                    width = step * 0.6
                    params[j] = [
                        max(input_min, center - width),
                        center - width/2,
                        center + width/2,
                        min(input_max, center + width)
                    ]
            else:
                raise ValueError(f"Unsupported mf_type: {mf_type}")

            if domain is not None:
                print(f"\n⚠️  {domain['name']}: Using uniform distribution (fallback)")

        mf_params.append(params)

    print("\n" + "="*70)
    return mf_params