"""
ANFIS Core Module
=================
Glavna klasa ANFISAdvanced sa forward pass metodama.
Koristi eksterne module za layers, membership functions, i utils.
"""

import numpy as np
from typing import Literal, List, Tuple

# Import modula
import torch
from src.anfis.config import ANFISConfig
from src.anfis.membership import initialize_mf_params
from src.anfis.utils import generate_rule_base, setup_linguistic_terms
import src.anfis.layers as layers


class ANFISAdvanced:
    """
    Napredni ANFIS (Adaptive Neuro-Fuzzy Inference System)
    
    Features:
    - Domain-aware inicijalizacija (bazirana na industrijskim standardima)
    - Različiti tipovi membership functions
    - Adaptivne širine (kritične zone su uže)
    - Config-driven architecture
    """
    
    def __init__(self, config: ANFISConfig = None):
        """
        Args:
            config: ANFISConfig instance (ako None, koristi default)
        """
        # Load config
        if config is None:
            config = ANFISConfig()
        
        self.config = config
        
        # Učitaj parametre iz config-a
        self.n_inputs = config.N_INPUTS
        self.n_mfs_per_input = config.N_MFS_PER_INPUT
        self.mf_type = config.MF_TYPE
        self.use_domain_knowledge = config.USE_DOMAIN_KNOWLEDGE
        
        # Input ranges i domain knowledge
        self.input_ranges = config.FEATURE_RANGES
        self.domain_knowledge = config.DOMAIN_KNOWLEDGE
        
        # Lingvistički termini
        self.linguistic_terms = setup_linguistic_terms(config.LINGUISTIC_TERMS, self.n_mfs_per_input)
        
        # Ukupan broj pravila
        self.n_rules = int(np.prod(self.n_mfs_per_input))
        
        # INICIJALIZACIJA
        # Layer 1: Membership function parameters (iz anfis_membership.py)
        self.mf_params = initialize_mf_params(
            n_inputs=self.n_inputs,
            n_mfs_per_input=self.n_mfs_per_input,
            mf_type=self.mf_type,
            input_ranges=self.input_ranges,
            domain_knowledge=self.domain_knowledge,
            use_domain_knowledge=self.use_domain_knowledge
        )
        
        # Generate rule base (iz anfis_utils.py)
        self.rule_base = generate_rule_base(self.n_mfs_per_input)
        
        # Layer 4: Consequent parameters (random init)
        self.consequent_params = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.1
        
        # Print initialization info
        self._print_initialization()
    
    
    # ==========================================
    # FORWARD PASS METHODS
    # ==========================================
    
    def forward(self, X):
        """
        COMPLETE FORWARD PASS - Layer 1 → 2 → 3 → 4 → 5
        
        Args:
            X: Input matrix shape (n_samples, n_inputs)
        
        Returns:
            output: Final ANFIS output shape (n_samples,)
            intermediates: Dict sa intermediate rezultatima (za debugging)
        """
        # Layer 1: Fuzzification
        mu = layers.layer1_fuzzification(self, X)
        
        # Layer 2: Rule firing
        w = layers.layer2_rule_firing(self, mu)
        
        # Layer 3: Normalization
        w_bar = layers.layer3_normalization(w)
        
        # Layer 4: Consequent
        f = layers.layer4_consequent(self, X, w_bar)
        
        # Layer 5: Output
        output = layers.layer5_output(w_bar, f)
        
        # Sačuvaj intermediate rezultate za debugging
        intermediates = {
            'mu': mu,           # Membership degrees
            'w': w,             # Firing strengths
            'w_bar': w_bar,     # Normalized weights
            'f': f,             # Rule outputs
            'output': output    # Final output
        }
        
        return output, intermediates
    
    
    def forward_verbose(self, X, sample_idx=0):
        """
        Verbose forward pass sa detaljnim ispisom svih slojeva.
        
        Args:
            X: Input data
            sample_idx: Index sample-a za prikaz
        """
        print(f"\n{'='*70}")
        print(f"🔮 ANFIS FORWARD PASS - Sample {sample_idx}")
        print(f"{'='*70}")
        
        # Input
        print(f"\n📥 INPUT:")
        feature_names = self.config.FEATURE_NAMES
        for i, name in enumerate(feature_names[:self.n_inputs]):
            print(f"   {name:15s} = {X[sample_idx, i]:.4f}")
        
        # Forward pass
        output, inter = self.forward(X)
        
        # Layer 1
        print(f"\n🔵 LAYER 1: FUZZIFICATION")
        for i, name in enumerate(feature_names[:self.n_inputs]):
            n_mfs = self.n_mfs_per_input[i]
            print(f"   {name}:")
            for j in range(n_mfs):
                term = self.linguistic_terms[j] if j < len(self.linguistic_terms) else f"MF_{j}"
                print(f"      μ_{term:8s} = {inter['mu'][i][sample_idx, j]:.4f}")
        
        # Layer 2
        print(f"\n🔵 LAYER 2: RULE FIRING")
        top_rules = 3
        w_sorted = sorted(enumerate(inter['w'][sample_idx]), key=lambda x: x[1], reverse=True)
        print(f"   Top {top_rules} firing rules:")
        for rank, (rule_idx, strength) in enumerate(w_sorted[:top_rules], 1):
            from utils import rule_to_string
            rule_str = rule_to_string(self, self.rule_base[rule_idx])
            print(f"   {rank}. w{rule_idx+1:2d} = {strength:.4f} | {rule_str}")
        
        # Layer 3
        print(f"\n🔵 LAYER 3: NORMALIZATION")
        w_sum = inter['w'][sample_idx].sum()
        print(f"   Σ(wi) = {w_sum:.4f}")
        print(f"   Top {top_rules} normalized weights:")
        for rank, (rule_idx, _) in enumerate(w_sorted[:top_rules], 1):
            w_bar_val = inter['w_bar'][sample_idx, rule_idx]
            print(f"   {rank}. w̄{rule_idx+1:2d} = {w_bar_val:.4f} ({w_bar_val*100:.1f}%)")
        
        # Layer 4
        print(f"\n🔵 LAYER 4: CONSEQUENT")
        print(f"   Top {top_rules} rule outputs:")
        for rank, (rule_idx, _) in enumerate(w_sorted[:top_rules], 1):
            f_val = inter['f'][sample_idx, rule_idx]
            print(f"   {rank}. f{rule_idx+1:2d} = {f_val:.4f}")
        
        # Layer 5
        print(f"\n🔵 LAYER 5: WEIGHTED SUM")
        print(f"   Calculating: output = Σ(w̄i · fi)")
        contributions = []
        for rank, (rule_idx, _) in enumerate(w_sorted[:top_rules], 1):
            w_bar_val = inter['w_bar'][sample_idx, rule_idx]
            f_val = inter['f'][sample_idx, rule_idx]
            contrib = w_bar_val * f_val
            contributions.append((rule_idx, contrib))
            print(f"   Rule {rule_idx+1:2d}: {w_bar_val:.4f} × {f_val:.4f} = {contrib:.4f}")
        
        total_contrib = sum(c for _, c in contributions)
        print(f"   Top 3 contribution: {total_contrib:.4f}")
        print(f"   All rules total: {output[sample_idx]:.4f}")
        
        # Final
        print(f"\n📤 FINAL OUTPUT: {output[sample_idx]:.4f}")
        
        # Interpretacija
        risk_label = self.config.get_risk_label(output[sample_idx])
        print(f"   Risk Level: {risk_label}")
        print(f"{'='*70}\n")
        
        return output, inter
    
    
    def predict(self, X):
        """
        Inference na novim podacima.
        
        Args:
            X: Input data (NumPy array)
        
        Returns:
            predictions: Output (NumPy array)
        """
        output, _ = self.forward(X)
        return output
    

    # ==========================================
# PREMISE TRAINING ENABLEMENT
# ==========================================

    def enable_premise_training(self):
        """
        Omogući premise training (MF centers i widths postaju trainable).
        Pozovi PRIJE train_hybrid() ako želiš da treniraš premise parametre.
        """
        print("\n🔧 Enabling premise training...")
        
        # Premise parameters (Layer 1) - MF centers i widths
        self.mf_params_torch = []
        for i in range(self.n_inputs):
            params_tensor = torch.tensor(
                self.mf_params[i],  # (n_mfs, 2) = [center, sigma]
                dtype=torch.float32,
                requires_grad=True  # ← KLJUČNO: Omogućava gradijente!
            )
            self.mf_params_torch.append(params_tensor)
        
        print(f"✅ Premise trainable: {len(self.mf_params_torch)} input groups")
        print(f"✅ Total premise params: {sum(p.numel() for p in self.mf_params_torch)}")

    def get_trainable_params(self):
        """
        Vrati listu trainable parametara za optimizer.
        Premise: lr=1e-3 (sporije, preserve interpretability)
        Consequent: lr=1e-2 (brže, data-driven)
        """
        if not hasattr(self, 'mf_params_torch'):
            self.enable_premise_training()
        
        params = []
        
        # Premise parameters (slower learning)
        for mf_params in self.mf_params_torch:
            params.append({'params': mf_params, 'lr': 1e-3})
        
        # Consequent parameters (faster learning)
        if not hasattr(self, 'consequent_params_torch'):
            self.consequent_params_torch = torch.tensor(
                self.consequent_params, dtype=torch.float32, requires_grad=True
            )
        params.append({'params': self.consequent_params_torch, 'lr': 1e-2})
        
        return params

    def sync_params_from_torch(self):
        """
        Sinhronizuj PyTorch parametre nazad u NumPy nakon treninga.
        """
        # Premise
        self.mf_params = [p.detach().cpu().numpy() for p in self.mf_params_torch]
        
        # Consequent  
        self.consequent_params = self.consequent_params_torch.detach().cpu().numpy()
        
        print("✅ Params synced from PyTorch → NumPy")

        
    def set_premise_training(self, enable: bool = True):
        """
        Kontroliraj premise training:
        enable=True: Premise trainable (adaptivni MF)
        enable=False: Premise fixed (domain knowledge only)
        """
        self.premise_training_enabled = enable
        
        if enable:
            print("🔧 Premise training ENABLED (adaptivni MF centri/širine)")
            self.enable_premise_training()
        else:
            print("🔒 Premise training DISABLED (fixed ISO/OSHA init)")
            if hasattr(self, 'mf_params_torch'):
                for p in self.mf_params_torch:
                    p.requires_grad = False
        
        return self.premise_training_enabled
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    def _print_initialization(self):
        """Info o inicijalizaciji"""
        print("\n" + "="*70)
        print("✅ ANFIS ADVANCED INITIALIZED")
        print("="*70)
        print(f"📊 Configuration:")
        print(f"   - Inputs: {self.n_inputs}")
        print(f"   - MFs per input: {self.n_mfs_per_input}")
        print(f"   - Total rules: {self.n_rules}")
        print(f"   - MF type: {self.mf_type}")
        print(f"   - Domain-aware: {self.use_domain_knowledge}")
        print(f"\n🏷️  Linguistic terms: {', '.join(self.linguistic_terms)}")
        print("="*70)