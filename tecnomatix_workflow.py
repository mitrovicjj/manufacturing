# tecnomatix_workflow.py - ISPRAVLJENO
"""
Optimizovan workflow - batch processing umesto cycle-by-cycle
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.ml.predict_production import ProductionPredictor
from src.ml.features import build_inference_features


class TecnomatixWorkflow:
    """Kompletan ML workflow za Tecnomatix - OPTIMIZOVAN"""
    
    def __init__(self, output_dir=r"C:\temp\tecnomatix_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.predictor = ProductionPredictor()
        
    def process_simulation_data_fast(self, input_csv, output_prefix="run", max_cycles=None):
        """
        BRZI batch processing - generiše feature-e jednom za sve cikluse.
        """
        
        print("="*70)
        print(f"🚀 TECNOMATIX ML WORKFLOW - FAST BATCH MODE")
        print("="*70)
        
        # 1. Učitaj podatke
        df = pd.read_csv(input_csv)
        
        if max_cycles:
            df = df.head(max_cycles)
        
        print(f"\n✅ Loaded {len(df)} cycles from: {input_csv}")
        
        # 2. Proveri da li postoje potrebne kolone
        required_cols = ['cycle_time', 'temperature', 'vibration', 'pressure']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"\n❌ Missing required columns: {missing}")
            print(f"   Available: {list(df.columns)}")
            return None
        
        # Dodaj default vrednosti ako ne postoje
        if 'operator' not in df.columns:
            df['operator'] = 'op1'
        if 'maintenance_type' not in df.columns:
            df['maintenance_type'] = 'preventive'
        if 'cycle_id' not in df.columns:
            df['cycle_id'] = range(1, len(df) + 1)
        
        # 3. BATCH PROCESSING
        print(f"\n🔄 Generating features for all {len(df)} cycles (batch mode)...")
        
        df_features = build_inference_features(df)
        
        # Align sa model feature-ima
        if self.predictor.model_features is not None:
            missing_cols = set(self.predictor.model_features) - set(df_features.columns)
            for col in missing_cols:
                df_features[col] = 0
            df_features = df_features[self.predictor.model_features]
        
        print(f"✅ Features generated! Shape: {df_features.shape}")
        
        # 4. Batch predikcija
        print(f"\n🎯 Running batch predictions...")
        
        risk_scores = self.predictor.model.predict_proba(df_features)[:, 1]
        
        # Kreiraj rezultate
        results = pd.DataFrame({
            'cycle_id': df['cycle_id'].values,
            'risk_score': risk_scores,
            'maintenance_trigger': risk_scores >= 0.35,
            'priority': ['HIGH' if r > 0.6 else 'MEDIUM' if r > 0.35 else 'LOW' for r in risk_scores]
        })
        
        print(f"✅ Predictions complete!")
        
        # 5. Snimi rezultate
        output_csv = self.output_dir / f"{output_prefix}_predictions.csv"
        results.to_csv(output_csv, index=False)
        print(f"\n💾 Saved predictions: {output_csv}")
        
        # 6. Kreiraj vizualizaciju
        self.create_visualization(results, output_prefix)
        
        # 7. Kreiraj report
        self.create_report(results, output_prefix)
        
        # 8. Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total cycles: {len(results)}")
        print(f"   LOW risk:    {len(results[results['priority']=='LOW'])} ({len(results[results['priority']=='LOW'])/len(results)*100:.1f}%)")
        print(f"   MEDIUM risk: {len(results[results['priority']=='MEDIUM'])} ({len(results[results['priority']=='MEDIUM'])/len(results)*100:.1f}%)")
        print(f"   HIGH risk:   {len(results[results['priority']=='HIGH'])} ({len(results[results['priority']=='HIGH'])/len(results)*100:.1f}%)")
        print(f"   Maintenance triggers: {results['maintenance_trigger'].sum()} ({results['maintenance_trigger'].sum()/len(results)*100:.1f}%)")
        print(f"   Average risk: {results['risk_score'].mean():.4f}")
        print(f"   Max risk: {results['risk_score'].max():.4f}")
        
        print("\n" + "="*70)
        print(f"✅ All outputs saved to: {self.output_dir}")
        print("="*70)
        
        return results
    
    def create_visualization(self, df, prefix):
        """Kreira grafikone"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ML Analysis: {prefix}', fontsize=16, fontweight='bold')
        
        # 1. Risk score timeline
        colors = ['green' if p=='LOW' else 'orange' if p=='MEDIUM' else 'red' for p in df['priority']]
        axes[0,0].scatter(df['cycle_id'], df['risk_score'], c=colors, alpha=0.6, s=20)
        axes[0,0].axhline(0.35, color='orange', linestyle='--', alpha=0.7, label='Medium threshold')
        axes[0,0].axhline(0.60, color='red', linestyle='--', alpha=0.7, label='High threshold')
        axes[0,0].set_xlabel('Cycle ID')
        axes[0,0].set_ylabel('Risk Score')
        axes[0,0].set_title('Risk Score Timeline')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Priority distribution
        priority_counts = df['priority'].value_counts()
        colors_bar = ['green' if x=='LOW' else 'orange' if x=='MEDIUM' else 'red' for x in priority_counts.index]
        bars = axes[0,1].bar(priority_counts.index, priority_counts.values, color=colors_bar, edgecolor='black')
        axes[0,1].set_ylabel('Number of Cycles')
        axes[0,1].set_title('Risk Distribution')
        for bar in bars:
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        axes[0,1].grid(True, axis='y', alpha=0.3)
        
        # 3. Risk histogram
        axes[1,0].hist(df['risk_score'], bins=30, edgecolor='black', color='steelblue', alpha=0.7)
        axes[1,0].axvline(0.35, color='orange', linestyle='--', linewidth=2, label='Medium threshold')
        axes[1,0].axvline(0.60, color='red', linestyle='--', linewidth=2, label='High threshold')
        axes[1,0].set_xlabel('Risk Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Risk Score Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Maintenance triggers pie chart
        triggers = df['maintenance_trigger'].value_counts()
        labels = ['Normal Operation', 'Maintenance Required']
        colors_pie = ['green', 'red']
        axes[1,1].pie(triggers, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        axes[1,1].set_title('Maintenance Triggers')
        
        plt.tight_layout()
        output_png = self.output_dir / f"{prefix}_analysis.png"
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"📊 Saved visualization: {output_png}")
        plt.close()

    def create_report(self, df, prefix):
        """Kreira tekstualni izveštaj"""
        output_txt = self.output_dir / f"{prefix}_report.txt"
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"TECNOMATIX ML ANALYSIS REPORT - {prefix}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Cycles Analyzed: {len(df)}\n\n")
            
            f.write("RISK DISTRIBUTION:\n")
            f.write(f"  LOW Risk:    {len(df[df['priority']=='LOW']):>5} cycles ({len(df[df['priority']=='LOW'])/len(df)*100:>5.1f}%)\n")
            f.write(f"  MEDIUM Risk: {len(df[df['priority']=='MEDIUM']):>5} cycles ({len(df[df['priority']=='MEDIUM'])/len(df)*100:>5.1f}%)\n")
            f.write(f"  HIGH Risk:   {len(df[df['priority']=='HIGH']):>5} cycles ({len(df[df['priority']=='HIGH'])/len(df)*100:>5.1f}%)\n\n")
            
            f.write("MAINTENANCE ANALYSIS:\n")
            f.write(f"  Maintenance Triggers: {df['maintenance_trigger'].sum()}\n")
            f.write(f"  Maintenance Rate: {df['maintenance_trigger'].sum()/len(df)*100:.2f}%\n\n")
            
            f.write("RISK STATISTICS:\n")
            f.write(f"  Average Risk Score: {df['risk_score'].mean():.4f}\n")
            f.write(f"  Median Risk Score:  {df['risk_score'].median():.4f}\n")
            f.write(f"  Max Risk Score:     {df['risk_score'].max():.4f}\n")
            f.write(f"  Min Risk Score:     {df['risk_score'].min():.4f}\n\n")
            
            high_risk = df[df['maintenance_trigger']==True].sort_values('risk_score', ascending=False)
            if len(high_risk) > 0:
                f.write("HIGH RISK CYCLES (Top 20):\n")
                for i, (_, row) in enumerate(high_risk.head(20).iterrows(), 1):
                    f.write(f"  {i:>2}. Cycle {int(row['cycle_id']):>5}: Risk={row['risk_score']:.4f}\n")
        
        print(f"📄 Saved report: {output_txt}")


# MAIN
if __name__ == "__main__":
    workflow = TecnomatixWorkflow()
    
    print("\n🎯 Choose processing mode:")
    print("  1. Full dataset (may take time)")
    print("  2. Sample (first 500 cycles) - RECOMMENDED")
    
    choice = input("\nEnter choice (1/2) [default=2]: ").strip() or "2"
    
    if choice == "1":
        max_cycles = None
        prefix = "full_data"
    else:
        max_cycles = 500
        prefix = "sample_500"
    
    # Procesuj
    results = workflow.process_simulation_data_fast(
        input_csv="data/raw/logs_all_machines_v2.csv",
        output_prefix=prefix,
        max_cycles=max_cycles
    )
    
    print(f"\n✅ Check outputs in: C:\\temp\\tecnomatix_results\\")