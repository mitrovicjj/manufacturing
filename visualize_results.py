"""
Vizualizuje rezultate iz output CSV-a
"""
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_CSV = r"C:\temp\cycles_output.csv"

# Učitaj rezultate
df = pd.read_csv(OUTPUT_CSV)

# Kreiraj grafikon
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Graf 1: Risk score tokom vremena
colors = ['green' if p == 'LOW' else 'orange' if p == 'MEDIUM' else 'red' 
          for p in df['priority']]

ax1.plot(df['cycle_id'], df['risk_score'], 'o-', color='gray', linewidth=2, markersize=8)
ax1.scatter(df['cycle_id'], df['risk_score'], c=colors, s=100, zorder=5)
ax1.axhline(y=0.35, color='orange', linestyle='--', label='Medium threshold (35%)')
ax1.axhline(y=0.60, color='red', linestyle='--', label='High threshold (60%)')
ax1.set_xlabel('Cycle ID', fontsize=12)
ax1.set_ylabel('Risk Score', fontsize=12)
ax1.set_title('Risk Score per Cycle', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Graf 2: Broj ciklusa po prioritetu
priority_counts = df['priority'].value_counts()
colors_bar = ['green', 'orange', 'red']
ax2.bar(priority_counts.index, priority_counts.values, color=colors_bar)
ax2.set_xlabel('Priority Level', fontsize=12)
ax2.set_ylabel('Number of Cycles', fontsize=12)
ax2.set_title('Cycle Distribution by Risk Priority', fontsize=14, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

# Dodaj vrednosti na barove
for i, (priority, count) in enumerate(priority_counts.items()):
    ax2.text(i, count + 0.1, str(count), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(r'C:\temp\risk_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: C:\\temp\\risk_analysis.png")
plt.show()