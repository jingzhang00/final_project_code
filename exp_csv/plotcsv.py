import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent

crp_files = list(script_dir.glob('CRP-DS/metrics_rep*.csv'))
pwlpv_files = list(script_dir.glob('PW-LPV/metrics_rep*.csv'))

if not crp_files:
    print("No CRP-DS files found!")
    exit()
if not pwlpv_files:
    print("No PW-LPV files found!")
    exit()

print(f"Found {len(crp_files)} CRP-DS files and {len(pwlpv_files)} PW-LPV files")


def process_experiment(files, exp_name):
    dfs = [pd.read_csv(f, index_col=0) for f in files]
    shapes = dfs[0].columns.tolist()
    metrics = dfs[0].index.tolist()

    all_values = {m: {sh: [] for sh in shapes} for m in metrics}
    for df in dfs:
        for m in metrics:
            for sh in shapes:
                all_values[m][sh].append(df.at[m, sh])

    print(f"\n=== {exp_name} RESULTS ===")
    mean_df = pd.DataFrame({
        m: [np.mean(all_values[m][sh]) for sh in shapes]
        for m in metrics
    }, index=shapes)
    print("Mean values:")
    print(mean_df.round(3))

    return all_values, shapes, metrics


crp_values, shapes, metrics = process_experiment(crp_files, "CRP-DS")
pwlpv_values, _, _ = process_experiment(pwlpv_files, "PW-LPV")

for m in metrics:
    plt.figure(figsize=(5, 3))

    crp_data = [crp_values[m][sh] for sh in shapes]
    pwlpv_data = [pwlpv_values[m][sh] for sh in shapes]

    all_crp_values = [val for shape_vals in crp_data for val in shape_vals]
    min_val = min(all_crp_values)
    max_val = max(all_crp_values)

    crp_normalized = [
        [(val - min_val) / (max_val - min_val) for val in shape_vals]
        for shape_vals in crp_data
    ]
    pwlpv_normalized = [
        [(val - min_val) / (max_val - min_val) for val in shape_vals]
        for shape_vals in pwlpv_data
    ]

    positions_crp = [i - 0.2 for i in range(1, len(shapes) + 1)]
    positions_pwlpv = [i + 0.2 for i in range(1, len(shapes) + 1)]

    bp1 = plt.boxplot(crp_normalized, positions=positions_crp, widths=0.3,
                      patch_artist=True, showfliers=False)
    bp2 = plt.boxplot(pwlpv_normalized, positions=positions_pwlpv, widths=0.3,
                      patch_artist=True, showfliers=False)

    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)

    # plt.title(f'{m} Comparison: CRP-DS vs PW-LPV\n(Normalized by CRP-DS range)',
    #           fontsize=14, fontweight='bold')
    # plt.xlabel('Shape', fontsize=12)
    plt.ylabel(f'{m}', fontsize=12)

    plt.xticks(range(1, len(shapes) + 1), shapes, rotation=45)

    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5,
                label='CRP-DS maximum')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5,
                label='CRP-DS minimum')

    # plt.legend([bp1["boxes"][0], bp2["boxes"][0], plt.Line2D([0], [0], color='gray', linestyle='--')],
    #            ['CRP-DS', 'PW-LPV', 'CRP-DS range'], loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"\n{m} - CRP-DS range: [{min_val:.3f}, {max_val:.3f}]")

    pwlpv_flat = [val for shape_vals in pwlpv_data for val in shape_vals]
    if any(val > max_val for val in pwlpv_flat):
        print(f"  PW-LPV has values above CRP-DS maximum (will show > 1.0)")
    if any(val < min_val for val in pwlpv_flat):
        print(f"  PW-LPV has values below CRP-DS minimum (will show < 0.0)")

    filename = f'{m.replace(" ", "_").replace("/", "_")}.pdf'
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")

    plt.show()

dummy_crp = plt.Rectangle((0,0), 1, 1, facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1)
dummy_pwlpv = plt.Rectangle((0,0), 1, 1, facecolor='lightcoral', alpha=0.7, edgecolor='black', linewidth=1)
dummy_baseline_max = plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.8, linewidth=2)
dummy_baseline_min = plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.8, linewidth=2)

legend = plt.legend([dummy_crp, dummy_pwlpv, dummy_baseline_max],
                   ['CRP-DS', 'PW-LPV', 'CRP-DS baseline (0-1 normalized)'],
                   loc='center', ncol=3, fontsize=14,
                   frameon=True, fancybox=True, shadow=True,
                   columnspacing=2, handletextpad=1)

# Remove axes
plt.gca().set_axis_off()

plt.savefig('legend.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: legend.pdf")
plt.show()

print("\n" + "=" * 50)
print("SUMMARY: Mean Performance Comparison")
print("=" * 50)

for m in metrics:
    print(f"\n{m}:")
    crp_means = [np.mean(crp_values[m][sh]) for sh in shapes]
    pwlpv_means = [np.mean(pwlpv_values[m][sh]) for sh in shapes]

    for i, sh in enumerate(shapes):
        improvement = ((crp_means[i] - pwlpv_means[i]) / crp_means[i]) * 100
        status = "↓ Better" if improvement > 0 else "↑ Worse"
        print(f"  {sh:15}: CRP-DS={crp_means[i]:7.3f}, PW-LPV={pwlpv_means[i]:7.3f} ({improvement:+5.1f}% {status})")