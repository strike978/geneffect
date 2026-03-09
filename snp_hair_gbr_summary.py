import pandas as pd
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import csv


def analyze_group(group_name, traits, genotypes, hair_cols):
    if group_name == 'ALL_COMBINED':
        # Use all individuals from all populations
        traits_group = traits.copy()
    else:
        # Filter for specific group
        traits_group = traits[traits['group'] == group_name]

    group_ids = traits_group['individual'].tolist()
    genotypes_group = genotypes[genotypes['individual'].isin(group_ids)]
    hair_cols_group = [col for col in hair_cols if col in traits_group.columns]
    print(f"\n--- {group_name} diagnostics ---")
    print(f"Individuals: {len(group_ids)}")
    print(f"Hair columns: {hair_cols_group}")
    print(f"Trait sample:\n{traits_group.head()}\n")
    traits_hair_group = traits_group[['individual'] + hair_cols_group]
    snp_cols_group = [col for col in genotypes_group.columns if col not in [
        'source', 'group', 'group_full', 'individual']]
    snp_stats = []
    for snp in snp_cols_group:
        merged = pd.merge(traits_hair_group, genotypes_group[[
                          'individual', snp]], on='individual')
        if snp == snp_cols_group[0]:
            print(f"Merged sample for SNP {snp}:\n{merged.head()}\n")
        # Encode genotype for regression
        le = LabelEncoder()
        try:
            merged['snp_encoded'] = le.fit_transform(
                merged[snp].astype(str))
        except Exception as e:
            print(f"Encoding error for SNP {snp}: {e}")
            continue
        # For each hair color, run ANOVA and regression
        for hair_col in ['hair_light_blond', 'hair_black']:
            if hair_col in merged.columns:
                # ANOVA
                groups = [group[hair_col].dropna().values for name,
                          group in merged.groupby(snp)]
                if len(groups) > 1:
                    try:
                        f_stat, p_val = f_oneway(*groups)
                    except Exception as e:
                        f_stat, p_val = float('nan'), float('nan')
                else:
                    f_stat, p_val = float('nan'), float('nan')
                # Linear regression for R^2
                X = merged['snp_encoded'].values.reshape(-1, 1)
                y = merged[hair_col].values
                if len(set(X.flatten())) > 1:
                    model = LinearRegression()
                    model.fit(X, y)
                    r2 = model.score(X, y)
                else:
                    r2 = float('nan')
                snp_stats.append({
                    'SNP': snp,
                    'Hair_Color': hair_col,
                    'F_stat': f_stat,
                    'p_value': p_val,
                    'R2': r2
                })
    snp_stats_df = pd.DataFrame(snp_stats)

    # Apply Bonferroni correction for multiple testing
    unique_snps = snp_stats_df['SNP'].nunique()
    bonferroni_threshold = 0.05 / unique_snps
    snp_stats_df['bonferroni_corrected_p'] = snp_stats_df['p_value'] * unique_snps
    # Cap corrected p-values at 1.0
    snp_stats_df['bonferroni_corrected_p'] = snp_stats_df['bonferroni_corrected_p'].clip(
        upper=1.0)
    snp_stats_df['bonferroni_significant'] = snp_stats_df['bonferroni_corrected_p'] < 0.05

    snp_stats_df.sort_values(['Hair_Color', 'R2'], ascending=[
                             True, False], inplace=True)
    snp_stats_df.to_csv(
        f'snp_hair_anova_r2_summary_{group_name}.csv', index=False)

    print(f"\n=== BONFERRONI CORRECTION INFO for {group_name} ===")
    print(f"Number of SNPs tested: {unique_snps}")
    print(f"Bonferroni corrected threshold: p < {bonferroni_threshold:.2e}")
    print(
        f"Significant associations after correction: {snp_stats_df['bonferroni_significant'].sum()}")

    print(f"\nTop SNPs by R^2 for {group_name}:")
    for hair_col in ['hair_light_blond', 'hair_black']:
        top = snp_stats_df[snp_stats_df['Hair_Color'] == hair_col].head(10)
        print(f"\n{hair_col}:")
        print(top[['SNP', 'R2', 'p_value',
              'bonferroni_corrected_p', 'bonferroni_significant']])

    # Print top SNPs with interpretation
    print(
        f"\n=== KEY FINDINGS for {group_name} ===\nBonferroni-corrected results (conservative):")
    for hair_col in ['hair_light_blond', 'hair_black']:
        # Filter for Bonferroni significant results first
        bonf_significant = snp_stats_df[(snp_stats_df['Hair_Color'] == hair_col) & (
            snp_stats_df['bonferroni_significant'])]
        top_5 = bonf_significant.head(5)
        if not top_5.empty:
            print(f"\nTop Bonferroni-significant SNPs affecting {hair_col}:")
            for _, row in top_5.iterrows():
                significance = "***" if row['bonferroni_corrected_p'] < 0.001 else "**" if row[
                    'bonferroni_corrected_p'] < 0.01 else "*" if row['bonferroni_corrected_p'] < 0.05 else ""
                print(
                    f"  {row['SNP']}: R²={row['R2']:.3f} ({row['R2']*100:.1f}% variance), p_corrected={row['bonferroni_corrected_p']:.2e} {significance}")
        else:
            print(
                f"\nNo Bonferroni-significant SNPs for {hair_col} (all p_corrected > 0.05)")

    # Summary stats
    print(f"\n=== SUMMARY STATISTICS for {group_name} ===")
    raw_significant = snp_stats_df[snp_stats_df['p_value'] < 0.05]
    bonf_significant = snp_stats_df[snp_stats_df['bonferroni_significant']]

    print(f"Raw significant associations (p<0.05): {len(raw_significant)}")
    print(
        f"Bonferroni-corrected significant (p_corrected<0.05): {len(bonf_significant)}")
    print(
        f"False discovery rate: {(len(raw_significant) - len(bonf_significant)) / max(len(raw_significant), 1) * 100:.1f}%")

    if len(bonf_significant) > 0:
        best_r2 = bonf_significant['R2'].max()
        print(
            f"Strongest Bonferroni-significant effect: R²={best_r2:.3f} ({best_r2*100:.1f}% variance)")
        print(
            f"Average effect size (Bonferroni-significant): R²={bonf_significant['R2'].mean():.3f}")
    else:
        print("No associations survive Bonferroni correction")

    print(f"\n=== TRAIT STATISTICS for {group_name} ===\n")
    print(traits_group[hair_cols_group].describe())


def main():
    traits = pd.read_csv('merged_traits.csv')
    genotypes = pd.read_csv('merged_genotypes.csv')
    hair_cols = [
        'hair_light_blond', 'hair_dark_blond', 'hair_light_brown', 'hair_dark_brown', 'hair_black'
    ]

    print("=" * 60)
    print("ANALYZING ALL POPULATIONS COMBINED")
    print("=" * 60)
    print(f"Total individuals in dataset: {len(traits)}")
    print(f"Number of population groups: {traits['group'].nunique()}")
    print(f"Top 10 groups by size:")
    print(traits['group'].value_counts().head(10))
    print("\n")

    analyze_group('ALL_COMBINED', traits, genotypes, hair_cols)


if __name__ == '__main__':
    main()
