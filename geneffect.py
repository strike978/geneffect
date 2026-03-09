import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_data
def load_data():
    """Load and cache the datasets"""
    traits = pd.read_csv('merged_traits.csv')
    genotypes = pd.read_csv('merged_genotypes.csv')
    return traits, genotypes


# @st.cache_data
def load_snp_gene_mapping():
    """Load and cache the SNP-to-gene mapping"""
    try:
        with open('snp_to_gene.json', 'r') as f:
            snp_to_gene = json.load(f)
        return snp_to_gene
    except FileNotFoundError:
        st.warning(
            "SNP-to-gene mapping file not found. Gene information will not be available.")
        return {}


def get_trait_columns(traits_df):
    """Get all available trait columns"""
    exclude_cols = ['source', 'group', 'group_full', 'individual']
    trait_cols = [col for col in traits_df.columns if col not in exclude_cols]

    # Categorize traits
    hair_traits = [col for col in trait_cols if col.startswith('hair_')]
    eye_traits = [col for col in trait_cols if col.startswith('eye_')]
    skin_traits = [col for col in trait_cols if col.startswith('skin_')]

    return {
        'Hair Color': hair_traits,
        'Eye Color': eye_traits,
        'Skin Color': skin_traits
    }


def run_snp_analysis(traits_df, genotypes_df, selected_traits, selected_groups, dominance_mode=False):
    """Run ANOVA + R² analysis for selected traits and populations"""

    # Load SNP-to-gene mapping
    snp_to_gene = load_snp_gene_mapping()

    # Filter populations
    if 'ALL_COMBINED' in selected_groups:
        traits_filtered = traits_df.copy()
    else:
        traits_filtered = traits_df[traits_df['group'].isin(selected_groups)]

    group_ids = traits_filtered['individual'].tolist()
    genotypes_filtered = genotypes_df[genotypes_df['individual'].isin(
        group_ids)]

    # Get SNP columns
    snp_cols = [col for col in genotypes_filtered.columns if col not in [
        'source', 'group', 'group_full', 'individual']]

    # Prepare traits data
    traits_data = traits_filtered[['individual'] + selected_traits]

    results = []
    progress_bar = st.progress(0)
    total_combinations = len(snp_cols) * len(selected_traits)
    current_progress = 0

    for snp in snp_cols:
        # Merge SNP data with traits
        snp_data = genotypes_filtered[['individual', snp]]
        merged = pd.merge(traits_data, snp_data, on='individual')

        for trait in selected_traits:
            if trait in merged.columns:
                # ANOVA - exclude missing genotypes
                merged_anova = merged[merged[snp] != '--'].copy()
                merged_anova = merged_anova.dropna(
                    subset=[snp])  # Remove NaN values too
                groups = [group[trait].dropna().values for name,
                          group in merged_anova.groupby(snp)]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    try:
                        f_stat, p_val = f_oneway(*groups)
                    except:
                        f_stat, p_val = float('nan'), float('nan')
                else:
                    f_stat, p_val = float('nan'), float('nan')

                # Linear regression for R² with proper genetic coding
                # Exclude missing genotypes for regression too
                merged_reg = merged[merged[snp] != '--'].copy()
                merged_reg = merged_reg.dropna(
                    subset=[snp])  # Remove NaN values too

                r2 = float('nan')
                if len(merged_reg) > 1 and len(set(merged_reg[snp])) > 1:
                    genotypes = merged_reg[snp].unique()
                    genotypes_str = [str(gt)
                                     for gt in genotypes if str(gt) != 'nan']
                    if len(genotypes_str) >= 2:
                        sorted_genotypes = sorted(genotypes_str)
                        genotype_map = {gt: i for i,
                                        gt in enumerate(sorted_genotypes)}
                        merged_reg['snp_additive'] = merged_reg[snp].astype(
                            str).map(genotype_map)

                        X = merged_reg['snp_additive'].values.reshape(-1, 1)
                        y = merged_reg[trait].values

                        if dominance_mode:
                            # Add dominance coding: heterozygote = 1, else 0
                            # Assumes biallelic SNPs (e.g., AA, AG, GG)
                            het_genotypes = [
                                gt for gt in sorted_genotypes if len(set(gt)) > 1]
                            merged_reg['snp_dominance'] = merged_reg[snp].astype(
                                str).apply(lambda x: 1 if x in het_genotypes else 0)
                            X = np.column_stack(
                                [merged_reg['snp_additive'].values, merged_reg['snp_dominance'].values])

                        if not np.isnan(y).all() and len(set(X.flatten())) > 1:
                            model = LinearRegression()
                            model.fit(X, y)
                            r2 = model.score(X, y)
                    # else r2 stays nan

                # Get allele information and effect direction
                # Exclude missing genotypes (--) and NaN from effect calculation
                merged_clean = merged[merged[snp] != '--'].copy()
                merged_clean = merged_clean.dropna(
                    subset=[snp])  # Remove NaN values too
                if len(merged_clean) > 0:
                    genotype_means = merged_clean.groupby(
                        snp)[trait].mean().sort_values(ascending=False)
                    if len(genotype_means) > 0:
                        effect_allele = genotype_means.index[0]  # Highest mean
                        effect_value = genotype_means.iloc[0]
                        valid_genotypes = merged_clean[snp].dropna().unique()
                        all_alleles = '/'.join(
                            sorted([str(gt) for gt in valid_genotypes if str(gt) != 'nan']))
                    else:
                        effect_allele = 'Unknown'
                        effect_value = np.nan
                        all_alleles = 'Unknown'
                else:
                    effect_allele = 'Unknown'
                    effect_value = np.nan
                    all_alleles = 'Unknown'

                # Get gene information
                gene = snp_to_gene.get(snp, 'Unknown')
                if gene == '':
                    gene = 'Unknown'

                results.append({
                    'SNP': snp,
                    'Gene': gene,
                    'Trait': trait,
                    'Effect_Allele': effect_allele,
                    'Effect_Value': effect_value,
                    'All_Alleles': all_alleles,
                    'F_stat': f_stat,
                    'p_value': p_val,
                    'R2': r2
                })

            current_progress += 1
            progress_bar.progress(current_progress / total_combinations)

    results_df = pd.DataFrame(results)

    # Apply Bonferroni correction
    if len(results_df) > 0:
        unique_snps = results_df['SNP'].nunique()
        results_df['bonferroni_corrected_p'] = results_df['p_value'] * unique_snps
        results_df['bonferroni_corrected_p'] = results_df['bonferroni_corrected_p'].clip(
            upper=1.0)
        results_df['bonferroni_significant'] = results_df['bonferroni_corrected_p'] < 0.05

        # Sort by R²
        results_df = results_df.sort_values(
            ['Trait', 'R2'], ascending=[True, False])

    progress_bar.empty()
    return results_df


def create_visualizations(results_df, selected_traits, top_bar_count=15, top_table_count=10):
    """Create interactive visualizations"""

    if len(results_df) == 0:
        st.warning("No results to display")
        return

    # Top effects bar chart
    st.subheader("🏆 Strongest Genetic Effects")

    # Filter significant results
    significant_results = results_df[results_df['bonferroni_significant'] == True]

    if len(significant_results) > 0:
        top_effects = significant_results.nlargest(top_bar_count, 'R2')

        fig = px.bar(top_effects,
                     x='R2',
                     y='SNP',
                     color='Trait',
                     title=f"Top {top_bar_count} SNP-Trait Associations (Bonferroni Significant)",
                     labels={'R2': 'R² (Variance Explained)',
                             'SNP': 'Genetic Variant'},
                     orientation='h')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Trait-specific results
    for trait in selected_traits:
        trait_data = results_df[results_df['Trait'] == trait]
        trait_significant = trait_data[trait_data['bonferroni_significant'] == True]

        if len(trait_significant) > 0:
            st.subheader(f"🧬 Results for {trait}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Significant SNPs", len(trait_significant))
                if len(trait_significant) > 0:
                    st.metric("Max R²", f"{trait_significant['R2'].max():.3f}")

            with col2:
                st.metric("Total SNPs Tested", len(trait_data))
                if len(trait_significant) > 0:
                    st.metric("Average R²",
                              f"{trait_significant['R2'].mean():.3f}")

            # Top results table
            st.write(f"**Top {top_table_count} Results:**")
            display_cols = ['SNP', 'Gene',
                            'Effect_Allele', 'Effect_Value', 'R2']
            display_df = trait_significant[display_cols].head(
                top_table_count).copy()
            display_df['Effect_Value'] = display_df['Effect_Value'].round(2)
            display_df['R2'] = display_df['R2'].round(3)
            st.dataframe(display_df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="GenEffect",
        page_icon="🧬",
        layout="wide"
    )

    st.title("GenEffect")
    st.markdown(
        "**See which genetic variants impact hair, skin, and eye color, ranked by statistical strength.**")

    # Load data
    with st.spinner("Loading datasets..."):
        traits_df, genotypes_df = load_data()
        snp_to_gene = load_snp_gene_mapping()

    # Sidebar configuration
    st.sidebar.header("📋 Analysis Configuration")

    # Hide population selection menu
    selected_groups = ['ALL_COMBINED']
    # ...existing code...

    # Trait selection
    trait_categories = get_trait_columns(traits_df)

    st.sidebar.subheader("Select Trait Category:")
    trait_category = st.sidebar.selectbox(
        "Category:",
        list(trait_categories.keys()),
        help="Choose a category of traits to analyze"
    )

    available_traits = trait_categories[trait_category]
    selected_traits = st.sidebar.multiselect(
        "Select Specific Traits:",
        available_traits,
        default=available_traits[:3] if len(
            available_traits) >= 3 else available_traits,
        help="Choose specific traits within the selected category"
    )

    if not selected_traits:
        st.sidebar.error("Please select at least one trait")
        return

    # Display configuration
    st.sidebar.subheader("📊 Display Options")
    show_raw_data = st.sidebar.checkbox("Show raw results table", value=False)
    min_r2_threshold = st.sidebar.slider(
        "Minimum R² threshold for display", 0.0, 1.0, 0.01, 0.01)
    top_bar_count = st.sidebar.slider(
        "Top results in bar chart", 5, 30, 15, 1)
    top_table_count = st.sidebar.slider(
        "Top results in trait table", 5, 30, 10, 1)
    dominance_mode = st.sidebar.checkbox(
        "Include dominance effects", value=False, help="Add dominance coding to regression analysis.")

    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", f"{len(traits_df):,} individuals")
    with col2:
        st.metric("Population Groups", len(traits_df['group'].unique()))
    with col3:
        st.metric("Genetic Variants", len([col for col in genotypes_df.columns if col not in [
                  'source', 'group', 'group_full', 'individual']]))

    # Run analysis button
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running genetic analysis..."):
            results = run_snp_analysis(
                traits_df, genotypes_df, selected_traits, selected_groups, dominance_mode)

        if len(results) == 0:
            st.error("No results generated. Please check your selections.")
            return

        # Filter by R² threshold
        filtered_results = results[results['R2'] >= min_r2_threshold]

        st.success(
            f"Analysis complete! Found {len(filtered_results)} associations above R² = {min_r2_threshold}")

        # Summary statistics
        st.subheader("📈 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_tests = len(results)
            st.metric("Total Tests", f"{total_tests:,}")

        with col2:
            raw_significant = len(results[results['p_value'] < 0.05])
            st.metric("Raw Significant (p<0.05)", raw_significant)

        with col3:
            bonf_significant = len(
                results[results['bonferroni_significant'] == True])
            st.metric("Bonferroni Significant", bonf_significant)

        with col4:
            if bonf_significant > 0:
                max_r2 = results[results['bonferroni_significant']
                                 == True]['R2'].max()
                st.metric("Max R²", f"{max_r2:.3f}")
            else:
                st.metric("Max R²", "0.000")

        # Visualizations
        create_visualizations(
            filtered_results, selected_traits, top_bar_count, top_table_count)

        # Raw data table
        if show_raw_data:
            st.subheader("📋 Complete Results Table")
            display_cols = ['SNP', 'Trait', 'Effect_Allele',
                            'Effect_Value', 'All_Alleles', 'R2', 'bonferroni_significant']
            display_df = filtered_results[display_cols].copy()
            display_df['Effect_Value'] = display_df['Effect_Value'].round(2)
            display_df['R2'] = display_df['R2'].round(3)
            st.dataframe(display_df, use_container_width=True)

        # Download option
        st.subheader("💾 Download Results")
        csv = filtered_results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"genetic_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
