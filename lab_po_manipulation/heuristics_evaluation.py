import pandas as pd

from lab_po_manipulation.generate_prms import load_prm
from lab_po_manipulation.pomdp_model.pomanipulation_problem import POMProblem
from rocksample_experiments.heuristics_evaluation import evaluate_rank_of_best_heuristic, partial_ordering_agreement, \
    evaluate_top_k_accuracy, evaluate_spearman_correlation, evaluate_ci_weighted_correlation


def test_heuristic_on_problem(voa_df, heuristic_function, heuristic_kwargs=None):
    """
    Test a heuristic function on one problem instance across all helping actions.

    Args:
        voa_df: DataFrame with empirical VOA values
        problem: POMProblem instance
        heuristic_function: Function to evaluate the heuristic
        heuristic_kwargs: Optional kwargs for the heuristic function

    Returns:
        DataFrame with heuristic computation results for each help action
    """
    if heuristic_kwargs is None:
        heuristic_kwargs = {}

    prm = load_prm(obstacles_to_exclude=[], constrained=False)
    prm_y_up = load_prm(obstacles_to_exclude=[], constrained=True)
    problem = POMProblem(prm, prm_y_up)

    results = []

    for _, row in voa_df.iterrows():
        help_id = int(row['help_id'])
        obstacles_to_exclude = [help_id + 1]

        # Compute heuristic value and time
        heuristic_value, computation_time = heuristic_function(
            problem,
            obstacles_to_exclude,
            **heuristic_kwargs
        )

        results.append({
            'help_id': help_id,
            'heuristic_value': heuristic_value,
            'computation_time': computation_time,
            'empirical_voa': row['empirical_voa'],
            'ci_low_95': row.get('ci_95_low', None),
            'ci_high_95': row.get('ci_95_high', None),
            'ci_low_90': row.get('ci_90_low', None),
            'ci_high_90': row.get('ci_90_high', None),
            'ci_low_80': row.get('ci_80_low', None),
            'ci_high_80': row.get('ci_80_high', None)
        })

    return pd.DataFrame(results)


def heuristic_metrics(results_df):
    """
    Comprehensive evaluation of heuristic performance with flat metrics structure,
    adapted for manipulation domain
    """
    rank_of_best, voa_of_best = evaluate_rank_of_best_heuristic(results_df)
    partial_ordering = partial_ordering_agreement(results_df)

    evaluation = {
        'top_1_accuracy': evaluate_top_k_accuracy(results_df, k=1),
        'top_2_accuracy': evaluate_top_k_accuracy(results_df, k=2),
        'spearman_correlation': evaluate_spearman_correlation(results_df)[0],
        'spearman_correlation_pvalue': evaluate_spearman_correlation(results_df)[1],
        'ci_weighted_correlation': evaluate_ci_weighted_correlation(results_df),
        'rank_of_best_heuristic': rank_of_best,
        'voa_of_best_heuristic': voa_of_best,
        # Partial ordering agreement metrics
        'partial_ordering_agreement': partial_ordering['agreement'],
        'partial_ordering_strict_pairs_ratio': partial_ordering['strict_pairs_ratio'],
        'partial_ordering_n_strict_pairs': partial_ordering['n_strict_pairs'],
        'partial_ordering_n_total_pairs': partial_ordering['n_total_pairs'],
        # Computation time metrics
        'mean_computation_time': results_df['computation_time'].mean(),
        'std_computation_time': results_df['computation_time'].std(),
        'max_computation_time': results_df['computation_time'].max(),
        'min_computation_time': results_df['computation_time'].min(),
    }

    return evaluation