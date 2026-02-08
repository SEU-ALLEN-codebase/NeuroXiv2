import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== Statistical Analyzer ====================

class StatisticalAnalyzer:
    """
    统计分析器（v3.0 - 公平且严谨）

    🔧 关键修复：
    - 正确处理None值
    - 选择合适的统计检验
    - FDR校正
    - 分层比较
    """

    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "detailed_results.json"

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'r') as f:
            self.raw_results = json.load(f)

        logger.info(f"✅ Loaded results from {self.results_file}")

        # 🔧 加载评估配置
        from evaluators import EVALUATION_CONFIG
        self.eval_config = EVALUATION_CONFIG

    def run_full_analysis(self) -> pd.DataFrame:
        """运行完整统计分析"""

        logger.info("\n" + "="*80)
        logger.info("📊 STATISTICAL ANALYSIS (v3.0 - Fair & Rigorous)")
        logger.info("="*80)

        # 提取指标数据
        metrics_data = self._extract_metrics_with_none_handling()

        # 运行对比检验
        comparisons = []

        aipom_scores = metrics_data.get('AIPOM-CoT', {})

        baseline_methods = ['Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        for method in baseline_methods:
            if method not in metrics_data:
                logger.warning(f"  Method '{method}' not found in results")
                continue

            method_scores = metrics_data[method]

            # 🔧 分层比较：核心指标 vs 系统指标
            comparable_metrics = self._get_comparable_metrics(
                aipom_scores,
                method_scores,
                method
            )

            logger.info(f"\n🔬 Comparing AIPOM-CoT vs {method}:")
            logger.info(f"  Comparable metrics: {comparable_metrics}")

            for metric_name in comparable_metrics:
                comparison = self.compare_methods_robust(
                    aipom_scores[metric_name],
                    method_scores[metric_name],
                    'AIPOM-CoT',
                    method,
                    metric_name
                )
                comparisons.append(comparison)

        # 转为DataFrame
        df = pd.DataFrame(comparisons)

        if len(df) == 0:
            logger.warning("⚠️  No valid comparisons generated")
            return df

        # 🔧 多重比较校正
        df = self._apply_fdr_correction(df)

        # 保存
        output_file = self.results_dir / "statistical_analysis.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"\n✅ Statistical analysis saved to: {output_file}")

        # 生成LaTeX表格
        self._generate_latex_table(df)

        # 打印摘要
        self._print_summary(df)

        return df

    def _extract_metrics_with_none_handling(self) -> Dict[str, Dict[str, List]]:
        """
        🔧 提取指标，正确处理None值

        Returns:
            {
                'method_name': {
                    'metric_name': [score1, score2, ...],  # None保留为None
                }
            }
        """

        metrics_data = defaultdict(lambda: defaultdict(list))

        for method, results_list in self.raw_results.items():
            for result in results_list:
                metrics = result.get('metrics', {})

                # 提取所有指标
                metric_names = [
                    'entity_f1',
                    'factual_accuracy',
                    'answer_completeness',
                    'scientific_rigor',
                    'depth_matching_accuracy',
                    'plan_coherence',
                    'modality_coverage',
                    'closed_loop_achieved',
                ]

                for metric_name in metric_names:
                    value = getattr(metrics, metric_name, None)

                    # 🔧 处理closed_loop_achieved (bool → float)
                    if metric_name == 'closed_loop_achieved' and value is not None:
                        value = 1.0 if value else 0.0

                    # 保留None（不转为0）
                    metrics_data[method][metric_name].append(value)

        return dict(metrics_data)

    def _get_comparable_metrics(self,
                                scores_a: Dict[str, List],
                                scores_b: Dict[str, List],
                                method_b_name: str) -> List[str]:
        """获取可比较的指标（v3.1）"""

        # 1. 核心指标：所有方法都比较
        core_metrics = list(self.eval_config['core_metrics'].keys())
        comparable = set(core_metrics)

        # 2. 系统指标：检查方法是否适用
        for metric_name, config in self.eval_config['system_metrics'].items():
            applicable_methods = config['methods']

            # 🔧 reasoning_depth对所有方法都适用
            if metric_name == 'reasoning_depth' and applicable_methods == 'all':
                comparable.add(metric_name)

            # 其他系统指标按原逻辑
            elif 'AIPOM-CoT' in applicable_methods and \
                    (method_b_name in applicable_methods or applicable_methods == 'all'):
                comparable.add(metric_name)

        # 3. 检查数据是否存在
        valid_comparable = []
        for metric_name in comparable:
            actual_field = self._map_metric_name(metric_name)

            scores_a_list = scores_a.get(actual_field, [])
            scores_b_list = scores_b.get(actual_field, [])

            valid_a = [s for s in scores_a_list if s is not None]
            valid_b = [s for s in scores_b_list if s is not None]

            if len(valid_a) >= 3 and len(valid_b) >= 3:
                valid_comparable.append(actual_field)

        return valid_comparable

    def _map_metric_name(self, metric_name: str) -> str:
        """映射metric名称（v3.1）"""
        mapping = {
            'entity_f1': 'entity_f1',
            'factual_accuracy': 'factual_accuracy',
            'answer_completeness': 'answer_completeness',
            'scientific_rigor': 'scientific_rigor',
            'reasoning_depth': 'reasoning_depth',  # 🔧 改名
            'plan_coherence': 'plan_coherence',
            'modality_coverage': 'modality_coverage',
            'closed_loop': 'closed_loop_achieved',
        }
        return mapping.get(metric_name, metric_name)

    def compare_methods_robust(self,
                               scores_a: List[Optional[float]],
                               scores_b: List[Optional[float]],
                               method_a: str,
                               method_b: str,
                               metric_name: str) -> Dict:
        """
        🔧 稳健的方法对比

        处理：
        - None值
        - 配对vs独立检验
        - 小样本
        """

        # 🔧 过滤None值，创建配对数据
        paired_scores = []
        for a, b in zip(scores_a, scores_b):
            if a is not None and b is not None:
                paired_scores.append((a, b))

        n_pairs = len(paired_scores)

        if n_pairs < 3:
            # 样本太少
            return {
                'metric': metric_name,
                'method_a': method_a,
                'method_b': method_b,
                'n': n_pairs,
                'mean_a': np.nan,
                'mean_b': np.nan,
                'p_value': np.nan,
                'p_value_raw': np.nan,
                'significance': 'insufficient_data',
                'cohens_d': np.nan,
                'test_type': 'none',
            }

        scores_a_clean, scores_b_clean = zip(*paired_scores)
        scores_a_clean = np.array(scores_a_clean)
        scores_b_clean = np.array(scores_b_clean)

        # 基本统计
        mean_a = np.mean(scores_a_clean)
        mean_b = np.mean(scores_b_clean)
        std_a = np.std(scores_a_clean, ddof=1)
        std_b = np.std(scores_b_clean, ddof=1)

        # 🔧 选择统计检验
        # 如果是同样的问题（配对），用paired t-test
        # 否则用independent t-test

        try:
            # 尝试paired t-test（假设配对）
            if len(scores_a_clean) == len(scores_b_clean):
                t_stat, p_value = stats.ttest_rel(scores_a_clean, scores_b_clean)
                test_type = 'paired_t'
            else:
                # Fallback: independent
                t_stat, p_value = stats.ttest_ind(scores_a_clean, scores_b_clean)
                test_type = 'independent_t'

        except Exception as e:
            logger.warning(f"  t-test failed for {metric_name}: {e}")
            # Fallback: Mann-Whitney U test (非参数)
            try:
                u_stat, p_value = stats.mannwhitneyu(scores_a_clean, scores_b_clean, alternative='two-sided')
                test_type = 'mann_whitney'
                t_stat = u_stat
            except:
                return {
                    'metric': metric_name,
                    'method_a': method_a,
                    'method_b': method_b,
                    'n': n_pairs,
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'p_value': np.nan,
                    'p_value_raw': np.nan,
                    'significance': 'test_failed',
                    'cohens_d': np.nan,
                    'test_type': 'failed',
                }

        # Cohen's d
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # 95% CI for difference
        diff = scores_a_clean - scores_b_clean
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_95_lower = np.mean(diff) - 1.96 * se
        ci_95_upper = np.mean(diff) + 1.96 * se

        # Significance (before FDR correction)
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'

        # Effect size label
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_size_label = 'negligible'
        elif abs_d < 0.5:
            effect_size_label = 'small'
        elif abs_d < 0.8:
            effect_size_label = 'medium'
        else:
            effect_size_label = 'large'

        return {
            'metric': metric_name,
            'method_a': method_a,
            'method_b': method_b,
            'n': n_pairs,
            'mean_a': mean_a,
            'std_a': std_a,
            'mean_b': mean_b,
            'std_b': std_b,
            'mean_diff': mean_a - mean_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'p_value_raw': p_value,
            'significance': significance,
            'cohens_d': cohens_d,
            'effect_size': effect_size_label,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'test_type': test_type,
        }

    def _apply_fdr_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 应用FDR多重比较校正（Benjamini-Hochberg）
        """

        from statsmodels.stats.multitest import multipletests

        # 提取所有p值
        p_values = df['p_value_raw'].fillna(1.0).values

        if len(p_values) == 0:
            return df

        # FDR校正
        try:
            reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values,
                alpha=0.05,
                method='fdr_bh'
            )

            # 添加校正后的列
            df['p_value_fdr'] = p_corrected
            df['significant_fdr'] = reject

            # 更新significance标记
            df['significance_fdr'] = df['p_value_fdr'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            )

            logger.info(f"\n🔧 FDR Correction Applied:")
            logger.info(f"  Total comparisons: {len(df)}")
            logger.info(f"  Significant before FDR: {sum(df['p_value_raw'] < 0.05)}")
            logger.info(f"  Significant after FDR: {sum(reject)}")

        except Exception as e:
            logger.error(f"  FDR correction failed: {e}")
            df['p_value_fdr'] = df['p_value_raw']
            df['significant_fdr'] = df['p_value_raw'] < 0.05
            df['significance_fdr'] = df['significance']

        return df

    def _print_summary(self, df: pd.DataFrame):
        """打印统计摘要"""

        print("\n" + "="*80)
        print("📊 KEY FINDINGS (with FDR correction)")
        print("="*80)

        # 按方法分组
        for method_b in df['method_b'].unique():
            method_df = df[df['method_b'] == method_b]

            print(f"\n{'AIPOM-CoT vs ' + method_b}")
            print("-" * 60)

            # 按指标分类
            core_metrics = ['entity_f1', 'factual_accuracy', 'answer_completeness', 'scientific_rigor']
            system_metrics = ['depth_matching_accuracy', 'plan_coherence', 'modality_coverage', 'closed_loop_achieved']

            print("  Core Metrics:")
            for _, row in method_df[method_df['metric'].isin(core_metrics)].iterrows():
                self._print_comparison_row(row, indent="    ")

            print("\n  System Metrics:")
            for _, row in method_df[method_df['metric'].isin(system_metrics)].iterrows():
                self._print_comparison_row(row, indent="    ")

        print("\n" + "="*80)
        print("Legend:")
        print("  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05 (FDR-corrected)")
        print("  Effect size: Cohen's d (negligible:<0.2, small:<0.5, medium:<0.8, large:≥0.8)")
        print("="*80)

    def _print_comparison_row(self, row: pd.Series, indent: str = ""):
        """打印单个比较结果"""

        metric = row['metric'].replace('_', ' ').title()
        mean_a = row['mean_a']
        mean_b = row['mean_b']
        diff = row['mean_diff']
        p_fdr = row['p_value_fdr']
        sig = row['significance_fdr']
        cohens_d = row['cohens_d']
        effect = row['effect_size']

        # 计算提升百分比
        if mean_b > 0:
            improvement = (diff / mean_b) * 100
        else:
            improvement = 0

        print(f"{indent}{metric:25s}: {mean_a:.3f} vs {mean_b:.3f} "
              f"(Δ={diff:+.3f}, {improvement:+.1f}%)")
        print(f"{indent}{'':25s}  p_FDR={p_fdr:.4f}{sig}, d={cohens_d:.2f} ({effect})")

    def _generate_latex_table(self, df: pd.DataFrame):
        """生成LaTeX表格"""

        output_file = self.results_dir / "statistical_analysis.tex"

        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Statistical Comparison of AIPOM-CoT vs Baselines (FDR-corrected)}")
        latex_lines.append(r"\label{tab:statistical_analysis}")
        latex_lines.append(r"\begin{tabular}{llccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Metric & Baseline & AIPOM-CoT & Baseline & $p_{FDR}$ & Cohen's $d$ & Effect Size \\")
        latex_lines.append(r"\midrule")

        # 按metric和method分组
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric]

            metric_name = metric.replace('_', ' ').title()

            for idx, row in metric_df.iterrows():
                method_b = row['method_b'].replace('GPT-4o', r'GPT-4\textit{o}')
                mean_a = row['mean_a']
                mean_b = row['mean_b']
                p_fdr = row['p_value_fdr']
                sig = row['significance_fdr'].replace('***', r'$^{***}$').replace('**', r'$^{**}$').replace('*', r'$^{*}$')
                cohens_d = row['cohens_d']
                effect = row['effect_size']

                if idx == metric_df.index[0]:
                    latex_lines.append(f"{metric_name} & {method_b} & {mean_a:.3f} & {mean_b:.3f} & {p_fdr:.4f}{sig} & {cohens_d:.2f} & {effect} \\\\")
                else:
                    latex_lines.append(f"& {method_b} & {mean_a:.3f} & {mean_b:.3f} & {p_fdr:.4f}{sig} & {cohens_d:.2f} & {effect} \\\\")

            latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\begin{tablenotes}")
        latex_lines.append(r"\small")
        latex_lines.append(r"\item $^{***}$p<0.001, $^{**}$p<0.01, $^{*}$p<0.05 (FDR-corrected)")
        latex_lines.append(r"\item Effect size: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)")
        latex_lines.append(r"\end{tablenotes}")
        latex_lines.append(r"\end{table}")

        with open(output_file, 'w') as f:
            f.write('\n'.join(latex_lines))

        logger.info(f"  LaTeX table saved to: {output_file}")

    def generate_performance_summary_table(self) -> pd.DataFrame:
        """
        生成性能摘要表（用于论文）

        Returns:
            DataFrame with mean ± std for each method and metric
        """

        metrics_data = self._extract_metrics_with_none_handling()

        summary_rows = []

        metric_names = [
            'entity_f1',
            'factual_accuracy',
            'answer_completeness',
            'scientific_rigor',
            'reasoning_depth',
            'modality_coverage',
        ]

        for metric_name in metric_names:
            row = {'Metric': metric_name.replace('_', ' ').title()}

            for method in ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']:
                if method not in metrics_data:
                    row[method] = 'N/A'
                    continue

                scores = metrics_data[method].get(metric_name, [])
                valid_scores = [s for s in scores if s is not None]

                if len(valid_scores) == 0:
                    row[method] = 'N/A'
                else:
                    mean = np.mean(valid_scores)
                    std = np.std(valid_scores, ddof=1)
                    row[method] = f"{mean:.3f} ± {std:.3f}"

            summary_rows.append(row)

        df = pd.DataFrame(summary_rows)

        # 保存
        output_file = self.results_dir / "performance_summary.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"\n📊 Performance summary saved to: {output_file}")

        return df

    def analyze_by_complexity(self) -> pd.DataFrame:
        """
        🔧 按复杂度等级分析性能

        Returns:
            DataFrame with performance by complexity level
        """

        complexity_results = defaultdict(lambda: defaultdict(list))

        for method, results_list in self.raw_results.items():
            for result in results_list:
                # 获取复杂度等级
                question_data = result.get('question_data', {})
                tier = question_data.get('tier', 'unknown')

                # 获取overall score
                metrics = result.get('metrics', {})
                overall = getattr(metrics, 'overall_score', None)

                if overall is not None:
                    complexity_results[method][tier].append(overall)

        # 转为DataFrame
        summary_rows = []

        for method in ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']:
            if method not in complexity_results:
                continue

            row = {'Method': method}

            for tier in ['simple', 'medium', 'deep', 'screening']:
                scores = complexity_results[method].get(tier, [])

                if len(scores) == 0:
                    row[tier.capitalize()] = 'N/A'
                else:
                    mean = np.mean(scores)
                    std = np.std(scores, ddof=1)
                    row[tier.capitalize()] = f"{mean:.3f} ± {std:.3f}"

            summary_rows.append(row)

        df = pd.DataFrame(summary_rows)

        # 保存
        output_file = self.results_dir / "performance_by_complexity.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"\n📊 Complexity analysis saved to: {output_file}")

        return df


# ==================== Convenience Functions ====================

def run_statistical_analysis(results_dir: str = "./benchmark_results") -> pd.DataFrame:
    """
    便捷函数：运行完整统计分析

    Usage:
        from statistical_analysis import run_statistical_analysis
        df = run_statistical_analysis("./benchmark_results")
    """

    analyzer = StatisticalAnalyzer(results_dir)

    # 主要统计分析
    df = analyzer.run_full_analysis()

    # 生成摘要表
    summary_df = analyzer.generate_performance_summary_table()

    # 复杂度分析
    complexity_df = analyzer.analyze_by_complexity()

    return df


# ==================== Test ====================

if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("📊 Statistical Analysis Tool v3.0 (Fair & Rigorous)")
    print("="*80)

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "./benchmark_results"

    print(f"\nAnalyzing results from: {results_dir}")

    try:
        analyzer = StatisticalAnalyzer(results_dir)

        print("\n🔬 Running statistical analysis...")
        df = analyzer.run_full_analysis()

        print("\n📊 Generating performance summary...")
        summary_df = analyzer.generate_performance_summary_table()

        print("\n📈 Analyzing by complexity...")
        complexity_df = analyzer.analyze_by_complexity()

        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("="*80)
        print("\nGenerated files:")
        print(f"  - {results_dir}/statistical_analysis.csv")
        print(f"  - {results_dir}/statistical_analysis.tex")
        print(f"  - {results_dir}/performance_summary.csv")
        print(f"  - {results_dir}/performance_by_complexity.csv")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)