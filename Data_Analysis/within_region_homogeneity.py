import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 统计
from scipy.stats import wilcoxon

# 可视化
import matplotlib.pyplot as plt

# Neo4j
import neo4j

# 设置绘图
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = '#333333'


@dataclass
class RegionDispersion:
    """区域离散度"""
    region: str
    n_neurons: int
    morph_disp: float
    gene_disp: float
    proj_disp: float
    multi_disp: float


@dataclass
class GlobalStats:
    """全局统计"""
    modality: str
    n_dims: int
    global_dispersion: float  # trace(Cov)/d
    raw_trace_cov: float      # trace(Cov) 未归一化


@dataclass
class VectorInfo:
    """向量信息"""
    name: str
    original_dims: int
    pca_dims: int
    variance_explained: float


class WithinRegionHomogeneityExperiment:
    """区域内同质性实验 V3"""

    AXONAL_FEATURES = [
        'axonal_total_length', 'axonal_volume', 'axonal_area',
        'axonal_number_of_bifurcations', 'axonal_max_branch_order',
        'axonal_max_euclidean_distance', 'axonal_max_path_distance',
        'axonal_average_euclidean_distance', 'axonal_average_path_distance',
        'axonal_75pct_euclidean_distance', 'axonal_75pct_path_distance',
        'axonal_50pct_euclidean_distance', 'axonal_50pct_path_distance',
        'axonal_25pct_euclidean_distance', 'axonal_25pct_path_distance',
        'axonal_average_bifurcation_angle_local', 'axonal_average_bifurcation_angle_remote',
        'axonal_average_contraction',
        'axonal_width', 'axonal_height', 'axonal_depth',
        'axonal_width_95ci', 'axonal_height_95ci', 'axonal_depth_95ci',
        'axonal_flatness', 'axonal_flatness_95ci',
        'axonal_slimness', 'axonal_slimness_95ci',
        'axonal_center_shift', 'axonal_relative_center_shift',
        'axonal_2d_density', 'axonal_3d_density'
    ]

    DENDRITIC_FEATURES = [
        'dendritic_total_length', 'dendritic_volume', 'dendritic_area',
        'dendritic_number_of_bifurcations', 'dendritic_max_branch_order',
        'dendritic_max_euclidean_distance', 'dendritic_max_path_distance',
        'dendritic_average_euclidean_distance', 'dendritic_average_path_distance',
        'dendritic_75pct_euclidean_distance', 'dendritic_75pct_path_distance',
        'dendritic_50pct_euclidean_distance', 'dendritic_50pct_path_distance',
        'dendritic_25pct_euclidean_distance', 'dendritic_25pct_path_distance',
        'dendritic_average_bifurcation_angle_local', 'dendritic_average_bifurcation_angle_remote',
        'dendritic_average_contraction',
        'dendritic_width', 'dendritic_height', 'dendritic_depth',
        'dendritic_width_95ci', 'dendritic_height_95ci', 'dendritic_depth_95ci',
        'dendritic_flatness', 'dendritic_flatness_95ci',
        'dendritic_slimness', 'dendritic_slimness_95ci',
        'dendritic_center_shift', 'dendritic_relative_center_shift',
        'dendritic_2d_density', 'dendritic_3d_density'
    ]

    def __init__(self, uri: str, user: str, password: str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 8.0,
                 pca_variance_threshold: float = 0.95,
                 min_neurons_per_region: int = 10,
                 subsample_n: int = 50,
                 subsample_repeats: int = 50):
        """
        参数:
            pca_variance_threshold: PCA方差解释阈值（95%）
            min_neurons_per_region: 区域内最少神经元数
            subsample_n: 区域内subsampling数量
            subsample_repeats: subsampling重复次数
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.min_neurons_per_region = min_neurons_per_region
        self.subsample_n = subsample_n
        self.subsample_repeats = subsample_repeats

        # 数据
        self.valid_neuron_ids: List[str] = []
        self.neuron_regions: Dict[str, str] = {}

        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 处理后的向量
        self.morph_vectors: np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None
        self.multi_vectors: np.ndarray = None

        # 向量信息
        self.vector_info: Dict[str, VectorInfo] = {}

        # 结果
        self.region_dispersions: List[RegionDispersion] = []
        self.valid_regions: List[str] = []
        self.global_stats: Dict[str, GlobalStats] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 数据加载 ====================

    def load_all_data(self) -> int:
        """加载数据"""
        print("\n" + "=" * 80)
        print("加载数据")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._filter_valid_neurons()
        self._process_all_vectors()
        self._compute_global_stats()
        self._filter_regions()

        print(f"\n✓ 数据加载完成:")
        print(f"  神经元数: {len(self.valid_neuron_ids)}")
        print(f"  有效区域数: {len(self.valid_regions)} (>={self.min_neurons_per_region}神经元)")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.local_gene_features_raw = cache_data['local_environments']
        self.all_subclasses = cache_data['all_subclasses']
        print(f"  加载了 {len(self.local_gene_features_raw)} 个神经元的分子环境")

    def _get_global_dimensions(self):
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]
        print(f"  投射目标: {len(self.all_target_regions)} 个脑区")

    def _load_all_neuron_features(self):
        """加载神经元特征和所属区域"""
        axon_return = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n:Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
          AND n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
        OPTIONAL MATCH (n)-[:LOCATE_AT_SUBREGION]->(s:Subregion)
        OPTIONAL MATCH (n)-[:LOCATE_AT]->(r:Region)
        RETURN n.neuron_id AS neuron_id, 
               s.acronym AS subregion,
               r.acronym AS region,
               n.base_region AS base_region,
               n.celltype AS celltype,
               {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        proj_query = """
        MATCH (n:Neuron {neuron_id: $neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            records = list(result)

            for record in records:
                neuron_id = record['neuron_id']

                region = (record['subregion'] or
                         record['region'] or
                         record['base_region'] or
                         record['celltype'])

                if region:
                    self.neuron_regions[neuron_id] = region

                axon_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.AXONAL_FEATURES]
                self.axon_features_raw[neuron_id] = np.array(axon_feats)

                dend_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.DENDRITIC_FEATURES]
                self.dendrite_features_raw[neuron_id] = np.array(dend_feats)

                proj_result = session.run(proj_query, neuron_id=neuron_id)
                proj_dict = {r['target']: r['weight'] for r in proj_result
                             if r['target'] and r['weight']}
                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]
                    self.projection_vectors_raw[neuron_id] = proj_vector

        print(f"  加载了 {len(self.axon_features_raw)} 个神经元")
        print(f"  有区域信息的神经元: {len(self.neuron_regions)}")

    def _filter_valid_neurons(self):
        """过滤有效神经元"""
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        candidates &= set(self.neuron_regions.keys())

        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  有效神经元: {len(self.valid_neuron_ids)}")

    def _process_all_vectors(self):
        """
        处理向量 - 标准方法：
        - Morph: Z-score → PCA 95%
        - Gene: 剪枝 → Z-score → PCA 95%
        - Proj: 剪枝 → log1p → Z-score → PCA 95%
        - Multi: concat([Morph_PCA, Gene_PCA, Proj_PCA])
        """
        print(f"\n处理向量 (PCA方差阈值={self.pca_variance_threshold:.0%})...")
        neurons = self.valid_neuron_ids

        # ===== 形态向量: Z-score → PCA 95% =====
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_info = self._process_morph(morph_raw)
        self.vector_info['Morph'] = morph_info

        # ===== 分子向量: 剪枝 → Z-score → PCA 95% =====
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        self.gene_vectors, gene_info = self._process_gene(gene_raw)
        self.vector_info['Gene'] = gene_info

        # ===== 投射向量: 剪枝 → log1p → Z-score → PCA 95% =====
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        self.proj_vectors, proj_info = self._process_proj(proj_raw)
        self.vector_info['Proj'] = proj_info

        # ===== Multi向量: concat =====
        self.multi_vectors = np.hstack([self.morph_vectors, self.gene_vectors, self.proj_vectors])
        multi_dims = self.multi_vectors.shape[1]
        self.vector_info['Multi'] = VectorInfo(
            'Multi',
            morph_info.original_dims + gene_info.original_dims + proj_info.original_dims,
            multi_dims,
            1.0  # concat后方差解释无意义
        )
        print(f"  Multi: concat({morph_info.pca_dims}+{gene_info.pca_dims}+{proj_info.pca_dims}) = {multi_dims}D")

    def _process_morph(self, X_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """形态向量: Z-score → PCA 95%"""
        original_dims = X_raw.shape[1]
        print(f"  Morph: {original_dims}D", end="")

        # Z-score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # PCA 95%
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f" → Z-score → PCA → {X_pca.shape[1]}D (var={variance:.1%})")

        return X_pca, VectorInfo('Morph', original_dims, X_pca.shape[1], variance)

    def _process_gene(self, X_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """分子向量: 剪枝 → Z-score → PCA 95%"""
        original_dims = X_raw.shape[1]
        print(f"  Gene: {original_dims}D", end="")

        # 剪枝（去掉全零列）
        col_sums = X_raw.sum(axis=0)
        n_pruned = (col_sums == 0).sum()
        X_pruned = X_raw[:, col_sums > 0]
        print(f" → prune(-{n_pruned})={X_pruned.shape[1]}D", end="")

        # Z-score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pruned)

        # PCA 95%
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f" → Z-score → PCA → {X_pca.shape[1]}D (var={variance:.1%})")

        return X_pca, VectorInfo('Gene', original_dims, X_pca.shape[1], variance)

    def _process_proj(self, X_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """投射向量: 剪枝 → log1p → Z-score → PCA 95%"""
        original_dims = X_raw.shape[1]
        print(f"  Proj: {original_dims}D", end="")

        # 剪枝
        col_sums = X_raw.sum(axis=0)
        n_pruned = (col_sums == 0).sum()
        X_pruned = X_raw[:, col_sums > 0]
        print(f" → prune(-{n_pruned})={X_pruned.shape[1]}D", end="")

        # Log1p变换
        X_log = np.log1p(X_pruned)
        print(f" → log1p", end="")

        # Z-score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)

        # PCA 95%
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f" → Z-score → PCA → {X_pca.shape[1]}D (var={variance:.1%})")

        return X_pca, VectorInfo('Proj', original_dims, X_pca.shape[1], variance)

    def _compute_global_stats(self):
        """计算全局统计"""
        print("\n计算全局统计...")

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
            ('Multi', self.multi_vectors),
        ]

        for name, vectors in modalities:
            d = vectors.shape[1]

            # trace(Cov)
            cov = np.cov(vectors.T)
            raw_trace = np.trace(cov)

            # 维度归一化离散度: trace(Cov) / d
            disp_normalized = raw_trace / d

            self.global_stats[name] = GlobalStats(
                modality=name,
                n_dims=d,
                global_dispersion=disp_normalized,
                raw_trace_cov=raw_trace
            )

            print(f"  {name}: {d}D, trace(Cov)/d = {disp_normalized:.4f}")

    def _filter_regions(self):
        """过滤有效区域"""
        region_counts = defaultdict(int)
        for nid in self.valid_neuron_ids:
            region = self.neuron_regions[nid]
            region_counts[region] += 1

        self.valid_regions = [r for r, c in region_counts.items()
                              if c >= self.min_neurons_per_region]

        print(f"  区域数: {len(region_counts)}, 有效区域: {len(self.valid_regions)}")

    # ==================== 离散度计算 ====================

    def _compute_dispersion_normalized(self, X: np.ndarray) -> float:
        """
        计算维度归一化离散度: E[||x - x̄||² / d] = trace(Cov) / d

        这个指标：
        - 不随维度增大而增大
        - 不同维度的模态可以直接比较
        - 等价于"单位方差能量"
        """
        if len(X) < 2:
            return 0.0

        d = X.shape[1]

        # 方法1: trace(Cov) / d
        cov = np.cov(X.T)
        if cov.ndim == 0:  # 单维度情况
            return float(cov)
        return np.trace(cov) / d

        # 方法2 (等价): E[||x - x̄||² / d]
        # centroid = X.mean(axis=0)
        # mse = ((X - centroid) ** 2).sum(axis=1).mean()
        # return mse / d

    def _compute_dispersion_with_subsample(self, X: np.ndarray, n: int, repeats: int) -> Tuple[float, float]:
        """带subsampling的离散度计算"""
        if len(X) <= n:
            return self._compute_dispersion_normalized(X), 0.0

        dispersions = []
        for _ in range(repeats):
            idx = np.random.choice(len(X), n, replace=False)
            disp = self._compute_dispersion_normalized(X[idx])
            dispersions.append(disp)

        return np.mean(dispersions), np.std(dispersions)

    def compute_region_dispersion(self, use_subsample: bool = True):
        """计算每个区域的离散度"""
        print("\n" + "=" * 80)
        print("计算区域内离散度（维度归一化: trace(Cov)/d）")
        if use_subsample:
            print(f"  Subsampling: n={self.subsample_n}, repeats={self.subsample_repeats}")
        print("=" * 80)

        id_to_idx = {nid: i for i, nid in enumerate(self.valid_neuron_ids)}

        for region in self.valid_regions:
            region_neuron_ids = [nid for nid in self.valid_neuron_ids
                                  if self.neuron_regions[nid] == region]
            region_indices = [id_to_idx[nid] for nid in region_neuron_ids]
            n = len(region_indices)

            if n < self.min_neurons_per_region:
                continue

            # 提取各模态向量
            morph_region = self.morph_vectors[region_indices]
            gene_region = self.gene_vectors[region_indices]
            proj_region = self.proj_vectors[region_indices]
            multi_region = self.multi_vectors[region_indices]

            if use_subsample and n > self.subsample_n:
                morph_disp, _ = self._compute_dispersion_with_subsample(
                    morph_region, self.subsample_n, self.subsample_repeats)
                gene_disp, _ = self._compute_dispersion_with_subsample(
                    gene_region, self.subsample_n, self.subsample_repeats)
                proj_disp, _ = self._compute_dispersion_with_subsample(
                    proj_region, self.subsample_n, self.subsample_repeats)
                multi_disp, _ = self._compute_dispersion_with_subsample(
                    multi_region, self.subsample_n, self.subsample_repeats)
            else:
                morph_disp = self._compute_dispersion_normalized(morph_region)
                gene_disp = self._compute_dispersion_normalized(gene_region)
                proj_disp = self._compute_dispersion_normalized(proj_region)
                multi_disp = self._compute_dispersion_normalized(multi_region)

            self.region_dispersions.append(RegionDispersion(
                region=region,
                n_neurons=n,
                morph_disp=morph_disp,
                gene_disp=gene_disp,
                proj_disp=proj_disp,
                multi_disp=multi_disp
            ))

        print(f"  计算了 {len(self.region_dispersions)} 个区域的离散度")

    # ==================== 统计分析 ====================

    def statistical_analysis(self) -> Dict:
        """统计分析"""
        print("\n" + "=" * 80)
        print("统计分析")
        print("=" * 80)

        results = {}

        comparisons = [
            ('Multi vs Morph', 'morph_disp'),
            ('Multi vs Gene', 'gene_disp'),
            ('Multi vs Proj', 'proj_disp'),
        ]

        for name, single_attr in comparisons:
            multi_vals = np.array([r.multi_disp for r in self.region_dispersions])
            single_vals = np.array([getattr(r, single_attr) for r in self.region_dispersions])

            delta = multi_vals - single_vals
            improvement_pct = (delta < 0).mean() * 100

            # Wilcoxon signed-rank test
            stat, pval_two = wilcoxon(multi_vals, single_vals)
            _, pval_less = wilcoxon(multi_vals, single_vals, alternative='less')

            results[name] = {
                'delta': delta,
                'improvement_pct': improvement_pct,
                'wilcoxon_stat': stat,
                'p_value_two_sided': pval_two,
                'p_value_less': pval_less,
                'mean_delta': delta.mean(),
                'median_delta': np.median(delta),
            }

            sig = '***' if pval_less < 0.001 else '**' if pval_less < 0.01 else '*' if pval_less < 0.05 else 'n.s.'
            print(f"\n{name}:")
            print(f"  Improvement: {improvement_pct:.1f}% (Δ<0)")
            print(f"  Mean Δ: {delta.mean():.4f}, Median Δ: {np.median(delta):.4f}")
            print(f"  Wilcoxon (less) p-value: {pval_less:.2e} {sig}")

        return results

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("生成可视化")
        print("=" * 80)

        stats = self.statistical_analysis()

        self._plot_main_histograms(output_dir, stats)
        self._plot_summary_bar(output_dir, stats)
        self._plot_scatter_comparison(output_dir)
        self._plot_dimension_check(output_dir)

        print(f"\n✓ 图表已保存到: {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=1200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_main_histograms(self, output_dir: str, stats: Dict):
        """主图：三个直方图（类似参考图风格）"""
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        comparisons = [
            ('Multi vs Morph', 'morph_disp', 'Morphology'),
            ('Multi vs Gene', 'gene_disp', 'Molecule'),
            ('Multi vs Proj', 'proj_disp', 'Projection'),
        ]

        for ax, (name, attr, label) in zip(axes, comparisons):
            delta = stats[name]['delta']
            improvement = stats[name]['improvement_pct']
            pval = stats[name]['p_value_less']

            bins = np.linspace(delta.min() - 0.02, delta.max() + 0.02, 40)
            ax.hist(delta, bins=bins, color='#808080', edgecolor='#606060', alpha=0.9)

            ax.axvline(x=0, color='red', linewidth=2, zorder=10)

            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''

            ax.text(0.02, 0.95, f'Percentage of\nimprovement:\n{improvement:.1f}%',
                   transform=ax.transAxes, fontsize=11, color='red',
                   verticalalignment='top', fontweight='bold')

            ax.text(0.98, 0.95, label, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right',
                   fontweight='bold')

            ax.set_xlabel('Δ Dispersion (Multi - Single)\n[trace(Cov)/d, dimension-normalized]', fontsize=10)
            ax.set_ylabel('Number of CCF regions', fontsize=11)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Spatial homogeneity change\nwith multimodal representation',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_main_histograms.png")

    def _plot_summary_bar(self, output_dir: str, stats: Dict):
        """汇总条形图"""
        fig, ax = plt.subplots(figsize=(8, 5))

        names = ['vs Morph', 'vs Gene', 'vs Proj']
        improvements = [stats[f'Multi {n}']['improvement_pct'] for n in names]
        pvals = [stats[f'Multi {n}']['p_value_less'] for n in names]

        colors = ['#3498DB', '#27AE60', '#E74C3C']
        bars = ax.bar(range(len(names)), improvements, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        for i, (bar, imp, pval) in enumerate(zip(bars, improvements, pvals)):
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
            ax.annotate(f'{imp:.1f}%\n({sig})',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.set_ylabel('Improvement %\n(regions with Δ < 0)', fontsize=11)
        ax.set_title('Multimodal vs Single-modal\nWithin-region Homogeneity (dimension-normalized)',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f'Multi {n}' for n in names], fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_summary_bar.png")

    def _plot_scatter_comparison(self, output_dir: str):
        """散点图对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        comparisons = [
            ('morph_disp', 'Morphology', '#3498DB'),
            ('gene_disp', 'Molecular', '#27AE60'),
            ('proj_disp', 'Projection', '#E74C3C'),
        ]

        multi_vals = np.array([r.multi_disp for r in self.region_dispersions])

        for ax, (attr, label, color) in zip(axes, comparisons):
            single_vals = np.array([getattr(r, attr) for r in self.region_dispersions])

            ax.scatter(single_vals, multi_vals, c=color, alpha=0.6, s=30, edgecolor='white')

            # 对角线
            all_vals = np.concatenate([single_vals, multi_vals])
            lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5)

            below = (multi_vals < single_vals).mean() * 100
            ax.text(0.05, 0.95, f'{below:.1f}% below\ndiagonal',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top')

            ax.set_xlabel(f'{label} Dispersion [trace(Cov)/d]', fontsize=10)
            ax.set_ylabel('Multimodal Dispersion [trace(Cov)/d]', fontsize=10)
            ax.set_title(f'Multi vs {label}', fontweight='bold')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Within-region Dispersion Comparison (dimension-normalized)',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_scatter_comparison.png")

    def _plot_dimension_check(self, output_dir: str):
        """维度和离散度检查图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        modalities = ['Morph', 'Gene', 'Proj', 'Multi']
        colors = ['#3498DB', '#27AE60', '#E74C3C', '#9B59B6']

        # 1. 各模态维度
        ax1 = axes[0]
        dims = [self.vector_info[m].pca_dims for m in modalities]
        bars = ax1.bar(modalities, dims, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Dimensions')
        ax1.set_title('PCA Dimensions per Modality', fontweight='bold')
        for bar, d in zip(bars, dims):
            ax1.annotate(f'{d}', xy=(bar.get_x() + bar.get_width()/2, d),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2. 全局离散度（维度归一化）
        ax2 = axes[1]
        disps = [self.global_stats[m].global_dispersion for m in modalities]
        bars = ax2.bar(modalities, disps, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Global Dispersion [trace(Cov)/d]')
        ax2.set_title('Global Dispersion (dimension-normalized)\nComparable across modalities', fontweight='bold')
        for bar, d in zip(bars, disps):
            ax2.annotate(f'{d:.3f}', xy=(bar.get_x() + bar.get_width()/2, d),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.suptitle('Dimension Check: trace(Cov)/d removes dimension effect',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_dimension_check.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 区域离散度详情
        rows = [{
            'region': r.region,
            'n_neurons': r.n_neurons,
            'morph_disp': r.morph_disp,
            'gene_disp': r.gene_disp,
            'proj_disp': r.proj_disp,
            'multi_disp': r.multi_disp,
            'delta_vs_morph': r.multi_disp - r.morph_disp,
            'delta_vs_gene': r.multi_disp - r.gene_disp,
            'delta_vs_proj': r.multi_disp - r.proj_disp,
        } for r in self.region_dispersions]

        pd.DataFrame(rows).to_csv(f"{output_dir}/region_dispersions.csv", index=False)

        # 全局统计
        global_rows = [{
            'modality': s.modality,
            'n_dims': s.n_dims,
            'global_dispersion_normalized': s.global_dispersion,
            'raw_trace_cov': s.raw_trace_cov,
        } for s in self.global_stats.values()]

        pd.DataFrame(global_rows).to_csv(f"{output_dir}/global_stats.csv", index=False)

        # 向量信息
        vector_rows = [{
            'modality': v.name,
            'original_dims': v.original_dims,
            'pca_dims': v.pca_dims,
            'variance_explained': v.variance_explained,
        } for v in self.vector_info.values()]

        pd.DataFrame(vector_rows).to_csv(f"{output_dir}/vector_info.csv", index=False)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./homogeneity_results_v3"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("全脑空间同质性实验 V3")
        print("=" * 80)
        print(f"\n配置:")
        print(f"  PCA方差阈值: {self.pca_variance_threshold:.0%}")
        print(f"  最小神经元数/区域: {self.min_neurons_per_region}")
        print(f"  Subsampling: n={self.subsample_n}, repeats={self.subsample_repeats}")
        print(f"  离散度计算: trace(Cov)/d（维度归一化）")
        print(f"  Multi表示: concat([Morph_PCA, Gene_PCA, Proj_PCA])")

        n = self.load_all_data()
        if n == 0:
            return

        self.compute_region_dispersion(use_subsample=True)
        self.visualize_results(output_dir)
        self.save_results(output_dir)

        print("\n" + "=" * 80)
        print("完成!")
        print("=" * 80)


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./homogeneity_results"

    with WithinRegionHomogeneityExperiment(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            min_neurons_per_region=10,
            subsample_n=50,
            subsample_repeats=50,
    ) as experiment:
        experiment.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()