import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import multiprocessing

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.stats import zscore, sem
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import neo4j

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

N_CORES = multiprocessing.cpu_count()


@dataclass
class ModalityData:
    """存储单个模态的处理后数据"""
    name:str
    vectors_train:np.ndarray
    vectors_test:np.ndarray
    original_dim:int
    reduced_dim:int
    variance_explained:float


@dataclass
class ConfusionResult:
    """存储Confusion Score分析结果"""
    modality:str
    # 原始距离矩阵（未归一化）
    raw_distance_matrix:np.ndarray
    # 归一化后的距离矩阵（用于可视化）
    normalized_matrix:np.ndarray
    # Confusion Score = within / between
    confusion_score:float
    confusion_score_sem:float  # 标准误
    # 原始的within和between距离
    mean_within_raw:float
    mean_between_raw:float
    region_labels:List[str]
    n_neurons_per_region:Dict[str, int]
    # 每个区域的within距离
    within_region_distances:Dict[str, float]
    # 每对区域的between距离
    between_region_distances:Dict[Tuple[str, str], float]


class RegionConfusionAnalysisV2:
    """
    区域混淆度分析 V2 (修正版)

    关键修正：
    1.Confusion Score = mean(within) / mean(between)
       - 越低越好（within距离小，between距离大）
    2.全局归一化：所有模态的热图使用相同的颜色尺度
    3.Train/Test Split：避免过拟合
    4.对等降维：多模态融合前先降到相同维度
    """

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

    def __init__(self, uri:str, user:str, password:str,
                 data_dir:str, database:str = "neo4j",
                 search_radius:float = 4.0,
                 pca_variance_threshold:float = 0.95,
                 min_neurons_per_region:int = 10,
                 distance_metric:str = 'euclidean',
                 test_ratio:float = 0.2,
                 equal_dim_for_fusion:int = 20,  # 对等降维目标维度
                 random_seed:int = 42):

        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.min_neurons_per_region = min_neurons_per_region
        self.distance_metric = distance_metric
        self.test_ratio = test_ratio
        self.equal_dim_for_fusion = equal_dim_for_fusion
        self.random_seed = random_seed

        print(f"Configuration (V2 - Fixed):")
        print(f"  PCA variance threshold:{self.pca_variance_threshold}")
        print(f"  Min neurons per region:{self.min_neurons_per_region}")
        print(f"  Distance metric:{self.distance_metric}")
        print(f"  Test ratio:{self.test_ratio}")
        print(f"  Equal dim for fusion:{self.equal_dim_for_fusion}")
        print(f"  Random seed:{self.random_seed}")

        # Data storage
        self.valid_neuron_ids:List[str] = []
        self.neuron_regions:Dict[str, str] = {}
        self.region_neurons:Dict[str, List[str]] = {}

        # Train/Test split indices (per region stratified)
        self.train_neuron_ids:List[str] = []
        self.test_neuron_ids:List[str] = []

        # Raw features
        self.axon_features_raw:Dict[str, np.ndarray] = {}
        self.dendrite_features_raw:Dict[str, np.ndarray] = {}
        self.local_gene_features_raw:Dict[str, np.ndarray] = {}
        self.projection_vectors_raw:Dict[str, np.ndarray] = {}

        self.all_subclasses:List[str] = []
        self.all_target_regions:List[str] = []
        self.valid_regions:List[str] = []

        # Processed vectors (分开存储train和test)
        self.modality_data:Dict[str, ModalityData] = {}

        # 对等降维后的向量
        self.equal_dim_vectors_train:Dict[str, np.ndarray] = {}
        self.equal_dim_vectors_test:Dict[str, np.ndarray] = {}

        # Results
        self.confusion_results:Dict[str, ConfusionResult] = {}

        # 全局归一化参数
        self.global_max_distance:float = 0.0

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Data Loading ====================

    def load_all_data(self) -> int:
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._load_neuron_regions()
        self._filter_valid_neurons()
        self._filter_valid_regions()
        self._stratified_train_test_split()
        self._process_all_vectors()

        print(f"\n✓ Data loading complete:")
        print(f"  Total neurons:{len(self.valid_neuron_ids)}")
        print(f"  Train neurons:{len(self.train_neuron_ids)}")
        print(f"  Test neurons:{len(self.test_neuron_ids)}")
        print(f"  Valid regions:{len(self.valid_regions)}")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found:{cache_file}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.local_gene_features_raw = cache_data['local_environments']
        self.all_subclasses = cache_data['all_subclasses']
        print(f"  Loaded molecular environment for {len(self.local_gene_features_raw)} neurons")

    def _get_global_dimensions(self):
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]
        print(f"  Projection targets:{len(self.all_target_regions)} brain regions")

    def _load_all_neuron_features(self):
        axon_return = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n:Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
          AND n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
        RETURN n.neuron_id AS neuron_id, {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        proj_query = """
        MATCH (n:Neuron {neuron_id:$neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            records = list(result)

            for record in records:
                neuron_id = record['neuron_id']
                axon_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.AXONAL_FEATURES]
                self.axon_features_raw[neuron_id] = np.array(axon_feats)

                dend_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.DENDRITIC_FEATURES]
                self.dendrite_features_raw[neuron_id] = np.array(dend_feats)

                proj_result = session.run(proj_query, neuron_id=neuron_id)
                proj_dict = {r['target']:r['weight'] for r in proj_result
                             if r['target'] and r['weight']}
                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]
                    self.projection_vectors_raw[neuron_id] = proj_vector

        print(f"  Loaded {len(self.axon_features_raw)} neurons with morphology")

    def _load_neuron_regions(self):
        """加载神经元所属的脑区信息"""
        query = """
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
        RETURN n.neuron_id AS neuron_id, r.acronym AS region
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                neuron_id = record['neuron_id']
                region = record['region']
                if neuron_id and region:
                    self.neuron_regions[neuron_id] = region

        print(f"  Loaded region info for {len(self.neuron_regions)} neurons")

    def _filter_valid_neurons(self):
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        candidates &= set(self.neuron_regions.keys())
        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  Valid neurons (with all modalities + region):{len(self.valid_neuron_ids)}")

    def _filter_valid_regions(self):
        """筛选有足够神经元的脑区"""
        region_counts = {}
        for nid in self.valid_neuron_ids:
            region = self.neuron_regions[nid]
            if region not in region_counts:
                region_counts[region] = []
            region_counts[region].append(nid)

        self.region_neurons = {}
        for region, neurons in region_counts.items():
            if len(neurons) >= self.min_neurons_per_region:
                self.region_neurons[region] = neurons

        self.valid_regions = sorted(self.region_neurons.keys())

        valid_neuron_set = set()
        for neurons in self.region_neurons.values():
            valid_neuron_set.update(neurons)
        self.valid_neuron_ids = sorted(list(valid_neuron_set))

        print(f"  Valid regions (>={self.min_neurons_per_region} neurons):{len(self.valid_regions)}")
        for region in self.valid_regions:
            print(f"    {region}:{len(self.region_neurons[region])} neurons")

    def _stratified_train_test_split(self):
        """分层采样：每个区域按比例划分train/test"""
        np.random.seed(self.random_seed)

        self.train_neuron_ids = []
        self.test_neuron_ids = []

        for region, neurons in self.region_neurons.items():
            neurons = np.array(neurons)
            np.random.shuffle(neurons)

            n_test = max(1, int(len(neurons) * self.test_ratio))

            self.test_neuron_ids.extend(neurons[:n_test].tolist())
            self.train_neuron_ids.extend(neurons[n_test:].tolist())

        print(f"  Stratified split:{len(self.train_neuron_ids)} train, {len(self.test_neuron_ids)} test")

    def _process_all_vectors(self):
        """处理所有模态的向量（在训练集上fit，应用到测试集）"""
        print("\nProcessing vectors (fit on TRAIN, transform on TEST)...")

        # ========== Morphology:Z-score → PCA 95% ==========
        morph_train, morph_test, morph_info = self._process_morph_vector()
        self.modality_data['Morph'] = ModalityData(
            name='Morphology',
            vectors_train=morph_train,
            vectors_test=morph_test,
            original_dim=morph_info['original_dim'],
            reduced_dim=morph_train.shape[1],
            variance_explained=morph_info['variance']
        )

        # ========== Gene:剪枝 → Z-score → PCA 95% ==========
        gene_train, gene_test, gene_info = self._process_gene_vector()
        self.modality_data['Gene'] = ModalityData(
            name='Molecular',
            vectors_train=gene_train,
            vectors_test=gene_test,
            original_dim=gene_info['original_dim'],
            reduced_dim=gene_train.shape[1],
            variance_explained=gene_info['variance']
        )

        # ========== Projection:剪枝 → log1p → Z-score → PCA 95% ==========
        proj_train, proj_test, proj_info = self._process_proj_vector()
        self.modality_data['Proj'] = ModalityData(
            name='Projection',
            vectors_train=proj_train,
            vectors_test=proj_test,
            original_dim=proj_info['original_dim'],
            reduced_dim=proj_train.shape[1],
            variance_explained=proj_info['variance']
        )

        # ========== 对等降维用于多模态融合 ==========
        self._create_equal_dim_vectors()

        # ========== 多模态融合（使用对等降维后的向量） ==========
        self._create_multimodal_vectors()

    def _process_morph_vector(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Morph:Z-score → PCA 95%"""
        # 获取原始数据
        train_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in self.train_neuron_ids
        ])
        test_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in self.test_neuron_ids
        ])

        original_dim = train_raw.shape[1]

        # Z-score (fit on train)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        # PCA (fit on train)
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  Morph:{original_dim}D → {train_pca.shape[1]}D ({variance:.1%} variance)")

        return train_pca, test_pca, {'variance':variance, 'original_dim':original_dim}

    def _process_gene_vector(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Gene:剪枝 → Z-score → PCA 95%"""
        train_raw = np.array([self.local_gene_features_raw[nid] for nid in self.train_neuron_ids])
        test_raw = np.array([self.local_gene_features_raw[nid] for nid in self.test_neuron_ids])

        original_dim = train_raw.shape[1]

        # 剪枝：基于训练集确定有效列
        col_sums = train_raw.sum(axis=0)
        valid_cols = col_sums > 0
        train_pruned = train_raw[:, valid_cols]
        test_pruned = test_raw[:, valid_cols]

        # Z-score (fit on train)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_pruned)
        test_scaled = scaler.transform(test_pruned)

        # PCA (fit on train)
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  Gene:{original_dim}D → {train_pruned.shape[1]}D (pruned) → {train_pca.shape[1]}D ({variance:.1%})")

        return train_pca, test_pca, {'variance':variance, 'original_dim':original_dim}

    def _process_proj_vector(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Proj:剪枝 → log1p → Z-score → PCA 95%"""
        train_raw = np.array([self.projection_vectors_raw[nid] for nid in self.train_neuron_ids])
        test_raw = np.array([self.projection_vectors_raw[nid] for nid in self.test_neuron_ids])

        original_dim = train_raw.shape[1]

        # 剪枝：基于训练集确定有效列
        col_sums = train_raw.sum(axis=0)
        valid_cols = col_sums > 0
        train_pruned = train_raw[:, valid_cols]
        test_pruned = test_raw[:, valid_cols]

        # log1p
        train_log = np.log1p(train_pruned)
        test_log = np.log1p(test_pruned)

        # Z-score (fit on train)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_log)
        test_scaled = scaler.transform(test_log)

        # PCA (fit on train)
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  Proj:{original_dim}D → {train_pruned.shape[1]}D → log1p → {train_pca.shape[1]}D ({variance:.1%})")

        return train_pca, test_pca, {'variance':variance, 'original_dim':original_dim}

    def _create_equal_dim_vectors(self):
        """创建对等维度的向量，用于公平的多模态融合"""
        target_dim = self.equal_dim_for_fusion
        print(f"\nCreating equal dimension vectors ({target_dim}D each for fusion)...")

        for modality in ['Morph', 'Gene', 'Proj']:
            data = self.modality_data[modality]
            train_vec = data.vectors_train
            test_vec = data.vectors_test

            if train_vec.shape[1] > target_dim:
                # 需要进一步降维
                pca = PCA(n_components=target_dim)
                train_reduced = pca.fit_transform(train_vec)
                test_reduced = pca.transform(test_vec)
                print(f"  {modality}:{train_vec.shape[1]}D → {target_dim}D")
            elif train_vec.shape[1] < target_dim:
                # 维度不足，用零填充（或保持原样）
                train_reduced = train_vec
                test_reduced = test_vec
                print(f"  {modality}:{train_vec.shape[1]}D (kept as is, < {target_dim})")
            else:
                train_reduced = train_vec
                test_reduced = test_vec
                print(f"  {modality}:{train_vec.shape[1]}D (already {target_dim}D)")

            self.equal_dim_vectors_train[modality] = train_reduced
            self.equal_dim_vectors_test[modality] = test_reduced

    def _create_multimodal_vectors(self):
        """创建多模态融合向量（使用对等降维后的向量）"""
        print("\nCreating multimodal fusion vectors...")

        # 使用对等降维后的向量进行融合
        combinations = [
            ('Morph+Gene', ['Morph', 'Gene']),
            ('Gene+Proj', ['Gene', 'Proj']),
            ('Morph+Proj', ['Morph', 'Proj']),
            ('All', ['Morph', 'Gene', 'Proj']),
        ]

        for name, modalities in combinations:
            train_vectors = [self.equal_dim_vectors_train[m] for m in modalities]
            test_vectors = [self.equal_dim_vectors_test[m] for m in modalities]

            train_concat = np.hstack(train_vectors)
            test_concat = np.hstack(test_vectors)

            self.modality_data[name] = ModalityData(
                name=name,
                vectors_train=train_concat,
                vectors_test=test_concat,
                original_dim=sum(v.shape[1] for v in train_vectors),
                reduced_dim=train_concat.shape[1],
                variance_explained=0.0  # N/A for concatenation
            )
            print(f"  {name}:{train_concat.shape[1]}D")

    # ==================== Confusion Score Calculation ====================

    def compute_confusion_scores(self):
        """计算所有模态的confusion score（在测试集上评估）"""
        print("\n" + "=" * 80)
        print("Computing Confusion Scores (on TEST set)")
        print("=" * 80)
        print("\nFormula:Confusion Score = mean(within-region) / mean(between-region)")
        print("Lower is better (small within, large between)")

        # 首先计算所有模态的原始距离，找到全局最大值
        all_raw_matrices = {}
        for modality_key, modality_data in self.modality_data.items():
            raw_matrix = self._compute_raw_distance_matrix(modality_key, modality_data)
            all_raw_matrices[modality_key] = raw_matrix

        # 找到全局最大距离（用于统一归一化）
        self.global_max_distance = max(np.max(m) for m in all_raw_matrices.values())
        print(f"\nGlobal max distance for normalization:{self.global_max_distance:.4f}")

        # 计算每个模态的confusion score
        for modality_key, modality_data in self.modality_data.items():
            result = self._compute_confusion_for_modality(
                modality_key, modality_data, all_raw_matrices[modality_key]
            )
            self.confusion_results[modality_key] = result

        # 打印结果摘要
        print("\n" + "-" * 60)
        print("Confusion Score Summary (lower is better):")
        print("-" * 60)
        print(f"{'Modality':<15} {'Score':>10} {'Within':>10} {'Between':>10} {'Dim':>6}")
        print("-" * 60)
        for key, result in sorted(self.confusion_results.items(), key=lambda x:x[1].confusion_score):
            dim = self.modality_data[key].reduced_dim
            print(f"{key:<15} {result.confusion_score:>10.4f} {result.mean_within_raw:>10.4f} "
                  f"{result.mean_between_raw:>10.4f} {dim:>6}")

    def _compute_raw_distance_matrix(self, modality_key:str,
                                     modality_data:ModalityData) -> np.ndarray:
        """计算原始的区域间距离矩阵（未归一化）"""
        vectors = modality_data.vectors_test
        n_regions = len(self.valid_regions)

        # 建立测试集神经元到索引的映射
        neuron_to_idx = {nid:i for i, nid in enumerate(self.test_neuron_ids)}

        distance_matrix = np.zeros((n_regions, n_regions))

        for i, region_i in enumerate(self.valid_regions):
            # 只使用测试集中的神经元
            neurons_i = [nid for nid in self.region_neurons[region_i]
                         if nid in self.test_neuron_ids]
            if not neurons_i:
                continue
            idx_i = [neuron_to_idx[nid] for nid in neurons_i]
            vectors_i = vectors[idx_i]

            for j, region_j in enumerate(self.valid_regions):
                neurons_j = [nid for nid in self.region_neurons[region_j]
                             if nid in self.test_neuron_ids]
                if not neurons_j:
                    continue
                idx_j = [neuron_to_idx[nid] for nid in neurons_j]
                vectors_j = vectors[idx_j]

                # 计算pairwise距离
                pairwise_dist = cdist(vectors_i, vectors_j, metric=self.distance_metric)

                if i == j:
                    # 同一区域内：取上三角（排除自己到自己的距离）
                    triu_indices = np.triu_indices(len(neurons_i), k=1)
                    if len(triu_indices[0]) > 0:
                        mean_dist = np.mean(pairwise_dist[triu_indices])
                    else:
                        mean_dist = 0.0
                else:
                    # 不同区域：取所有距离的均值
                    mean_dist = np.mean(pairwise_dist)

                distance_matrix[i, j] = mean_dist

        return distance_matrix

    def _compute_confusion_for_modality(self, modality_key:str,
                                        modality_data:ModalityData,
                                        raw_distance_matrix:np.ndarray) -> ConfusionResult:
        """计算单个模态的confusion score"""
        print(f"\n--- {modality_key} ---")

        vectors = modality_data.vectors_test
        n_regions = len(self.valid_regions)

        # 使用全局最大值归一化（确保所有模态可比）
        normalized_matrix = raw_distance_matrix / self.global_max_distance

        # 计算within和between距离
        diagonal = np.diag(raw_distance_matrix)
        off_diagonal_mask = ~np.eye(n_regions, dtype=bool)
        off_diagonal = raw_distance_matrix[off_diagonal_mask]

        mean_within = np.mean(diagonal)
        mean_between = np.mean(off_diagonal)

        # Confusion Score = within / between
        if mean_between > 0:
            confusion_score = mean_within / mean_between
        else:
            confusion_score = float('inf')

        # 计算标准误（使用bootstrap或简单估计）
        # 这里简化为使用within和between的标准差
        within_std = np.std(diagonal)
        between_std = np.std(off_diagonal)
        # 简化的误差传播
        if mean_between > 0:
            confusion_score_sem = confusion_score * np.sqrt(
                (within_std / mean_within / np.sqrt(len(diagonal))) ** 2 +
                (between_std / mean_between / np.sqrt(len(off_diagonal))) ** 2
            ) if mean_within > 0 else 0
        else:
            confusion_score_sem = 0

        # 统计每个区域的神经元数量（测试集中）
        n_neurons_per_region = {}
        within_distances = {}
        between_distances = {}

        neuron_to_idx = {nid:i for i, nid in enumerate(self.test_neuron_ids)}

        for i, region in enumerate(self.valid_regions):
            neurons = [nid for nid in self.region_neurons[region]
                       if nid in self.test_neuron_ids]
            n_neurons_per_region[region] = len(neurons)
            within_distances[region] = raw_distance_matrix[i, i]

        for i, region_i in enumerate(self.valid_regions):
            for j, region_j in enumerate(self.valid_regions):
                if i != j:
                    between_distances[(region_i, region_j)] = raw_distance_matrix[i, j]

        print(f"  Dimensions:{modality_data.reduced_dim}")
        print(f"  Mean within-region distance:{mean_within:.4f}")
        print(f"  Mean between-region distance:{mean_between:.4f}")
        print(f"  Confusion Score:{confusion_score:.4f} ± {confusion_score_sem:.4f}")

        return ConfusionResult(
            modality=modality_key,
            raw_distance_matrix=raw_distance_matrix,
            normalized_matrix=normalized_matrix,
            confusion_score=confusion_score,
            confusion_score_sem=confusion_score_sem,
            mean_within_raw=mean_within,
            mean_between_raw=mean_between,
            region_labels=self.valid_regions,
            n_neurons_per_region=n_neurons_per_region,
            within_region_distances=within_distances,
            between_region_distances=between_distances
        )

    # ==================== Visualization ====================

    def visualize_results(self, output_dir:str = "."):
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_single_modality_heatmaps(output_dir)
        self._plot_multimodal_heatmaps(output_dir)
        self._plot_all_heatmaps_unified(output_dir)
        self._plot_confusion_score_comparison(output_dir)
        self._plot_within_vs_between(output_dir)
        self._plot_comprehensive_analysis(output_dir)

        print(f"\n✓ Figures saved to:{output_dir}")

    def _create_custom_colormap(self):
        """创建类似参考图的红色渐变colormap"""
        colors = ['white', '#fee0d2', '#fc9272', '#de2d26', '#a50f15']
        return LinearSegmentedColormap.from_list('confusion_cmap', colors, N=256)

    def _save_figure(self, fig, output_dir:str, filename:str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_single_modality_heatmaps(self, output_dir:str):
        """绘制单模态的热力图（使用统一的颜色尺度）"""
        single_modalities = ['Morph', 'Gene', 'Proj']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap = self._create_custom_colormap()

        # 使用统一的颜色尺度 (0 到 1，因为已全局归一化)
        vmin, vmax = 0, 1

        for ax, modality in zip(axes, single_modalities):
            if modality not in self.confusion_results:
                continue

            result = self.confusion_results[modality]
            matrix = result.normalized_matrix
            labels = result.region_labels
            score = result.confusion_score

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

            modality_names = {'Morph':'Morphology', 'Gene':'Molecular', 'Proj':'Projection'}
            ax.set_title(f'{modality_names.get(modality, modality)}', fontsize=12, fontweight='bold')

            # 红色分数
            ax.text(0.5, 1.05, f'Confusion Score:{score:.3f}', transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='#c0392b', ha='center')

            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Normalized\ndistance', fontsize=9)

        plt.suptitle('Single Modality Distance Matrices\n(Globally Normalized)',
                     fontsize=14, fontweight='bold', y=1.08)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_single_modality_heatmaps.png")

    def _plot_multimodal_heatmaps(self, output_dir:str):
        """绘制多模态融合的热力图（使用统一的颜色尺度）"""
        multi_modalities = ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        cmap = self._create_custom_colormap()
        vmin, vmax = 0, 1

        for ax, modality in zip(axes, multi_modalities):
            if modality not in self.confusion_results:
                continue

            result = self.confusion_results[modality]
            matrix = result.normalized_matrix
            labels = result.region_labels
            score = result.confusion_score

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

            modality_display = modality.replace('+', ' + ')
            if modality == 'All':
                modality_display = 'All Modalities'
            ax.set_title(f'{modality_display}', fontsize=11, fontweight='bold')

            ax.text(0.5, 1.05, f'Confusion Score:{score:.3f}', transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='#c0392b', ha='center')

            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Normalized\ndistance', fontsize=9)

        plt.suptitle('Multimodal Fusion Distance Matrices\n(Equal Dim + Globally Normalized)',
                     fontsize=14, fontweight='bold', y=1.08)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_multimodal_heatmaps.png")

    def _plot_all_heatmaps_unified(self, output_dir:str):
        """绘制所有模态的热力图在一张图上，使用统一色标"""
        all_modalities = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']

        fig = plt.figure(figsize=(24, 10))
        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

        cmap = self._create_custom_colormap()
        vmin, vmax = 0, 1

        axes = []
        for i, modality in enumerate(all_modalities):
            if i < 4:
                ax = fig.add_subplot(gs[0, i])
            else:
                ax = fig.add_subplot(gs[1, i - 4])
            axes.append(ax)

            if modality not in self.confusion_results:
                continue

            result = self.confusion_results[modality]
            matrix = result.normalized_matrix
            labels = result.region_labels
            score = result.confusion_score

            # 标记是单模态还是多模态
            is_multi = '+' in modality or modality == 'All'
            title_color = '#e74c3c' if is_multi else '#3498db'

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)

            # 标题带边框颜色区分
            title_text = modality.replace('+', '+\n') if '+' in modality else modality
            ax.set_title(title_text, fontsize=11, fontweight='bold',
                         color=title_color, pad=10)

            # 分数
            ax.text(0.5, -0.15, f'Score:{score:.3f}', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color='#c0392b', ha='center')

        # 添加统一的颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Normalized Mean Distance\n(Global Scale)', fontsize=11)

        # 图例
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.3, edgecolor='#3498db',
                  linewidth=2, label='Single Modality'),
            Patch(facecolor='#e74c3c', alpha=0.3, edgecolor='#e74c3c',
                  linewidth=2, label='Multi-Modal Fusion'),
        ]
        fig.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(0.9, 0.98), fontsize=10)

        plt.suptitle('Region Distance Matrices:Single vs Multi-Modal\n'
                     '(All matrices use the same global color scale)',
                     fontsize=14, fontweight='bold', y=0.98)

        self._save_figure(fig, output_dir, "3_all_heatmaps_unified.png")

    def _plot_confusion_score_comparison(self, output_dir:str):
        """绘制confusion score对比条形图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：所有模态的对比（水平条形图，按分数排序）
        ax1 = axes[0]

        modalities = list(self.confusion_results.keys())
        scores = [self.confusion_results[m].confusion_score for m in modalities]
        sems = [self.confusion_results[m].confusion_score_sem for m in modalities]

        sorted_indices = np.argsort(scores)
        modalities_sorted = [modalities[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]
        sems_sorted = [sems[i] for i in sorted_indices]

        colors = []
        for m in modalities_sorted:
            if '+' in m or m == 'All':
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')

        y_pos = np.arange(len(modalities_sorted))
        bars = ax1.barh(y_pos, scores_sorted, xerr=sems_sorted, color=colors,
                        edgecolor='black', linewidth=1, capsize=3)

        for i, (bar, score, sem_val) in enumerate(zip(bars, scores_sorted, sems_sorted)):
            ax1.text(score + max(scores_sorted) * 0.02, i, f'{score:.3f}±{sem_val:.3f}',
                     va='center', fontsize=9, fontweight='bold')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(modalities_sorted, fontsize=11)
        ax1.set_xlabel('Confusion Score (within/between, lower is better)', fontsize=11)
        ax1.set_title('Confusion Score Comparison\n(All Modalities)', fontsize=13, fontweight='bold')
        ax1.axvline(x=1.0, color = 'gray', linestyle = '--', alpha = 0.5)
        ax1.set_xlim(0, max(scores_sorted) * 1.25)
        ax1.grid(axis='x', alpha=0.3)

        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Single Modality'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Multi-Modal Fusion'),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

        # 右图：改进百分比
        ax2 = axes[1]

        single_modalities = ['Morph', 'Gene', 'Proj']
        single_scores = [self.confusion_results[m].confusion_score for m in single_modalities]

        # 计算相对于最佳单模态的改进
        best_single = min(single_scores)
        best_single_name = single_modalities[single_scores.index(best_single)]

        multi_modalities = ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']
        improvements = []
        for m in multi_modalities:
            if m in self.confusion_results:
                multi_score = self.confusion_results[m].confusion_score
                # 改进百分比 = (single - multi) / single * 100
                improvement = (best_single - multi_score) / best_single * 100
                improvements.append(improvement)
            else:
                improvements.append(0)

        colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax2.bar(range(len(multi_modalities)), improvements, color=colors,
                       edgecolor='black', linewidth=1)

        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            va = 'bottom' if imp >= 0 else 'top'
            offset = 1 if imp >= 0 else -1
            ax2.annotate(f'{imp:+.1f}%', xy=(i, imp), xytext=(0, offset * 5),
                         textcoords='offset points', ha='center', va=va,
                         fontsize=11, fontweight='bold')

        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_xticks(range(len(multi_modalities)))
        ax2.set_xticklabels(multi_modalities, rotation=15, ha='right', fontsize=11)
        ax2.set_ylabel(f'Improvement vs Best Single ({best_single_name})', fontsize=11)
        ax2.set_title(f'Fusion Improvement\n(Relative to Best Single:{best_single_name}={best_single:.3f})',
                      fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_confusion_score_comparison.png")

    def _plot_within_vs_between(self, output_dir:str):
        """绘制within vs between距离的对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        modalities = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']
        within_vals = []
        between_vals = []

        for m in modalities:
            if m in self.confusion_results:
                result = self.confusion_results[m]
                within_vals.append(result.mean_within_raw)
                between_vals.append(result.mean_between_raw)

        # 左图：条形图对比
        ax1 = axes[0]
        x = np.arange(len(modalities))
        width = 0.35

        colors_within = ['#3498db' if '+' not in m and m != 'All' else '#9b59b6' for m in modalities]
        colors_between = ['#2ecc71' if '+' not in m and m != 'All' else '#e67e22' for m in modalities]

        bars1 = ax1.bar(x - width / 2, within_vals, width, label='Within-region',
                        color='#3498db', edgecolor='black', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, between_vals, width, label='Between-region',
                        color='#e74c3c', edgecolor='black', alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels(modalities, rotation=30, ha='right', fontsize=10)
        ax1.set_ylabel('Mean Distance (raw)', fontsize=11)
        ax1.set_title('Within-Region vs Between-Region Distances', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # 右图：散点图 (within vs between)
        ax2 = axes[1]

        colors = ['#3498db' if '+' not in m and m != 'All' else '#e74c3c' for m in modalities]

        for i, (w, b, m, c) in enumerate(zip(within_vals, between_vals, modalities, colors)):
            ax2.scatter(b, w, s=150, c=c, edgecolors='black', linewidth=1.5, zorder = 5)
            ax2.annotate(m, xy=(b, w), xytext=(5, 5), textcoords='offset points',
                         fontsize=9, fontweight='bold')

        # 添加等比例线
        max_val = max(max(within_vals), max(between_vals))
        min_val = min(min(within_vals), min(between_vals))
        ax2.plot([min_val * 0.9, max_val * 1.1], [min_val * 0.9, max_val * 1.1],
                 'k--', alpha=0.3, label='within = between')

        # 填充区域
        ax2.fill_between([min_val * 0.9, max_val * 1.1], [min_val * 0.9, max_val * 1.1],
                         [0, 0], alpha=0.1, color='green', label='Good separation')
        ax2.fill_between([min_val * 0.9, max_val * 1.1], [min_val * 0.9, max_val * 1.1],
                         [max_val * 1.1, max_val * 1.1], alpha=0.1, color='red', label='Poor separation')

        ax2.set_xlabel('Between-Region Distance', fontsize=11)
        ax2.set_ylabel('Within-Region Distance', fontsize=11)
        ax2.set_title('Separability Space\n(Lower-right = better separation)',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(alpha=0.3)

        # 添加图例区分单/多模态
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
                       markersize=12, markeredgecolor='black', label='Single Modality'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                       markersize=12, markeredgecolor='black', label='Multi-Modal'),
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_within_vs_between.png")

    def _plot_comprehensive_analysis(self, output_dir:str):
        """综合分析图"""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 4, height_ratios=[1.2, 1, 0.8], hspace=0.35, wspace=0.3)

        cmap = self._create_custom_colormap()
        vmin, vmax = 0, 1

        # ========== 第一行：单模态 + All ==========
        modalities_row1 = ['Morph', 'Gene', 'Proj', 'All']

        for idx, modality in enumerate(modalities_row1):
            ax = fig.add_subplot(gs[0, idx])

            if modality not in self.confusion_results:
                continue

            result = self.confusion_results[modality]
            matrix = result.normalized_matrix
            labels = result.region_labels
            score = result.confusion_score

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)

            is_multi = modality == 'All'
            title_color = '#e74c3c' if is_multi else '#3498db'
            ax.set_title(modality if modality != 'All' else 'All Modalities',
                         fontsize=12, fontweight='bold', color=title_color)

            ax.text(0.5, -0.12, f'Score:{score:.3f}', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color='#c0392b', ha='center')

            if idx == 3:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Normalized distance', fontsize=9)

        # ========== 第二行：分数排名 + 改进图 ==========
        ax_rank = fig.add_subplot(gs[1, :2])

        modalities = list(self.confusion_results.keys())
        scores = [self.confusion_results[m].confusion_score for m in modalities]
        sorted_idx = np.argsort(scores)

        colors = ['#e74c3c' if '+' in modalities[i] or modalities[i] == 'All' else '#3498db'
                  for i in sorted_idx]

        y_pos = np.arange(len(modalities))
        bars = ax_rank.barh(y_pos, [scores[i] for i in sorted_idx],
                            color=colors, edgecolor='black', linewidth=1)

        for i, idx_val in enumerate(sorted_idx):
            score = scores[idx_val]
            ax_rank.text(score + max(scores) * 0.02, i, f'{score:.3f}',
                         va='center', fontsize=9, fontweight='bold')

        ax_rank.set_yticks(y_pos)
        ax_rank.set_yticklabels([modalities[i] for i in sorted_idx], fontsize=10)
        ax_rank.set_xlabel('Confusion Score (lower = better)', fontsize=11)
        ax_rank.set_title('Confusion Score Ranking', fontsize=12, fontweight='bold')
        ax_rank.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax_rank.grid(axis='x', alpha=0.3)

        # Within vs Between scatter
        ax_scatter = fig.add_subplot(gs[1, 2:])

        all_modalities = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']
        within_vals = [self.confusion_results[m].mean_within_raw for m in all_modalities]
        between_vals = [self.confusion_results[m].mean_between_raw for m in all_modalities]

        colors_scatter = ['#3498db' if '+' not in m and m != 'All' else '#e74c3c' for m in all_modalities]

        for w, b, m, c in zip(within_vals, between_vals, all_modalities, colors_scatter):
            ax_scatter.scatter(b, w, s=120, c=c, edgecolors='black', linewidth=1.5, zorder = 5)
            ax_scatter.annotate(m, xy=(b, w), xytext=(5, 5), textcoords='offset points',
                                fontsize=8, fontweight='bold')

        max_val = max(max(within_vals), max(between_vals)) * 1.1
        min_val = min(min(within_vals), min(between_vals)) * 0.9
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
        ax_scatter.fill_between([min_val, max_val], [min_val, max_val], [0, 0],
                                alpha=0.1, color='green')

        ax_scatter.set_xlabel('Between-Region Distance', fontsize=11)
        ax_scatter.set_ylabel('Within-Region Distance', fontsize=11)
        ax_scatter.set_title('Separability Space\n(Lower-right = better)',
                             fontsize=12, fontweight='bold')
        ax_scatter.grid(alpha=0.3)

        # ========== 第三行：统计摘要 ==========
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')

        # 计算统计信息
        single_scores = {m:self.confusion_results[m].confusion_score
                         for m in ['Morph', 'Gene', 'Proj']}
        multi_scores = {m:self.confusion_results[m].confusion_score
                        for m in ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']}

        best_single = min(single_scores.items(), key=lambda x:x[1])
        worst_single = max(single_scores.items(), key=lambda x:x[1])
        best_multi = min(multi_scores.items(), key=lambda x:x[1])

        improvement = (best_single[1] - best_multi[1]) / best_single[1] * 100

        # 创建表格形式的摘要
        summary_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['─' * 20, '─' * 15, '─' * 30],
            ['Best Single Modality', f'{best_single[0]}:{best_single[1]:.4f}',
             'Baseline for comparison'],
            ['Worst Single Modality', f'{worst_single[0]}:{worst_single[1]:.4f}',
             'Most confused modality'],
            ['Best Multi-Modal', f'{best_multi[0]}:{best_multi[1]:.4f}',
             'Best fusion result'],
            ['Improvement', f'{improvement:+.2f}%',
             'Positive = fusion helps' if improvement > 0 else 'Negative = fusion hurts'],
            ['─' * 20, '─' * 15, '─' * 30],
            ['Global Max Distance', f'{self.global_max_distance:.4f}',
             'Used for normalization'],
            ['Number of Regions', f'{len(self.valid_regions)}',
             ', '.join(self.valid_regions[:5]) + ('...' if len(self.valid_regions) > 5 else '')],
            ['Test Set Size', f'{len(self.test_neuron_ids)}',
             'Neurons used for evaluation'],
        ]

        # 绘制表格
        table_text = '\n'.join([f'{row[0]:<25} {row[1]:<20} {row[2]}' for row in summary_data])

        ax_summary.text(0.5, 0.5, table_text, transform=ax_summary.transAxes,
                        fontsize=11, fontfamily='monospace', verticalalignment='center',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                                  edgecolor='#dee2e6', alpha=0.9))

        plt.suptitle('Comprehensive Region Confusion Analysis (V2)\n'
                     'Formula:Confusion Score = Within/Between (lower is better)',
                     fontsize=14, fontweight='bold', y=0.98)

        self._save_figure(fig, output_dir, "6_comprehensive_analysis.png")

    def _plot_publication_ready_heatmaps(self, output_dir:str):
        """生成类似参考图的精美热力图（用于发表）"""
        # 单模态图
        fig_single, axes_single = plt.subplots(1, 3, figsize=(12, 4))
        cmap = self._create_custom_colormap()

        single_modalities = ['Morph', 'Gene', 'Proj']
        titles = ['Morphology', 'Molecular', 'Projection']

        for ax, modality, title in zip(axes_single, single_modalities, titles):
            result = self.confusion_results[modality]
            matrix = result.normalized_matrix
            labels = result.region_labels
            score = result.confusion_score

            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)

            # 标题格式类似参考图
            ax.text(0.5, 1.15, f'Confusion score in {title}:',
                    transform=ax.transAxes, fontsize=11, ha='center')
            ax.text(0.5, 1.05, f'{score:.3f}',
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color='#c0392b', ha='center')

            ax.set_xlabel(f'{title} features', fontsize=10)
            ax.set_ylabel('Brain regions', fontsize=10)

        # 添加共享颜色条
        cbar_ax = fig_single.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig_single.colorbar(im, cax=cbar_ax)
        cbar.set_label('Normalized\nmean distance', fontsize=10)
        cbar.set_ticks([0, 0.5, 1.0])

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        self._save_figure(fig_single, output_dir, "7_publication_single_modality.png")

        # 多模态图
        fig_multi, axes_multi = plt.subplots(1, 4, figsize=(16, 4))

        multi_modalities = ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']
        multi_titles = ['Morph + Mol', 'Mol + Proj', 'Morph + Proj', 'All Modalities']

        for ax, modality, title in zip(axes_multi, multi_modalities, multi_titles):
            result = self.confusion_results[modality]
            matrix = result.normalized_matrix
            labels = result.region_labels
            score = result.confusion_score

            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

            ax.text(0.5, 1.15, f'Confusion score in',
                    transform=ax.transAxes, fontsize=10, ha='center')
            ax.text(0.5, 1.05, f'{title}:{score:.3f}',
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    color='#c0392b', ha='center')

        cbar_ax = fig_multi.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig_multi.colorbar(im, cax=cbar_ax)
        cbar.set_label('Normalized\nmean distance', fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.91, 1])
        self._save_figure(fig_multi, output_dir, "8_publication_multimodal.png")

        # ==================== Save Results ====================

    def save_results(self, output_dir:str = "."):
        os.makedirs(output_dir, exist_ok=True)

        # Save confusion scores
        rows = []
        for modality, result in self.confusion_results.items():
            rows.append({
                'modality':modality,
                'confusion_score':result.confusion_score,
                'confusion_score_sem':result.confusion_score_sem,
                'mean_within_distance':result.mean_within_raw,
                'mean_between_distance':result.mean_between_raw,
                'n_regions':len(result.region_labels),
                'dimensions':self.modality_data[modality].reduced_dim,
            })
        df_scores = pd.DataFrame(rows)
        df_scores = df_scores.sort_values('confusion_score')
        df_scores.to_csv(f"{output_dir}/confusion_scores.csv", index=False)

        # Save distance matrices
        for modality, result in self.confusion_results.items():
            # Raw distance matrix
            df_raw = pd.DataFrame(result.raw_distance_matrix,
                                  index=result.region_labels,
                                  columns=result.region_labels)
            df_raw.to_csv(f"{output_dir}/distance_matrix_raw_{modality}.csv")

            # Normalized distance matrix
            df_norm = pd.DataFrame(result.normalized_matrix,
                                   index=result.region_labels,
                                   columns=result.region_labels)
            df_norm.to_csv(f"{output_dir}/distance_matrix_normalized_{modality}.csv")

        # Save region info
        region_info = []
        for region in self.valid_regions:
            n_train = len([nid for nid in self.region_neurons[region]
                           if nid in self.train_neuron_ids])
            n_test = len([nid for nid in self.region_neurons[region]
                          if nid in self.test_neuron_ids])
            region_info.append({
                'region':region,
                'n_neurons_total':len(self.region_neurons[region]),
                'n_neurons_train':n_train,
                'n_neurons_test':n_test,
            })
        pd.DataFrame(region_info).to_csv(f"{output_dir}/region_info.csv", index=False)

        # Save full results as pickle
        with open(f"{output_dir}/full_confusion_results.pkl", 'wb') as f:
            pickle.dump({
                'confusion_results':self.confusion_results,
                'modality_data':{k:{'name':v.name, 'dim':v.reduced_dim}
                                  for k, v in self.modality_data.items()},
                'valid_regions':self.valid_regions,
                'global_max_distance':self.global_max_distance,
                'config':{
                    'pca_variance_threshold':self.pca_variance_threshold,
                    'min_neurons_per_region':self.min_neurons_per_region,
                    'distance_metric':self.distance_metric,
                    'test_ratio':self.test_ratio,
                    'equal_dim_for_fusion':self.equal_dim_for_fusion,
                    'random_seed':self.random_seed,
                }
            }, f)

        print(f"\n✓ Results saved to:{output_dir}")

    def run_full_pipeline(self, output_dir:str = "./confusion_analysis_v2"):
        print("\n" + "=" * 80)
        print("Region Confusion Score Analysis V2 (Fixed)")
        print("=" * 80)
        print("\nKey improvements:")
        print("  1.Confusion Score = within/between (lower is better)")
        print("  2.Global normalization for fair comparison across modalities")
        print("  3.Train/Test split to avoid overfitting")
        print("  4.Equal dimensionality for fair multimodal fusion")

        n = self.load_all_data()
        if n == 0:
            print("No valid data found!")
            return

        self.compute_confusion_scores()
        self.visualize_results(output_dir)
        self._plot_publication_ready_heatmaps(output_dir)
        self.save_results(output_dir)

        self._print_final_summary()

    def _print_final_summary(self):
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        print("\n【Configuration】")
        print(f"  PCA variance threshold:{self.pca_variance_threshold}")
        print(f"  Equal dim for fusion:{self.equal_dim_for_fusion}")
        print(f"  Test ratio:{self.test_ratio}")
        print(f"  Distance metric:{self.distance_metric}")

        print("\n【Data】")
        print(f"  Valid regions:{len(self.valid_regions)}")
        print(f"  Train neurons:{len(self.train_neuron_ids)}")
        print(f"  Test neurons:{len(self.test_neuron_ids)}")
        print(f"  Global max distance:{self.global_max_distance:.4f}")

        print("\n【Confusion Scores】(lower = better separation)")
        print(f"{'Modality':<15} {'Score':>10} {'Within':>10} {'Between':>10}")
        print("-" * 50)

        sorted_results = sorted(self.confusion_results.items(),
                                key=lambda x:x[1].confusion_score)

        for modality, result in sorted_results:
            marker = "★" if '+' in modality or modality == 'All' else " "
            print(f"{marker} {modality:<13} {result.confusion_score:>10.4f} "
                  f"{result.mean_within_raw:>10.4f} {result.mean_between_raw:>10.4f}")

        # 计算改进
        single_best = min(self.confusion_results[m].confusion_score
                          for m in ['Morph', 'Gene', 'Proj'])
        multi_best = min(self.confusion_results[m].confusion_score
                         for m in ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All'])

        improvement = (single_best - multi_best) / single_best * 100

        print("\n【Conclusion】")
        print(f"  Best single modality score:{single_best:.4f}")
        print(f"  Best multi-modal score:{multi_best:.4f}")
        print(f"  Improvement from fusion:{improvement:+.2f}%")

        if improvement > 0:
            print("  ✓ Multi-modal fusion IMPROVES region separability!")
        else:
            print("  ✗ Multi-modal fusion does NOT improve separability.")

def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./confusion_analysis_v2"

    with RegionConfusionAnalysisV2(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
    pca_variance_threshold=0.95,
    min_neurons_per_region=10,
    distance_metric='euclidean',
    test_ratio=0.2,
    equal_dim_for_fusion=20,  # 对等降维到20维
    random_seed=42,
    ) as analysis:
        analysis.run_full_pipeline(output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()