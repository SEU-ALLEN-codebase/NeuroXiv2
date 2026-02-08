import neo4j
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BrainRegionFingerprintsPCA:
    """脑区指纹计算类 - PCA版本（64维形态特征）"""

    # 32个Axonal特征（与clustering_exploration_v4一致）
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

    # 32个Dendritic特征
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

    def __init__(self, uri: str, user: str, password: str, pca_variance_threshold: float = 0.95):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
            pca_variance_threshold: PCA方差解释阈值，默认95%
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.pca_variance_threshold = pca_variance_threshold

        # 存储计算结果
        self.regions = []

        # 原始指纹
        self.mol_signatures_raw = {}
        self.morph_signatures_raw = {}
        self.proj_signatures_raw = {}

        # PCA处理后的指纹
        self.mol_signatures = {}
        self.morph_signatures = {}
        self.proj_signatures = {}

        # PCA模型和维度信息
        self.mol_pca = None
        self.morph_pca = None
        self.proj_pca = None

        self.mol_dim_info = {}
        self.morph_dim_info = {}
        self.proj_dim_info = {}

        self.all_subclasses = []
        self.all_target_subregions = []

        # 合并后的形态特征名称（64维）
        self.morph_feature_names = self.AXONAL_FEATURES + self.DENDRITIC_FEATURES

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 1. 分子指纹 (Molecular Fingerprint) ====================

    def get_all_subclasses(self):
        """获取全局所有subclass的列表"""
        query = """
        MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass)
        RETURN DISTINCT sc.name AS subclass_name
        ORDER BY subclass_name
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.all_subclasses = [record['subclass_name'] for record in result]
        print(f"找到 {len(self.all_subclasses)} 个全局subclass")
        return self.all_subclasses

    def compute_molecular_signature_raw(self, region: str) -> np.ndarray:
        """计算单个脑区的原始分子指纹"""
        query = """
        MATCH (r:Region {acronym: $region})
        MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
        ORDER BY subclass_name
        """
        with self.driver.session() as session:
            result = session.run(query, region=region)
            data = {record['subclass_name']: record['pct_cells'] for record in result}

        signature = np.zeros(len(self.all_subclasses))
        for i, subclass in enumerate(self.all_subclasses):
            if subclass in data:
                signature[i] = data[subclass]
        return signature

    def compute_all_molecular_signatures(self):
        """计算所有脑区的分子指纹并进行PCA"""
        print("\n=== 计算分子指纹 ===")

        # 获取所有脑区
        query = """
        MATCH (r:Region)
        WHERE EXISTS((r)-[:HAS_SUBCLASS]->())
        RETURN r.acronym AS acronym
        ORDER BY acronym
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.regions = [record['acronym'] for record in result]
        print(f"找到 {len(self.regions)} 个有分子数据的脑区")

        # 计算原始指纹
        for region in self.regions:
            sig = self.compute_molecular_signature_raw(region)
            self.mol_signatures_raw[region] = sig

        # 构建矩阵 [n_regions, n_subclasses]
        raw_matrix = np.array([self.mol_signatures_raw[r] for r in self.regions])
        print(f"原始分子指纹矩阵: {raw_matrix.shape}")

        # Step 1: 剪枝 - 去掉全零列（零值subclass）
        col_sums = np.sum(raw_matrix, axis=0)
        nonzero_cols = col_sums > 0
        n_removed = np.sum(~nonzero_cols)
        pruned_matrix = raw_matrix[:, nonzero_cols]
        print(f"Step 1 剪枝: 去掉 {n_removed} 个零值subclass，剩余 {pruned_matrix.shape[1]} 维")

        # Step 2: Z-score标准化（按列）
        scaler = StandardScaler()
        zscore_matrix = scaler.fit_transform(pruned_matrix)
        print(f"Step 2 Z-score标准化完成")

        # Step 3: PCA
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca_matrix = pca.fit_transform(zscore_matrix)
        self.mol_pca = pca
        print(f"Step 3 PCA: {pruned_matrix.shape[1]} 维 -> {pca_matrix.shape[1]} 维 "
              f"(方差解释: {pca.explained_variance_ratio_.sum()*100:.2f}%)")

        # 保存维度信息
        self.mol_dim_info = {
            'original_dim': len(self.all_subclasses),
            'after_pruning': pruned_matrix.shape[1],
            'after_pca': pca_matrix.shape[1],
            'variance_explained': pca.explained_variance_ratio_.sum(),
            'n_removed_subclasses': n_removed
        }

        # 保存PCA后的指纹
        for i, region in enumerate(self.regions):
            self.mol_signatures[region] = pca_matrix[i]

        print(f"完成 {len(self.mol_signatures)} 个分子指纹计算")

    # ==================== 2. 形态指纹 (Morphology Fingerprint) - 64维 ====================

    def compute_morphology_signature_raw(self, region: str) -> np.ndarray:
        """
        计算单个脑区的原始形态指纹（64维）
        从Region节点获取聚合后的形态特征（需要先在Region上聚合神经元特征）

        注意：这里假设Region节点上有这64个属性的聚合值（如mean）
        如果没有，需要从Neuron节点聚合
        """
        # 首先尝试直接从Region节点获取
        query_direct = """
        MATCH (r:Region {acronym: $region})
        RETURN r
        """

        with self.driver.session() as session:
            result = session.run(query_direct, region=region)
            record = result.single()

        if not record:
            return np.array([np.nan] * 64)

        node = record['r']

        # 尝试获取64个形态特征
        signature = []
        for feat in self.morph_feature_names:
            val = node.get(feat, None)
            if val is None:
                # 尝试不同的命名变体
                val = node.get(feat + '_mean', None)
            signature.append(val if val is not None else np.nan)

        # 检查有效值数量
        valid_count = np.sum(~np.isnan(signature))

        if valid_count < 10:
            # Region节点上没有足够的形态特征，需要从Neuron聚合
            return self._aggregate_morph_from_neurons(region)

        return np.array(signature)

    def _aggregate_morph_from_neurons(self, region: str) -> np.ndarray:
        """从该区域的神经元聚合形态特征"""
        # 构建查询
        axon_return = [f"avg(n.{feat}) AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"avg(n.{feat}) AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {{acronym: $region}})
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
        RETURN {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        with self.driver.session() as session:
            result = session.run(query, region=region)
            record = result.single()

        if not record:
            return np.array([np.nan] * 64)

        signature = []
        for feat in self.AXONAL_FEATURES:
            val = record[feat]
            signature.append(val if val is not None else np.nan)
        for feat in self.DENDRITIC_FEATURES:
            val = record[feat]
            signature.append(val if val is not None else np.nan)

        return np.array(signature)

    def compute_all_morphology_signatures(self):
        """计算所有脑区的形态指纹并进行PCA（64维）"""
        print("\n=== 计算形态指纹 (64维) ===")

        # 计算原始指纹
        for region in self.regions:
            sig = self.compute_morphology_signature_raw(region)
            self.morph_signatures_raw[region] = sig

        raw_matrix = np.array([self.morph_signatures_raw[r] for r in self.regions])
        print(f"原始形态指纹矩阵: {raw_matrix.shape}")

        # 统计有效值
        valid_per_col = np.sum(~np.isnan(raw_matrix), axis=0)
        valid_cols_mask = valid_per_col > len(self.regions) * 0.3  # 至少30%的脑区有值
        n_valid_cols = np.sum(valid_cols_mask)
        print(f"有效特征列: {n_valid_cols}/{raw_matrix.shape[1]}")

        # 处理缺失值：使用列均值填充
        col_means = np.nanmean(raw_matrix, axis=0)
        for j in range(raw_matrix.shape[1]):
            mask = np.isnan(raw_matrix[:, j])
            if np.isnan(col_means[j]):
                raw_matrix[mask, j] = 0
            else:
                raw_matrix[mask, j] = col_means[j]

        # 去掉零方差列
        col_vars = np.var(raw_matrix, axis=0)
        valid_cols = col_vars > 1e-10
        n_removed = np.sum(~valid_cols)
        pruned_matrix = raw_matrix[:, valid_cols]
        print(f"去掉 {n_removed} 个零方差列，剩余 {pruned_matrix.shape[1]} 维")

        # Step 1: Log1p变换（与clustering代码一致）
        log_matrix = np.log1p(np.abs(pruned_matrix))  # 使用abs处理可能的负值
        print(f"Step 1 Log1p变换完成")

        # Step 2: Z-score标准化（按列）
        scaler = StandardScaler()
        zscore_matrix = scaler.fit_transform(log_matrix)
        print(f"Step 2 Z-score标准化完成")

        # Step 3: PCA
        n_comp = min(self.pca_variance_threshold, pruned_matrix.shape[1], len(self.regions) - 1)
        if isinstance(n_comp, float):
            pca = PCA(n_components=n_comp, svd_solver='full')
        else:
            pca = PCA(n_components=int(n_comp))

        pca_matrix = pca.fit_transform(zscore_matrix)
        self.morph_pca = pca
        print(f"Step 3 PCA: {pruned_matrix.shape[1]} 维 -> {pca_matrix.shape[1]} 维 "
              f"(方差解释: {pca.explained_variance_ratio_.sum()*100:.2f}%)")

        # 保存维度信息
        self.morph_dim_info = {
            'original_dim': 64,
            'after_pruning': pruned_matrix.shape[1],
            'after_pca': pca_matrix.shape[1],
            'variance_explained': pca.explained_variance_ratio_.sum(),
            'n_removed_cols': n_removed
        }

        # 保存PCA后的指纹
        for i, region in enumerate(self.regions):
            self.morph_signatures[region] = pca_matrix[i]

        print(f"完成 {len(self.morph_signatures)} 个形态指纹计算")

    # ==================== 3. 投射指纹 (Projection Fingerprint) ====================

    def get_all_target_subregions(self):
        """获取全局所有投射目标subregion的列表"""
        query = """
        MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN DISTINCT t.acronym AS target_subregion
        ORDER BY target_subregion
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.all_target_subregions = [record['target_subregion'] for record in result]
        print(f"找到 {len(self.all_target_subregions)} 个全局投射目标subregion")
        return self.all_target_subregions

    def compute_projection_signature_raw(self, region: str) -> np.ndarray:
        """计算单个脑区的原始投射指纹"""
        query = """
        MATCH (r:Region {acronym: $region})
        OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT]->(r)
        OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r)
        OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r)
        WITH r, (COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) + COLLECT(DISTINCT n3)) AS ns
        UNWIND ns AS n
        WITH DISTINCT n
        WHERE n IS NOT NULL
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        WITH t.acronym AS tgt_subregion, SUM(p.weight) AS total_weight_to_tgt
        RETURN tgt_subregion, total_weight_to_tgt
        ORDER BY total_weight_to_tgt DESC
        """
        with self.driver.session() as session:
            result = session.run(query, region=region)
            data = {record['tgt_subregion']: record['total_weight_to_tgt'] for record in result}

        raw_values = np.zeros(len(self.all_target_subregions))
        for i, tgt in enumerate(self.all_target_subregions):
            if tgt in data:
                raw_values[i] = data[tgt]
        return raw_values

    def compute_all_projection_signatures(self):
        """计算所有脑区的投射指纹并进行PCA"""
        print("\n=== 计算投射指纹 ===")

        # 计算原始指纹
        for region in self.regions:
            sig = self.compute_projection_signature_raw(region)
            self.proj_signatures_raw[region] = sig

        raw_matrix = np.array([self.proj_signatures_raw[r] for r in self.regions])
        print(f"原始投射指纹矩阵: {raw_matrix.shape}")

        # Step 1: 剪枝 - 去掉全零列
        col_sums = np.sum(raw_matrix, axis=0)
        nonzero_cols = col_sums > 0
        n_removed = np.sum(~nonzero_cols)
        pruned_matrix = raw_matrix[:, nonzero_cols]
        print(f"Step 1 剪枝: 去掉 {n_removed} 个零值目标脑区，剩余 {pruned_matrix.shape[1]} 维")

        # Step 2: Log1p变换
        log_matrix = np.log1p(pruned_matrix)
        print(f"Step 2 Log1p变换完成")

        # Step 3: Z-score标准化（按列）
        scaler = StandardScaler()
        zscore_matrix = scaler.fit_transform(log_matrix)
        print(f"Step 3 Z-score标准化完成")

        # Step 4: PCA
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca_matrix = pca.fit_transform(zscore_matrix)
        self.proj_pca = pca
        print(f"Step 4 PCA: {pruned_matrix.shape[1]} 维 -> {pca_matrix.shape[1]} 维 "
              f"(方差解释: {pca.explained_variance_ratio_.sum()*100:.2f}%)")

        # 保存维度信息
        self.proj_dim_info = {
            'original_dim': len(self.all_target_subregions),
            'after_pruning': pruned_matrix.shape[1],
            'after_pca': pca_matrix.shape[1],
            'variance_explained': pca.explained_variance_ratio_.sum(),
            'n_removed_targets': n_removed
        }

        # 保存PCA后的指纹
        for i, region in enumerate(self.regions):
            self.proj_signatures[region] = pca_matrix[i]

        print(f"完成 {len(self.proj_signatures)} 个投射指纹计算")

    # ==================== 4. 相似度和距离计算 ====================

    def compute_distance_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """计算三种距离矩阵（使用PCA后的指纹）"""
        n = len(self.regions)
        mol_dist = np.zeros((n, n))
        morph_dist = np.zeros((n, n))
        proj_dist = np.zeros((n, n))

        for i, region_a in enumerate(self.regions):
            for j, region_b in enumerate(self.regions):
                if i == j:
                    continue

                # 分子距离 (cosine distance)
                try:
                    mol_a = self.mol_signatures[region_a]
                    mol_b = self.mol_signatures[region_b]
                    mol_dist[i, j] = cosine(mol_a, mol_b)
                except:
                    mol_dist[i, j] = np.nan

                # 形态距离 (Euclidean distance on PCA features)
                try:
                    morph_a = self.morph_signatures[region_a]
                    morph_b = self.morph_signatures[region_b]
                    morph_dist[i, j] = euclidean(morph_a, morph_b)
                except:
                    morph_dist[i, j] = np.nan

                # 投射距离 (cosine distance)
                try:
                    proj_a = self.proj_signatures[region_a]
                    proj_b = self.proj_signatures[region_b]
                    proj_dist[i, j] = cosine(proj_a, proj_b)
                except:
                    proj_dist[i, j] = np.nan

        mol_dist_df = pd.DataFrame(mol_dist, index=self.regions, columns=self.regions)
        morph_dist_df = pd.DataFrame(morph_dist, index=self.regions, columns=self.regions)
        proj_dist_df = pd.DataFrame(proj_dist, index=self.regions, columns=self.regions)

        return mol_dist_df, morph_dist_df, proj_dist_df

    def compute_mismatch_matrices(self, mol_dist_df: pd.DataFrame,
                                  morph_dist_df: pd.DataFrame,
                                  proj_dist_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """计算mismatch矩阵"""
        def minmax_normalize(df):
            values = df.values
            valid = ~np.isnan(values)
            if valid.sum() == 0:
                return df
            vmin = values[valid].min()
            vmax = values[valid].max()
            if vmax - vmin < 1e-9:
                return pd.DataFrame(np.zeros_like(values), index=df.index, columns=df.columns)
            normalized = (values - vmin) / (vmax - vmin)
            return pd.DataFrame(normalized, index=df.index, columns=df.columns)

        mol_norm = minmax_normalize(mol_dist_df)
        morph_norm = minmax_normalize(morph_dist_df)
        proj_norm = minmax_normalize(proj_dist_df)

        mol_morph_mismatch = np.abs(mol_norm - morph_norm)
        mol_proj_mismatch = np.abs(mol_norm - proj_norm)

        return mol_morph_mismatch, mol_proj_mismatch

    # ==================== 5. 数据保存 ====================

    def save_fingerprints_to_csv(self, output_dir: str = "."):
        """将三种指纹保存为CSV文件"""
        os.makedirs(output_dir, exist_ok=True)

        # 分子指纹（PCA后）
        mol_cols = [f'mol_PC{i+1}' for i in range(self.mol_dim_info['after_pca'])]
        mol_df = pd.DataFrame.from_dict(self.mol_signatures, orient='index', columns=mol_cols)
        mol_df.index.name = 'region'
        mol_df.to_csv(f"{output_dir}/molecular_fingerprints_pca.csv")
        print(f"\n分子指纹已保存: {output_dir}/molecular_fingerprints_pca.csv")

        # 形态指纹（PCA后）
        morph_cols = [f'morph_PC{i+1}' for i in range(self.morph_dim_info['after_pca'])]
        morph_df = pd.DataFrame.from_dict(self.morph_signatures, orient='index', columns=morph_cols)
        morph_df.index.name = 'region'
        morph_df.to_csv(f"{output_dir}/morphology_fingerprints_pca.csv")
        print(f"形态指纹已保存: {output_dir}/morphology_fingerprints_pca.csv")

        # 投射指纹（PCA后）
        proj_cols = [f'proj_PC{i+1}' for i in range(self.proj_dim_info['after_pca'])]
        proj_df = pd.DataFrame.from_dict(self.proj_signatures, orient='index', columns=proj_cols)
        proj_df.index.name = 'region'
        proj_df.to_csv(f"{output_dir}/projection_fingerprints_pca.csv")
        print(f"投射指纹已保存: {output_dir}/projection_fingerprints_pca.csv")

        # 保存维度信息
        dim_info = {
            'molecular': self.mol_dim_info,
            'morphology': self.morph_dim_info,
            'projection': self.proj_dim_info
        }
        dim_df = pd.DataFrame(dim_info).T
        dim_df.to_csv(f"{output_dir}/dimension_info.csv")
        print(f"维度信息已保存: {output_dir}/dimension_info.csv")

    # ==================== 6. 可视化 ====================

    def select_top_regions_by_neuron_count(self, n: int = 20) -> List[str]:
        """根据神经元数量选择top N个脑区"""
        query = """
        MATCH (r:Region)
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
        WITH r, COUNT(DISTINCT n) AS neuron_count
        WHERE neuron_count > 0
        RETURN r.acronym AS region, neuron_count
        ORDER BY neuron_count DESC
        LIMIT $n
        """
        with self.driver.session() as session:
            result = session.run(query, n=n)
            top_regions = [record['region'] for record in result]
        print(f"\n选择了神经元数量最多的 {len(top_regions)} 个脑区:")
        print(top_regions)
        return top_regions

    def visualize_matrices(self, top_regions: List[str], output_dir: str = "."):
        """
        可视化矩阵，保存两种版本：普通热力图和带聚类的热力图
        """
        os.makedirs(output_dir, exist_ok=True)

        valid_regions = [r for r in top_regions if r in self.regions]
        print(f"\n开始可视化 {len(valid_regions)} 个脑区的矩阵...")

        # 计算距离矩阵（只针对top regions）
        n = len(valid_regions)
        mol_dist = np.zeros((n, n))
        morph_dist = np.zeros((n, n))
        proj_dist = np.zeros((n, n))

        for i, region_a in enumerate(valid_regions):
            for j, region_b in enumerate(valid_regions):
                if i == j:
                    continue
                try:
                    mol_dist[i, j] = cosine(self.mol_signatures[region_a],
                                           self.mol_signatures[region_b])
                except:
                    mol_dist[i, j] = np.nan
                try:
                    morph_dist[i, j] = euclidean(self.morph_signatures[region_a],
                                                 self.morph_signatures[region_b])
                except:
                    morph_dist[i, j] = np.nan
                try:
                    proj_dist[i, j] = cosine(self.proj_signatures[region_a],
                                            self.proj_signatures[region_b])
                except:
                    proj_dist[i, j] = np.nan

        mol_dist_df = pd.DataFrame(mol_dist, index=valid_regions, columns=valid_regions)
        morph_dist_df = pd.DataFrame(morph_dist, index=valid_regions, columns=valid_regions)
        proj_dist_df = pd.DataFrame(proj_dist, index=valid_regions, columns=valid_regions)

        # Min-Max归一化
        def minmax_normalize_df(df):
            values = df.values
            valid = ~np.isnan(values)
            if valid.sum() == 0:
                return df
            vmin, vmax = values[valid].min(), values[valid].max()
            if vmax - vmin < 1e-9:
                return pd.DataFrame(np.zeros_like(values), index=df.index, columns=df.columns)
            normalized = (values - vmin) / (vmax - vmin)
            return pd.DataFrame(normalized, index=df.index, columns=df.columns)

        mol_dist_norm = minmax_normalize_df(mol_dist_df)
        morph_dist_norm = minmax_normalize_df(morph_dist_df)
        proj_dist_norm = minmax_normalize_df(proj_dist_df)

        mol_sim = 1 - mol_dist_norm
        morph_sim = 1 - morph_dist_norm
        proj_sim = 1 - proj_dist_norm

        mol_morph_mismatch, mol_proj_mismatch = self.compute_mismatch_matrices(
            mol_dist_df, morph_dist_df, proj_dist_df
        )

        # ==================== 普通热力图 ====================
        print("\n保存普通热力图...")
        self._save_regular_heatmaps(mol_sim, morph_sim, proj_sim,
                                     mol_morph_mismatch, mol_proj_mismatch, output_dir)

        # ==================== 聚类热力图 ====================
        print("\n保存聚类热力图...")
        self._save_clustered_heatmaps(mol_sim, morph_sim, proj_sim,
                                       mol_morph_mismatch, mol_proj_mismatch, output_dir)

        # 打印top mismatch pairs
        top_pairs = self._print_top_mismatch_pairs(mol_morph_mismatch, mol_proj_mismatch,
                                                   valid_regions, n=10)

        return top_pairs, mol_morph_mismatch, mol_proj_mismatch

    def _save_regular_heatmaps(self, mol_sim, morph_sim, proj_sim,
                                mol_morph_mismatch, mol_proj_mismatch, output_dir):
        """保存普通热力图（无聚类）"""

        # 组合图
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle('Brain Region Similarity and Mismatch Analysis\n(PCA-transformed, 64-dim morphology)',
                     fontsize=16, fontweight='bold', y=0.98)

        sns.heatmap(mol_sim, ax=axes[0, 0], cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'})
        axes[0, 0].set_title('Molecular Similarity', fontsize=14, fontweight='bold')

        sns.heatmap(morph_sim, ax=axes[0, 1], cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'})
        axes[0, 1].set_title('Morphology Similarity (64D)', fontsize=14, fontweight='bold')

        sns.heatmap(proj_sim, ax=axes[0, 2], cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'})
        axes[0, 2].set_title('Projection Similarity', fontsize=14, fontweight='bold')

        sns.heatmap(mol_morph_mismatch, ax=axes[1, 0], cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Mismatch'})
        axes[1, 0].set_title('Mol-Morph Mismatch', fontsize=14, fontweight='bold')

        sns.heatmap(mol_proj_mismatch, ax=axes[1, 1], cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Mismatch'})
        axes[1, 1].set_title('Mol-Proj Mismatch', fontsize=14, fontweight='bold')

        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_matrices_combined.png", dpi=1200, bbox_inches='tight')
        plt.close()

        # 单独保存每个矩阵
        matrices = [
            (mol_sim, 'Molecule Similarity', '1_molecule_similarity.png', 'Similarity'),
            (morph_sim, 'Morphology Similarity', '2_morphology_similarity.png', 'Similarity'),
            (proj_sim, 'Projection Similarity', '3_projection_similarity.png', 'Similarity'),
            (mol_morph_mismatch, 'Mol-Morph Mismatch', '4_mol_morph_mismatch.png', 'Mismatch'),
            (mol_proj_mismatch, 'Mol-Proj Mismatch', '5_mol_proj_mismatch.png', 'Mismatch'),
        ]

        for matrix, title, filename, label in matrices:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                        square=True, cbar_kws={'label': label},
                        xticklabels=True, yticklabels=True)
            # ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Region', fontsize=14, fontweight='bold')
            ax.set_ylabel('Region', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{filename}", dpi=1200, bbox_inches='tight')
            plt.close()

        print("✓ 普通热力图已保存")

    def _save_clustered_heatmaps(self, mol_sim, morph_sim, proj_sim,
                                  mol_morph_mismatch, mol_proj_mismatch, output_dir):
        """保存聚类热力图（clustermap）"""

        cluster_dir = f"{output_dir}/clustered"
        os.makedirs(cluster_dir, exist_ok=True)

        matrices = [
            (mol_sim, 'Molecular Similarity (Clustered)', '1_molecular_similarity_clustered.png'),
            (morph_sim, 'Morphology Similarity 64D (Clustered)', '2_morphology_similarity_clustered.png'),
            (proj_sim, 'Projection Similarity (Clustered)', '3_projection_similarity_clustered.png'),
            (mol_morph_mismatch, 'Mol-Morph Mismatch (Clustered)', '4_mol_morph_mismatch_clustered.png'),
            (mol_proj_mismatch, 'Mol-Proj Mismatch (Clustered)', '5_mol_proj_mismatch_clustered.png'),
        ]

        for matrix, title, filename in matrices:
            try:
                # 处理NaN值（clustermap不支持NaN）
                matrix_filled = matrix.fillna(0.5)

                g = sns.clustermap(matrix_filled, cmap='RdYlBu_r', vmin=0, vmax=1,
                                   figsize=(12, 10), dendrogram_ratio=0.15,
                                   cbar_pos=(0.02, 0.8, 0.03, 0.15))
                g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
                g.savefig(f"{cluster_dir}/{filename}", dpi=1200, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  警告: 无法生成 {filename} 的聚类图: {e}")

        print(f"✓ 聚类热力图已保存到 {cluster_dir}/")

    def _print_top_mismatch_pairs(self, mol_morph_mismatch: pd.DataFrame,
                                  mol_proj_mismatch: pd.DataFrame,
                                  regions: List[str], n: int = 10):
        """打印top N的mismatch脑区对"""
        print("\n" + "=" * 80)
        print("Top Mismatch Region Pairs (PCA-transformed, 64D morphology)")
        print("=" * 80)

        # Molecular-Morphology Mismatch
        print(f"\n【分子-形态 Mismatch Top {n}】")
        mm_values = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                val = mol_morph_mismatch.iloc[i, j]
                if not np.isnan(val):
                    mm_values.append((regions[i], regions[j], val))
        mm_values.sort(key=lambda x: x[2], reverse=True)
        for rank, (r1, r2, val) in enumerate(mm_values[:n], 1):
            print(f"{rank:2d}. {r1:10s} <-> {r2:10s}  |  Mismatch = {val:.4f}")

        # Molecular-Projection Mismatch
        print(f"\n【分子-投射 Mismatch Top {n}】")
        mp_values = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                val = mol_proj_mismatch.iloc[i, j]
                if not np.isnan(val):
                    mp_values.append((regions[i], regions[j], val))
        mp_values.sort(key=lambda x: x[2], reverse=True)
        for rank, (r1, r2, val) in enumerate(mp_values[:n], 1):
            print(f"{rank:2d}. {r1:10s} <-> {r2:10s}  |  Mismatch = {val:.4f}")

        return {'mol_morph': mm_values[:n], 'mol_proj': mp_values[:n]}

    def visualize_specific_pairs(self, mol_morph_pairs=None, mol_proj_pairs=None,
                                  output_dir=".", mol_morph_mismatch_df=None,
                                  mol_proj_mismatch_df=None):
        """手动可视化指定的脑区对"""
        os.makedirs(output_dir, exist_ok=True)

        if mol_morph_pairs:
            print("\nManual Molecular-Morphology comparisons:")
            for rank, pair in enumerate(mol_morph_pairs, 1):
                if len(pair) == 3:
                    r1, r2, mismatch = pair
                else:
                    r1, r2 = pair
                    mismatch = np.nan
                    if mol_morph_mismatch_df is not None:
                        if r1 in mol_morph_mismatch_df.index and r2 in mol_morph_mismatch_df.columns:
                            mismatch = mol_morph_mismatch_df.loc[r1, r2]
                        elif r2 in mol_morph_mismatch_df.index and r1 in mol_morph_mismatch_df.columns:
                            mismatch = mol_morph_mismatch_df.loc[r2, r1]
                self._plot_mol_morph_comparison(r1, r2, mismatch, rank, output_dir)

        if mol_proj_pairs:
            print("\nManual Molecular-Projection comparisons:")
            for rank, pair in enumerate(mol_proj_pairs, 1):
                if len(pair) == 3:
                    r1, r2, mismatch = pair
                else:
                    r1, r2 = pair
                    mismatch = np.nan
                    if mol_proj_mismatch_df is not None:
                        if r1 in mol_proj_mismatch_df.index and r2 in mol_proj_mismatch_df.columns:
                            mismatch = mol_proj_mismatch_df.loc[r1, r2]
                        elif r2 in mol_proj_mismatch_df.index and r1 in mol_proj_mismatch_df.columns:
                            mismatch = mol_proj_mismatch_df.loc[r2, r1]
                self._plot_mol_proj_comparison(r1, r2, mismatch, rank, output_dir)

    def _plot_mol_morph_comparison(self, region1: str, region2: str,
                                   mismatch: float, rank: int, output_dir: str):
        """绘制分子-形态 Mismatch 的详细对比图"""
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1])

        # === 1. 形态特征对比（使用PCA后的前几个主成分）===
        ax_morph = fig.add_subplot(gs[0])

        morph1 = self.morph_signatures.get(region1, np.zeros(10))
        morph2 = self.morph_signatures.get(region2, np.zeros(10))

        # 显示前10个主成分
        n_show = min(10, len(morph1))
        x = np.arange(n_show)
        width = 0.35

        ax_morph.bar(x - width/2, morph1[:n_show], width, label=region1, color='#E74C3C', alpha=0.8)
        ax_morph.bar(x + width/2, morph2[:n_show], width, label=region2, color='#3498DB', alpha=0.8)
        ax_morph.set_xticks(x)
        ax_morph.set_xticklabels([f'PC{i+1}' for i in range(n_show)], fontsize=9)
        ax_morph.set_xlabel('Morphology Principal Components', fontsize=11)
        ax_morph.set_ylabel('PC Score', fontsize=11)
        ax_morph.set_title(f'Morphology (64D→{self.morph_dim_info["after_pca"]}D PCA)',
                          fontsize=12, fontweight='bold')
        ax_morph.legend()
        ax_morph.grid(axis='y', alpha=0.3)

        # === 2. 分子组成对比 ===
        ax_mol = fig.add_subplot(gs[1])

        mol1 = self.mol_signatures_raw.get(region1, np.zeros(len(self.all_subclasses)))
        mol2 = self.mol_signatures_raw.get(region2, np.zeros(len(self.all_subclasses)))

        top_indices = np.argsort(mol1 + mol2)[-10:][::-1]
        top_subclasses = [self.all_subclasses[i] for i in top_indices]
        top_subclasses_short = [s[:25] + '...' if len(s) > 25 else s for s in top_subclasses]

        y = np.arange(len(top_subclasses))
        width = 0.35

        ax_mol.barh(y - width/2, mol1[top_indices], width, label=region1, color='#E74C3C', alpha=0.8)
        ax_mol.barh(y + width/2, mol2[top_indices], width, label=region2, color='#3498DB', alpha=0.8)
        ax_mol.set_yticks(y)
        ax_mol.set_yticklabels(top_subclasses_short, fontsize=9)
        ax_mol.set_xlabel('Cell Type %', fontsize=11)
        ax_mol.set_title('Top 10 Cell Types', fontsize=12, fontweight='bold')
        ax_mol.legend()
        ax_mol.grid(axis='x', alpha=0.3)

        # === 3. 说明文本 ===
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')

        info_text = f"""
Mol-Morph Mismatch Analysis
{'=' * 35}

Region Pair: {region1} ↔ {region2}
Rank: #{rank}

Mismatch Score: {mismatch:.4f}
{'=' * 35}

Dimension Info:
  Molecular: {self.mol_dim_info.get('original_dim', 'N/A')} → {self.mol_dim_info.get('after_pca', 'N/A')}
  Morphology: 64 → {self.morph_dim_info.get('after_pca', 'N/A')}
  
Variance Explained:
  Molecular: {self.mol_dim_info.get('variance_explained', 0)*100:.1f}%
  Morphology: {self.morph_dim_info.get('variance_explained', 0)*100:.1f}%
        """
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f'Mol-Morph Mismatch #{rank}: {region1} vs {region2}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detail_mol_morph_{rank}_{region1}_vs_{region2}.png",
                    dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: detail_mol_morph_{rank}_{region1}_vs_{region2}.png")

    def _plot_mol_proj_comparison(self, region1: str, region2: str,
                                  mismatch: float, rank: int, output_dir: str):
        """绘制分子-投射 Mismatch 的详细对比图"""
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

        # === 1. 投射分布对比 ===
        ax_proj = fig.add_subplot(gs[0])

        proj1 = self.proj_signatures_raw.get(region1, np.zeros(len(self.all_target_subregions)))
        proj2 = self.proj_signatures_raw.get(region2, np.zeros(len(self.all_target_subregions)))

        top_indices = np.argsort(proj1 + proj2)[-15:][::-1]
        top_targets = [self.all_target_subregions[i] for i in top_indices]

        y = np.arange(len(top_targets))
        width = 0.35

        # 使用log1p进行显示
        proj1_log = np.log1p(proj1[top_indices])
        proj2_log = np.log1p(proj2[top_indices])

        ax_proj.barh(y - width/2, proj1_log, width, label=region1, color='#E74C3C', alpha=0.8)
        ax_proj.barh(y + width/2, proj2_log, width, label=region2, color='#3498DB', alpha=0.8)
        ax_proj.set_yticks(y)
        ax_proj.set_yticklabels(top_targets, fontsize=9)
        ax_proj.set_xlabel('Projection Strength (log1p)', fontsize=11)
        ax_proj.set_title('Top 15 Projection Targets', fontsize=12, fontweight='bold')
        ax_proj.legend()
        ax_proj.grid(axis='x', alpha=0.3)

        # === 2. 分子组成对比 ===
        ax_mol = fig.add_subplot(gs[1])

        mol1 = self.mol_signatures_raw.get(region1, np.zeros(len(self.all_subclasses)))
        mol2 = self.mol_signatures_raw.get(region2, np.zeros(len(self.all_subclasses)))

        top_indices = np.argsort(mol1 + mol2)[-10:][::-1]
        top_subclasses = [self.all_subclasses[i] for i in top_indices]
        top_subclasses_short = [s[:20] + '...' if len(s) > 20 else s for s in top_subclasses]

        y = np.arange(len(top_subclasses))
        width = 0.35

        ax_mol.barh(y - width/2, mol1[top_indices], width, label=region1, color='#E74C3C', alpha=0.8)
        ax_mol.barh(y + width/2, mol2[top_indices], width, label=region2, color='#3498DB', alpha=0.8)
        ax_mol.set_yticks(y)
        ax_mol.set_yticklabels(top_subclasses_short, fontsize=9)
        ax_mol.set_xlabel('Cell Type %', fontsize=11)
        ax_mol.set_title('Top 10 Cell Types', fontsize=12, fontweight='bold')
        ax_mol.legend()
        ax_mol.grid(axis='x', alpha=0.3)

        # === 3. 说明文本 ===
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')

        info_text = f"""
Mol-Proj Mismatch Analysis
{'=' * 35}

Region Pair: {region1} ↔ {region2}
Rank: #{rank}

Mismatch Score: {mismatch:.4f}
{'=' * 35}

Dimension Info:
  Molecular: {self.mol_dim_info.get('original_dim', 'N/A')} → {self.mol_dim_info.get('after_pca', 'N/A')}
  Projection: {self.proj_dim_info.get('original_dim', 'N/A')} → {self.proj_dim_info.get('after_pca', 'N/A')}
  
Variance Explained:
  Molecular: {self.mol_dim_info.get('variance_explained', 0)*100:.1f}%
  Projection: {self.proj_dim_info.get('variance_explained', 0)*100:.1f}%
        """
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle(f'Mol-Proj Mismatch #{rank}: {region1} vs {region2}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detail_mol_proj_{rank}_{region1}_vs_{region2}.png",
                    dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: detail_mol_proj_{rank}_{region1}_vs_{region2}.png")

    # ==================== 7. 主流程 ====================

    def run_full_analysis(self, output_dir: str = "./fingerprint_results_pca_64d",
                          top_n_regions: int = 30):
        """运行完整分析流程"""
        print("\n" + "=" * 80)
        print("脑区指纹分析 - PCA版本（64维形态特征）")
        print(f"PCA方差解释阈值: {self.pca_variance_threshold*100:.1f}%")
        print("=" * 80)

        # Step 1: 获取全局维度
        self.get_all_subclasses()
        self.get_all_target_subregions()

        # Step 2: 计算三种指纹（含PCA）
        self.compute_all_molecular_signatures()
        self.compute_all_morphology_signatures()
        self.compute_all_projection_signatures()

        # 打印维度信息汇总
        print("\n" + "=" * 60)
        print("维度信息汇总")
        print("=" * 60)
        print(f"【分子向量】")
        print(f"  原始维度: {self.mol_dim_info['original_dim']}")
        print(f"  剪枝后: {self.mol_dim_info['after_pruning']} (去掉 {self.mol_dim_info['n_removed_subclasses']} 个零值)")
        print(f"  PCA后: {self.mol_dim_info['after_pca']} (方差解释: {self.mol_dim_info['variance_explained']*100:.2f}%)")

        print(f"【形态向量】")
        print(f"  原始维度: {self.morph_dim_info['original_dim']} (32 axonal + 32 dendritic)")
        print(f"  剪枝后: {self.morph_dim_info['after_pruning']}")
        print(f"  PCA后: {self.morph_dim_info['after_pca']} (方差解释: {self.morph_dim_info['variance_explained']*100:.2f}%)")

        print(f"【投射向量】")
        print(f"  原始维度: {self.proj_dim_info['original_dim']}")
        print(f"  剪枝后: {self.proj_dim_info['after_pruning']} (去掉 {self.proj_dim_info['n_removed_targets']} 个零值)")
        print(f"  PCA后: {self.proj_dim_info['after_pca']} (方差解释: {self.proj_dim_info['variance_explained']*100:.2f}%)")

        # Step 3: 保存指纹到CSV
        self.save_fingerprints_to_csv(output_dir)

        # Step 4: 选择top N脑区
        top_regions = self.select_top_regions_by_neuron_count(top_n_regions)

        # Step 5: 可视化矩阵
        top_pairs, mol_morph_mismatch, mol_proj_mismatch = self.visualize_matrices(
            top_regions, output_dir
        )

        # Step 6: 绘制详细对比图
        manual_mol_morph = [("CA3", "MOs"), ("CA3", "ACAd"), ("CA3", "SUB")]
        manual_mol_proj = [("CA3", "MOs"), ("CA3", "ACAd"), ("CA3", "SUB")]

        self.visualize_specific_pairs(
            mol_morph_pairs=manual_mol_morph,
            mol_proj_pairs=manual_mol_proj,
            output_dir=output_dir,
            mol_morph_mismatch_df=mol_morph_mismatch,
            mol_proj_mismatch_df=mol_proj_mismatch
        )

        print("\n" + "=" * 80)
        print("分析完成！")
        print(f"结果保存在: {output_dir}")
        print("=" * 80 + "\n")


# ==================== 主程序 ====================

def main():
    """主程序入口"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"

    OUTPUT_DIR = "./fingerprint_results_v6_pca_64d"
    TOP_N_REGIONS = 30
    PCA_VARIANCE = 0.95

    print("\n" + "=" * 80)
    print("脑区指纹计算与可视化 - PCA版本（64维形态特征）")
    print("=" * 80)
    print(f"\nNeo4j URI: {NEO4J_URI}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"选择前 {TOP_N_REGIONS} 个脑区进行可视化")
    print(f"PCA方差解释阈值: {PCA_VARIANCE*100:.1f}%")
    print(f"形态特征: 64维 (32 axonal + 32 dendritic)\n")

    with BrainRegionFingerprintsPCA(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                                     pca_variance_threshold=PCA_VARIANCE) as analyzer:
        analyzer.run_full_analysis(
            output_dir=OUTPUT_DIR,
            top_n_regions=TOP_N_REGIONS
        )


if __name__ == "__main__":
    main()