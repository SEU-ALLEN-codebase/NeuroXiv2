import json
import logging
import os
import time
import argparse
import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from neo4j_exec import Neo4jExec

# Intent routing and provenance (CC_SPEC_MS additions)
from intent_router import IntentRouter, IntentType, get_budget_for_intent, get_smalltalk_response, BudgetLimits
from provenance import ProvenanceLogger, create_provenance_logger, EventType
from evidence_buffer import EvidenceBuffer
from adaptive_planner import AdaptivePlanner, AnalysisDepth, AnalysisState
from aipom_cot_true_agent_v2 import (
    RealSchemaCache,
    StatisticalTools,
    AgentPhase,
    AgentState,
    ReasoningStep
)

# 导入新组件
from intelligent_entity_recognition import (
    IntelligentEntityRecognizer,
    EntityClusteringEngine
)
from schema_path_planner import DynamicSchemaPathPlanner
from structured_reflection import StructuredReflector

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

logger = logging.getLogger(__name__)


# ==================== Enhanced Agent State ====================

@dataclass
class EnhancedAgentState(AgentState):
    """扩展的Agent状态"""

    # 新增字段
    entity_matches: List = field(default_factory=list)  # EntityMatch列表
    entity_clusters: List = field(default_factory=list)  # EntityCluster列表
    structured_reflections: List = field(default_factory=list)  # StructuredReflection列表
    schema_paths_used: List = field(default_factory=list)  # 使用的schema路径

class RealFingerprintAnalyzer:
    """
    Multi-modal fingerprint analysis adapted to REAL schema

    Key changes from V8:
    - Molecular: Use Cluster nodes and HAS_CLUSTER relationships
    - Morphological: Aggregate from Neuron nodes via LOCATE_AT
    - Projection: Use PROJECT_TO (unchanged, but verify properties)
    """

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema
        self._cluster_cache = None
        self._target_cache = None

    def compute_region_fingerprint(self, region: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute tri-modal fingerprint for a region

        Returns:
            {
                'molecular': np.ndarray,    # Cluster composition
                'morphological': np.ndarray, # Aggregated neuron features
                'projection': np.ndarray     # Target distribution
            }
        """
        fingerprint = {}

        # Molecular fingerprint
        mol_fp = self.compute_molecular_fingerprint(region)
        if mol_fp is not None:
            fingerprint['molecular'] = mol_fp

        # Morphological fingerprint
        mor_fp = self.compute_morphological_fingerprint(region)
        if mor_fp is not None:
            fingerprint['morphological'] = mor_fp

        # Projection fingerprint
        proj_fp = self.compute_projection_fingerprint(region)
        if proj_fp is not None:
            fingerprint['projection'] = proj_fp

        return fingerprint if len(fingerprint) > 0 else None

    def compute_molecular_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """
        计算单个脑区的分子指纹 (Figure 4方法)

        🎯 分子指纹 = Subclass组成百分比

        使用关系: Region -[HAS_SUBCLASS]-> Subclass
        """
        query = """
        MATCH (r:Region {acronym: $acronym})-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN
          sc.name AS subclass_name,
          hs.pct_cells AS pct_cells
        ORDER BY sc.name
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            logger.warning(f"No molecular data for {region}")
            return None

        # 构建字典
        data = {}
        for row in result['data']:
            subclass_name = row.get('subclass_name')
            pct_cells = row.get('pct_cells')
            if subclass_name and pct_cells is not None:
                data[subclass_name] = float(pct_cells)

        if not data:
            return None

        # 获取全局subclass列表
        all_subclasses = self._get_all_subclasses()  # 这个方法返回所有subclass names

        if not all_subclasses:
            logger.error("No global subclasses found")
            return None

        # 构建固定维度的向量
        signature = np.zeros(len(all_subclasses), dtype=float)
        for i, subclass in enumerate(all_subclasses):
            if subclass in data:
                signature[i] = data[subclass]

        # 🔍 调试：检查是否是零向量
        nonzero_count = np.count_nonzero(signature)
        total_pct = np.sum(signature)

        if nonzero_count == 0:
            logger.warning(f"{region}: molecular fingerprint is all zeros!")
            return None

        logger.debug(f"{region}: molecular FP - {nonzero_count}/{len(signature)} nonzero, sum={total_pct:.2f}")

        return signature

    def compute_morphological_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """
        计算单个脑区的形态指纹 (对齐Figure 4)

        🔧 关键改进:
        1. 从Region节点的聚合属性读取（不是实时聚合）
        2. 返回8维向量（不是6维）
        3. 匹配Figure 4的特征顺序
        """
        query = """
        MATCH (r:Region {acronym: $acronym})
        RETURN
          r.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
          r.axonal_length AS axonal_length,
          r.axonal_branches AS axonal_branches,
          r.axonal_maximum_branch_order AS axonal_max_branch_order,
          r.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
          r.dendritic_length AS dendritic_length,
          r.dendritic_branches AS dendritic_branches,
          r.dendritic_maximum_branch_order AS dendritic_max_branch_order
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data'] or not result['data'][0]:
            return None

        record = result['data'][0]

        # 按照固定顺序提取特征值
        features = [
            'axonal_bifurcation_remote_angle',
            'axonal_length',
            'axonal_branches',
            'axonal_max_branch_order',
            'dendritic_bifurcation_remote_angle',
            'dendritic_length',
            'dendritic_branches',
            'dendritic_max_branch_order'
        ]

        signature = np.array([
            record.get(feat) if record.get(feat) is not None else np.nan
            for feat in features
        ], dtype=float)

        return signature

    def compute_projection_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """
        计算投射指纹 (对齐Ground Truth - 使用Neuron->Subregion)

        🔧 关键修复：
        1. 从Neuron级别聚合
        2. 投射目标是Subregion（不是Region）
        3. 聚合三种location关系
        """
        query = """
        MATCH (r:Region {acronym: $acronym})

        // 找属于这个区域的神经元
        OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT]->(r)
        OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r)
        OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r)
        WITH r, (COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) + COLLECT(DISTINCT n3)) AS ns
        UNWIND ns AS n
        WITH DISTINCT n
        WHERE n IS NOT NULL

        // 找这些神经元的投射到Subregion
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0

        WITH t.acronym AS tgt_subregion,
             SUM(p.weight) AS total_weight_to_tgt
        RETURN
          tgt_subregion,
          total_weight_to_tgt
        ORDER BY total_weight_to_tgt DESC
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            logger.warning(f"No projection data for {region}")
            return None

        # 获取所有Subregion targets
        all_targets = self._get_all_targets()

        # 构建原始权重向量
        target_dict = {row['tgt_subregion']: row['total_weight_to_tgt']
                       for row in result['data']}

        raw_values = np.array([target_dict.get(t, 0.0) for t in all_targets])

        # Log稳定化（对齐Ground Truth）
        log_values = np.log10(1 + raw_values)

        # 归一化成概率分布
        total = log_values.sum()
        if total > 0:
            signature = log_values / (total + 1e-9)
        else:
            signature = log_values

        return signature

    def compute_similarity(self, fp1: np.ndarray, fp2: np.ndarray,
                          metric: str = 'cosine') -> float:
        """Compute similarity between fingerprints"""
        if metric == 'cosine':
            norm1, norm2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(fp1, fp2) / (norm1 * norm2))
        elif metric == 'correlation':
            if len(fp1) < 2:
                return 0.0
            r, _ = stats.pearsonr(fp1, fp2)
            return float(r)
        else:
            return 0.0

    def compute_mismatch_index(self, region1: str, region2: str) -> Optional[Dict[str, float]]:
        """
        Compute cross-modal mismatch (Figure 4 metric)

        MM_GM = |sim_molecular - sim_morphological|
        MM_GP = |sim_molecular - sim_projection|
        """
        fp1 = self.compute_region_fingerprint(region1)
        fp2 = self.compute_region_fingerprint(region2)

        if fp1 is None or fp2 is None:
            return None

        sim_mol = self.compute_similarity(fp1['molecular'], fp2['molecular'])
        sim_mor = self.compute_similarity(fp1['morphological'], fp2['morphological'])
        sim_proj = self.compute_similarity(fp1['projection'], fp2['projection'])

        return {
            'sim_molecular': sim_mol,
            'sim_morphological': sim_mor,
            'sim_projection': sim_proj,
            'mismatch_GM': abs(sim_mol - sim_mor),
            'mismatch_GP': abs(sim_mol - sim_proj),
            'mismatch_MP': abs(sim_mor - sim_proj)
        }

    def _get_all_subclasses(self) -> List[str]:
        """
        获取所有subclass names (用于分子指纹)

        🔧 注意: 这里应该查询Subclass，不是Cluster
        """
        if self._cluster_cache is not None:
            return self._cluster_cache

        query = """
        MATCH (sc:Subclass)
        RETURN DISTINCT sc.name AS name
        ORDER BY name
        """

        result = self.db.run(query)

        if result['success'] and result['data']:
            self._cluster_cache = [row['name'] for row in result['data']]
            logger.info(f"Found {len(self._cluster_cache)} subclasses for molecular fingerprint")
        else:
            self._cluster_cache = []
            logger.error("No subclasses found in database!")

        return self._cluster_cache

    def _get_all_targets(self) -> List[str]:
        """
        获取所有投射目标Subregion (对齐Ground Truth)

        🔧 修复：从Subregion获取，不是Region
        """
        if self._target_cache is not None:
            return self._target_cache

        query = """
        MATCH ()-[:PROJECT_TO]->(t:Subregion)
        WHERE t.acronym IS NOT NULL
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        LIMIT 500
        """

        result = self.db.run(query)

        if result['success'] and result['data']:
            self._target_cache = [row['target'] for row in result['data']]
            logger.info(f"Found {len(self._target_cache)} Subregion projection targets")
        else:
            self._target_cache = []
            logger.error("No Subregion targets found!")

        return self._target_cache

    def get_region_fingerprint(self, region: str) -> Dict:
        """
        获取单个region的完整fingerprint

        🆕 新增方法 - 支持高性能版本的批量计算

        Args:
            region: 脑区acronym

        Returns:
            {
                'molecular': [array],
                'morphological': [array],
                'projection': [array]
            }
        """
        try:
            # 计算三种fingerprint
            molecular = self.compute_molecular_fingerprint(region)
            morphological = self.compute_morphological_fingerprint(region)
            projection = self.compute_projection_fingerprint(region)

            # 验证
            if molecular is None or morphological is None or projection is None:
                return None

            # 转换为list (确保JSON可序列化)
            return {
                'molecular': molecular.tolist() if hasattr(molecular, 'tolist') else list(molecular),
                'morphological': morphological.tolist() if hasattr(morphological, 'tolist') else list(morphological),
                'projection': projection.tolist() if hasattr(projection, 'tolist') else list(projection)
            }

        except Exception as e:
            logger.error(f"Failed to get fingerprint for {region}: {e}")
            return None

    def standardize_morphology_globally(self, regions: List[str]):
        """
        全局Z-score标准化形态指纹（对齐Ground Truth方法）

        🎯 关键：在计算mismatch前，对所有regions的形态数据做一次性全局标准化

        Args:
            regions: 需要标准化的region列表
        """
        logger.info("   Performing global morphology standardization...")

        # 收集所有regions的形态指纹
        all_morph = []
        valid_regions = []

        for region in regions:
            morph = self.compute_morphological_fingerprint(region)
            if morph is not None:
                all_morph.append(morph)
                valid_regions.append(region)

        if len(all_morph) < 2:
            logger.warning("   Insufficient morphology data for standardization")
            return

        all_morph = np.array(all_morph)  # (N_regions, 8)

        logger.info(f"      Morphology array shape: {all_morph.shape}")

        # 处理dendritic特征的0值 (索引4-7)
        dendritic_indices = [4, 5, 6, 7]
        for i in dendritic_indices:
            col = all_morph[:, i].copy()
            zero_mask = np.abs(col) < 1e-6
            n_zeros = zero_mask.sum()
            if n_zeros > 0:
                logger.debug(f"      Dendritic feature {i}: excluding {n_zeros}/{len(col)} zeros")
                col[zero_mask] = np.nan
                all_morph[:, i] = col

        # 对每个特征维度进行z-score
        from scipy.stats import zscore
        for i in range(all_morph.shape[1]):
            col = all_morph[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                col[valid] = zscore(col[valid])
                all_morph[:, i] = col

        # 缓存结果
        self._morph_cache = {}
        for idx, region in enumerate(valid_regions):
            self._morph_cache[region] = all_morph[idx]

        logger.info(f"      ✓ Global standardization complete for {len(valid_regions)} regions")


class Figure4PlottingTool:
    """
    Figure 4绘图工具

    封装了signaturev4.py的核心绘图功能，供Agent调用
    """

    def __init__(self, output_dir: str = "./figure4_results"):
        """
        初始化绘图工具

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 设置绘图样式
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        logger.info(f"Figure 4 plotting tool initialized. Output: {self.output_dir}")

    def plot_from_agent_data(self,
                             agent_result: Dict,
                             fingerprint_data: Optional[Dict] = None) -> Dict[str, str]:
        """
        从Agent结果生成所有Figure 4图表

        Args:
            agent_result: Agent的返回结果（包含mismatch数据）
            fingerprint_data: 可选的fingerprint原始数据

        Returns:
            生成的图片文件路径字典
        """
        logger.info("Starting Figure 4 visualization from agent data...")

        output_files = {}

        # 1. 从Agent结果中提取数据
        extracted = self._extract_data_from_agent_result(agent_result)

        if not extracted:
            logger.error("Failed to extract data from agent result")
            return {}

        regions = extracted['regions']
        mismatch_pairs = extracted['mismatch_pairs']

        logger.info(f"Extracted {len(regions)} regions and {len(mismatch_pairs)} mismatch pairs")

        # 2. 构建矩阵
        matrices = self._build_matrices_from_pairs(regions, mismatch_pairs)

        if not matrices:
            logger.error("Failed to build matrices")
            return {}

        # 3. 绘制similarity矩阵 (3个)
        logger.info("Plotting similarity matrices...")
        similarity_files = self._plot_similarity_matrices(
            matrices['mol_sim'],
            matrices['morph_sim'],
            matrices['proj_sim'],
            regions
        )
        output_files.update(similarity_files)

        # 4. 绘制mismatch矩阵 (2个)
        logger.info("Plotting mismatch matrices...")
        mismatch_files = self._plot_mismatch_matrices(
            matrices['mismatch_GM'],
            matrices['mismatch_GP'],
            regions
        )
        output_files.update(mismatch_files)

        # 5. 识别top pairs
        top_pairs = self._identify_top_pairs(
            matrices['mismatch_GM'],
            matrices['mismatch_GP'],
            regions,
            n=5
        )

        # 6. 绘制top pairs的详细对比 (可选，如果有fingerprint数据)
        if fingerprint_data:
            logger.info("Plotting detailed comparisons for top pairs...")
            detail_files = self._plot_detailed_comparisons(
                top_pairs,
                fingerprint_data,
                regions
            )
            output_files.update(detail_files)

        logger.info(f"✅ Generated {len(output_files)} figures")
        for name, path in output_files.items():
            logger.info(f"   • {name}: {path}")

        return output_files

    def _extract_data_from_agent_result(self, agent_result: Dict) -> Optional[Dict]:
        """
        从Agent结果中提取绘图所需数据 (增强版 - 支持多种数据位置)

        Returns:
            {
                'regions': List[str],
                'mismatch_pairs': List[Dict]
            }
        """
        regions = []
        mismatch_pairs = []

        logger.info("Extracting data from agent result...")

        # 🔍 策略1: 从executed_steps的actual_result提取
        for step in agent_result.get('executed_steps', []):
            purpose = step.get('purpose', '').lower()

            # 尝试获取actual_result
            actual_result = step.get('actual_result')

            if not actual_result:
                # 如果没有actual_result，跳过
                logger.debug(f"Step '{purpose[:40]}' has no actual_result")
                continue

            if not actual_result.get('success'):
                continue

            data = actual_result.get('data', [])

            if not data:
                continue

            logger.debug(f"Step '{purpose[:50]}': {len(data)} rows")

            # 提取regions
            if any(kw in purpose for kw in ['region', 'identify', 'top', 'neuron']):
                for row in data:
                    region = row.get('region') or row.get('acronym') or row.get('region_name')
                    if region and region not in regions:
                        regions.append(region)

                if regions:
                    logger.info(f"  Found {len(regions)} regions from: {purpose[:50]}")

            # 提取mismatch pairs
            if 'mismatch' in purpose:
                # 检查是否包含mismatch字段
                if data and isinstance(data[0], dict):
                    has_mismatch = any(
                        key in data[0]
                        for key in ['mismatch_combined', 'mismatch_GM', 'mismatch_GP']
                    )

                    if has_mismatch:
                        mismatch_pairs.extend(data)
                        logger.info(f"  Found {len(data)} mismatch pairs from: {purpose[:50]}")

        # 🔍 策略2: 从intermediate_data提取（Fallback）
        if not regions or not mismatch_pairs:
            logger.info("Strategy 1 failed, trying intermediate_data...")

            intermediate = agent_result.get('intermediate_data', {})

            for key, data in intermediate.items():
                if not data or not isinstance(data, list):
                    continue

                if not data:
                    continue

                first_row = data[0] if isinstance(data, list) and data else {}

                # 查找regions
                if not regions:
                    if isinstance(first_row, dict) and ('region' in first_row or 'acronym' in first_row):
                        for row in data:
                            region = row.get('region') or row.get('acronym')
                            if region and region not in regions:
                                regions.append(region)

                        if regions:
                            logger.info(f"  Found {len(regions)} regions from {key}")

                # 查找mismatch pairs
                if not mismatch_pairs:
                    if isinstance(first_row, dict) and any(
                            k in first_row
                            for k in ['mismatch_combined', 'mismatch_GM', 'region1', 'region2']
                    ):
                        mismatch_pairs.extend(data)
                        logger.info(f"  Found {len(data)} mismatch pairs from {key}")

        # 最终验证
        if not regions:
            logger.error("❌ No regions found in agent result")
            logger.error(f"   Available keys: {list(agent_result.keys())}")

            if 'executed_steps' in agent_result:
                logger.error(f"   Executed steps: {len(agent_result['executed_steps'])}")
                for i, step in enumerate(agent_result['executed_steps'], 1):
                    purpose = step.get('purpose', 'Unknown')
                    has_actual = 'actual_result' in step
                    logger.error(f"     Step {i}: {purpose[:40]} - has_actual_result: {has_actual}")

            return None

        if not mismatch_pairs:
            logger.error("❌ No mismatch pairs found in agent result")
            return None

        logger.info(f"✅ Successfully extracted: {len(regions)} regions, {len(mismatch_pairs)} pairs")

        return {
            'regions': regions,
            'mismatch_pairs': mismatch_pairs
        }

    def _build_matrices_from_pairs(self,
                                   regions: List[str],
                                   pairs: List[Dict]) -> Optional[Dict]:
        """
        从pair列表构建矩阵

        Returns:
            {
                'mol_sim': DataFrame,
                'morph_sim': DataFrame,
                'proj_sim': DataFrame,
                'mismatch_GM': DataFrame,
                'mismatch_GP': DataFrame
            }
        """
        n = len(regions)
        region_to_idx = {r: i for i, r in enumerate(regions)}

        # 初始化矩阵
        mol_sim = np.full((n, n), np.nan)
        morph_sim = np.full((n, n), np.nan)
        proj_sim = np.full((n, n), np.nan)
        mismatch_GM = np.full((n, n), np.nan)
        mismatch_GP = np.full((n, n), np.nan)

        # 对角线设为1（自己和自己相似度=1）
        np.fill_diagonal(mol_sim, 1.0)
        np.fill_diagonal(morph_sim, 1.0)
        np.fill_diagonal(proj_sim, 1.0)
        np.fill_diagonal(mismatch_GM, 0.0)
        np.fill_diagonal(mismatch_GP, 0.0)

        # 填充数据
        for pair in pairs:
            r1 = pair.get('region1')
            r2 = pair.get('region2')

            if not r1 or not r2:
                continue

            if r1 not in region_to_idx or r2 not in region_to_idx:
                continue

            i = region_to_idx[r1]
            j = region_to_idx[r2]

            # 相似度
            mol_sim[i, j] = mol_sim[j, i] = pair.get('sim_molecular', np.nan)
            morph_sim[i, j] = morph_sim[j, i] = pair.get('sim_morphological', np.nan)
            proj_sim[i, j] = proj_sim[j, i] = pair.get('sim_projection', np.nan)

            # Mismatch
            mismatch_GM[i, j] = mismatch_GM[j, i] = pair.get('mismatch_GM', np.nan)
            mismatch_GP[i, j] = mismatch_GP[j, i] = pair.get('mismatch_GP', np.nan)

        # 转换为DataFrame
        return {
            'mol_sim': pd.DataFrame(mol_sim, index=regions, columns=regions),
            'morph_sim': pd.DataFrame(morph_sim, index=regions, columns=regions),
            'proj_sim': pd.DataFrame(proj_sim, index=regions, columns=regions),
            'mismatch_GM': pd.DataFrame(mismatch_GM, index=regions, columns=regions),
            'mismatch_GP': pd.DataFrame(mismatch_GP, index=regions, columns=regions)
        }

    def _plot_similarity_matrices(self,
                                  mol_sim: pd.DataFrame,
                                  morph_sim: pd.DataFrame,
                                  proj_sim: pd.DataFrame,
                                  regions: List[str]) -> Dict[str, str]:
        """
        绘制3个similarity矩阵

        Returns:
            文件路径字典
        """
        output_files = {}

        # 1. Molecular Similarity
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mol_sim, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        ax.set_title('Molecular Fingerprint Similarity', fontsize=20, fontweight='bold')
        ax.set_xlabel('Region', fontsize=16, fontweight='bold')
        ax.set_ylabel('Region', fontsize=16, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / '1_molecular_similarity.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        output_files['molecular_similarity'] = str(filepath)

        # 2. Morphology Similarity
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(morph_sim, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        ax.set_title('Morphology Fingerprint Similarity', fontsize=20, fontweight='bold')
        ax.set_xlabel('Region', fontsize=16, fontweight='bold')
        ax.set_ylabel('Region', fontsize=16, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / '2_morphology_similarity.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        output_files['morphology_similarity'] = str(filepath)

        # 3. Projection Similarity
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(proj_sim, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        ax.set_title('Projection Fingerprint Similarity', fontsize=20, fontweight='bold')
        ax.set_xlabel('Region', fontsize=16, fontweight='bold')
        ax.set_ylabel('Region', fontsize=16, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / '3_projection_similarity.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        output_files['projection_similarity'] = str(filepath)

        return output_files

    def _plot_mismatch_matrices(self,
                                mismatch_GM: pd.DataFrame,
                                mismatch_GP: pd.DataFrame,
                                regions: List[str]) -> Dict[str, str]:
        """
        绘制2个mismatch矩阵
        """
        output_files = {}

        # 1. Molecular-Morphology Mismatch
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mismatch_GM, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Mismatch'},
                    xticklabels=True, yticklabels=True)
        ax.set_title('Molecular-Morphology Mismatch', fontsize=20, fontweight='bold')
        ax.set_xlabel('Region', fontsize=16, fontweight='bold')
        ax.set_ylabel('Region', fontsize=16, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / '4_mol_morph_mismatch.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        output_files['mol_morph_mismatch'] = str(filepath)

        # 2. Molecular-Projection Mismatch
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mismatch_GP, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Mismatch'},
                    xticklabels=True, yticklabels=True)
        ax.set_title('Molecular-Projection Mismatch', fontsize=20, fontweight='bold')
        ax.set_xlabel('Region', fontsize=16, fontweight='bold')
        ax.set_ylabel('Region', fontsize=16, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / '5_mol_proj_mismatch.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        output_files['mol_proj_mismatch'] = str(filepath)

        return output_files

    def _identify_top_pairs(self,
                            mismatch_GM: pd.DataFrame,
                            mismatch_GP: pd.DataFrame,
                            regions: List[str],
                            n: int = 5) -> Dict:
        """
        识别top N mismatch pairs

        Returns:
            {
                'mol_morph': [(r1, r2, mismatch_val), ...],
                'mol_proj': [(r1, r2, mismatch_val), ...]
            }
        """
        # Molecular-Morphology top pairs
        mm_values = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                val = mismatch_GM.iloc[i, j]
                if not np.isnan(val):
                    mm_values.append((regions[i], regions[j], val))

        mm_values.sort(key=lambda x: x[2], reverse=True)

        # Molecular-Projection top pairs
        mp_values = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                val = mismatch_GP.iloc[i, j]
                if not np.isnan(val):
                    mp_values.append((regions[i], regions[j], val))

        mp_values.sort(key=lambda x: x[2], reverse=True)

        return {
            'mol_morph': mm_values[:n],
            'mol_proj': mp_values[:n]
        }

    def _plot_detailed_comparisons(self,
                                   top_pairs: Dict,
                                   fingerprint_data: Dict,
                                   regions: List[str]) -> Dict[str, str]:
        """
        绘制top pairs的详细对比图（雷达图+柱状图）

        这需要原始的fingerprint数据
        """
        # TODO: 实现详细对比图
        # 需要从fingerprint_data中提取形态特征和投射数据
        logger.info("Detailed comparison plots not yet implemented")
        return {}


# ==================== Agent集成接口 ====================

def create_plotting_tool_for_agent(output_dir: str = "./figure4_agent_output") -> Figure4PlottingTool:
    """
    创建供Agent使用的绘图工具实例

    Args:
        output_dir: 输出目录

    Returns:
        Figure4PlottingTool实例
    """
    return Figure4PlottingTool(output_dir)


def generate_figure4_from_agent_result(agent_result: Dict,
                                       output_dir: str = "./figure4_agent_output") -> Dict[str, str]:
    """
    便捷函数：从Agent结果直接生成Figure 4所有图表

    Args:
        agent_result: Agent.answer()的返回结果
        output_dir: 输出目录

    Returns:
        生成的图片文件路径字典
    """
    tool = Figure4PlottingTool(output_dir)
    return tool.plot_from_agent_data(agent_result)
# ==================== Production Agent V10 ====================

class AIPOMCoTV10:
    """
    AIPOM-CoT V10 生产版本

    完整功能:
    1. 智能实体识别 (无需hardcoded列表)
    2. 动态Schema路径规划 (图算法)
    3. 结构化反思 (量化评估)
    4. 完整统计工具
    5. 多模态分析
    6. 自适应重规划
    """

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_pwd: str,
                 database: str,
                 schema_json_path: str,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o"):

        # 数据库连接
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)

        # Schema
        self.schema = RealSchemaCache(schema_json_path)

        # ===== 核心组件初始化 =====

        # P0-1: 智能实体识别
        logger.info("🔍 Initializing intelligent entity recognition...")
        self.entity_recognizer = IntelligentEntityRecognizer(self.db, self.schema)
        self.entity_clusterer = EntityClusteringEngine(self.db, self.schema)

        # P1-1: 动态Schema路径规划
        logger.info("🗺️  Initializing dynamic schema path planning...")
        self.path_planner = DynamicSchemaPathPlanner(self.schema)

        # P1-2: 结构化反思
        logger.info("🤔 Initializing structured reflection...")
        self.reflector = StructuredReflector()

        # 原有组件
        self.stats = StatisticalTools()
        self.fingerprint = RealFingerprintAnalyzer(self.db, self.schema)

        # OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model


        self.adaptive_planner = AdaptivePlanner(self.schema, self.path_planner,self.client)
        # 🆕 添加Focus-Driven Planner
        logger.info("🎯 Initializing focus-driven planning...")
        from focus_driven_planner import FocusDrivenPlanner
        self.focus_planner = FocusDrivenPlanner(self.schema, self.db)

        # 🆕 添加Comparative Analysis Planner
        logger.info("📊 Initializing comparative analysis planning...")
        from comparative_analysis_planner import ComparativeAnalysisPlanner
        self.comparative_planner = ComparativeAnalysisPlanner(
            self.db,
            self.fingerprint,
            self.stats
        )

        logger.info("✅ AIPOM-CoT V10 initialized successfully!")
        logger.info(f"   • Entity recognition: Ready")
        logger.info(f"   • Schema path planning: Ready")
        logger.info(f"   • Structured reflection: Ready")

    # ==================== Main Entry Point ====================

    """
    完整的answer方法实现 - 集成自适应规划
    """

    def answer(self, question: str, max_iterations: int = 15) -> Dict[str, Any]:
        """
        主入口: 回答问题 (完整版)

        完整流程:
        1. 智能实体识别
        2. 实体聚类
        3. 确定分析深度
        4. 智能选择规划器 (Adaptive/Focus-Driven/Comparative)
        5. 自适应执行循环 (包含统计分析)
        6. 答案合成 (科学叙事)
        """
        logger.info(f"🎯 Question: {question}")
        start_time = time.time()

        state = EnhancedAgentState(question=question)

        # ===== PHASE 1: INTELLIGENT PLANNING =====
        logger.info("\n" + "=" * 70)
        logger.info("📋 PHASE 1: INTELLIGENT PLANNING (Enhanced)")
        logger.info("=" * 70)

        state.phase = AgentPhase.PLANNING

        # Step 1-2: 实体识别 + 聚类
        logger.info("  [1/4] Intelligent entity recognition...")
        entity_matches = self.entity_recognizer.recognize_entities(question)
        state.entity_matches = entity_matches

        logger.info(f"     Found {len(entity_matches)} entity matches")
        for match in entity_matches[:5]:
            logger.info(f"       • {match.text} ({match.entity_type}) [{match.confidence:.2f}]")

        logger.info("  [2/4] Entity clustering...")
        entity_clusters = self.entity_clusterer.cluster_entities(entity_matches, question)
        state.entity_clusters = entity_clusters

        logger.info(f"     Created {len(entity_clusters)} entity clusters")
        for cluster in entity_clusters:
            logger.info(f"       • {cluster.cluster_type}: {cluster.primary_entity.text}")

        # 🆕 Step 3: 确定分析深度
        from adaptive_planner import determine_analysis_depth, AnalysisState

        logger.info("  [3/4] Determining analysis depth...")
        target_depth = determine_analysis_depth(question)
        logger.info(f"     Target depth: {target_depth.value}")

        # 🆕 Step 4: 初始化分析状态
        logger.info("  [4/4] Initializing analysis state...")

        analysis_state = AnalysisState(
            discovered_entities={},
            executed_steps=[],
            modalities_covered=[],
            current_focus='gene' if entity_clusters and entity_clusters[0].cluster_type == 'gene_marker' else 'region',
            target_depth=target_depth,
            question_intent=self._classify_question_intent(question)
        )

        # 填充初始实体
        for cluster in entity_clusters:
            entity_type = cluster.primary_entity.entity_type
            entity_id = cluster.primary_entity.entity_id

            analysis_state.discovered_entities.setdefault(entity_type, []).append(entity_id)

            for related in cluster.related_entities:
                analysis_state.discovered_entities.setdefault(
                    related.entity_type, []
                ).append(related.entity_id)

        # 兼容性
        state.entities = [
            {'text': m.text, 'type': m.entity_type, 'confidence': m.confidence}
            for m in entity_matches[:10]
        ]

        # 🆕 存储analysis_state到state
        state.analysis_state = analysis_state

        logger.info(f"✅ Planning complete")
        logger.info(f"   • Target depth: {target_depth.value}")
        logger.info(f"   • Initial entities: {list(analysis_state.discovered_entities.keys())}")

        # ===== PHASE 2: ADAPTIVE EXECUTION =====
        logger.info("\n" + "=" * 70)
        logger.info("⚙️ PHASE 2: ADAPTIVE EXECUTION (Multi-Planner)")
        logger.info("=" * 70)

        state.phase = AgentPhase.EXECUTING

        iteration = 0
        while iteration < max_iterations:
            # 🆕 决定是否继续
            if not self.adaptive_planner.should_continue(analysis_state, question):
                logger.info("📌 Analysis complete (adaptive decision)")
                break

            # 🆕 智能选择规划器
            planner_type = self._select_planner(analysis_state, question)

            if planner_type == 'focus_driven':
                logger.info(f"\n🎯 Using FOCUS-DRIVEN planner (iteration {iteration + 1})...")
                next_steps = self.focus_planner.generate_focus_driven_plan(
                    analysis_state,
                    question
                )

            elif planner_type == 'comparative':
                logger.info(f"\n📊 Using COMPARATIVE planner (iteration {iteration + 1})...")
                next_steps = self.comparative_planner.generate_comparative_plan(
                    analysis_state,
                    question
                )

            else:
                logger.info(f"\n🔄 Using ADAPTIVE planner (iteration {iteration + 1})...")
                next_steps = self.adaptive_planner.plan_next_steps(
                    analysis_state,
                    question,
                    max_steps=2
                )

            if not next_steps:
                logger.info("📌 No more steps available")
                break

            # 执行规划的步骤
            for candidate_step in next_steps:
                if iteration >= max_iterations:
                    break

                logger.info(f"\n🔹 Step {iteration + 1}: {candidate_step.purpose}")
                logger.info(f"   Type: {candidate_step.step_type}")
                logger.info(f"   Priority: {candidate_step.priority:.1f}")
                if hasattr(candidate_step, 'llm_score') and candidate_step.llm_score > 0:
                    logger.info(f"   LLM score: {candidate_step.llm_score:.2f}")

                # 🆕 转换为ReasoningStep
                reasoning_step = self._convert_candidate_to_reasoning(
                    candidate_step,
                    iteration + 1,
                    analysis_state
                )

                # 执行
                exec_result = self._execute_step(reasoning_step, state)

                if not exec_result['success']:
                    logger.error(f"   ❌ Failed: {exec_result.get('error')}")

                    if state.replanning_count < state.max_replanning:
                        logger.info(f"   🔄 Replanning...")
                        state.replanning_count += 1

                    continue

                # 🆕 结构化反思
                structured_reflection = self.reflector.reflect(
                    step_number=reasoning_step.step_number,
                    purpose=reasoning_step.purpose,
                    expected_result=reasoning_step.expected_result,
                    actual_result=reasoning_step.actual_result,
                    question_context=question
                )

                reasoning_step.reflection = structured_reflection.summary
                reasoning_step.validation_passed = (
                        structured_reflection.validation_status.value in ['passed', 'partial']
                )

                state.structured_reflections.append(structured_reflection)
                state.reflections.append(structured_reflection.summary)

                logger.info(f"   📊 Reflection: {structured_reflection.summary}")
                logger.info(f"   📈 Confidence: {structured_reflection.confidence_score:.3f}")

                # 🆕 更新分析状态
                self._update_analysis_state(
                    analysis_state,
                    reasoning_step,
                    exec_result,
                    candidate_step
                )

                state.executed_steps.append(reasoning_step)
                iteration += 1

        # ===== PHASE 3: ANSWER SYNTHESIS =====
        logger.info("\n" + "=" * 70)
        logger.info("📝 PHASE 3: ANSWER SYNTHESIS")
        logger.info("=" * 70)

        final_answer = self._synthesize_answer(state)

        execution_time = time.time() - start_time

        # 构建返回结果
        result = {
            'question': question,
            'answer': final_answer,

            'entities_recognized': [
                {
                    'text': m.text,
                    'type': m.entity_type,
                    'confidence': m.confidence,
                    'match_type': m.match_type
                }
                for m in state.entity_matches[:10]
            ],

            'reasoning_plan': [self._step_to_dict(s) for s in state.executed_steps],
            'executed_steps': [self._step_to_dict(s) for s in state.executed_steps],

            'reflections': state.reflections,
            'structured_reflections': [
                {
                    'step': r.step_number,
                    'status': r.validation_status.value,
                    'confidence': r.confidence_score,
                    'uncertainty': r.uncertainty.overall_uncertainty,
                    'should_replan': r.should_replan
                }
                for r in state.structured_reflections
            ],

            # 🆕 自适应规划信息
            'adaptive_planning': {
                'target_depth': target_depth.value,
                'final_depth': len(state.executed_steps),
                'modalities_covered': analysis_state.modalities_covered,
                'entities_discovered': {
                    k: len(v) for k, v in analysis_state.discovered_entities.items()
                },
                'primary_focus': getattr(analysis_state, 'primary_focus', None)
            },

            'replanning_count': state.replanning_count,
            'confidence_score': state.confidence_score,
            'execution_time': execution_time,
            'total_steps': len(state.executed_steps),
            'schema_paths_used': state.schema_paths_used,
            'intermediate_data': state.intermediate_data
        }

        logger.info(f"\n✅ Completed in {execution_time:.2f}s")
        logger.info(f"   • Steps executed: {len(state.executed_steps)}")
        logger.info(f"   • Confidence: {state.confidence_score:.3f}")
        logger.info(f"   • Modalities: {', '.join(analysis_state.modalities_covered)}")

        return result

    def answer_with_visualization(self,
                                  question: str,
                                  max_iterations: int = 15,
                                  generate_plots: bool = True,
                                  output_dir: str = "./figure4_results") -> Dict[str, Any]:
        """
        回答问题并生成可视化（Figure 4增强版）

        Args:
            question: 问题
            max_iterations: 最大迭代次数
            generate_plots: 是否生成图表
            output_dir: 图表输出目录

        Returns:
            包含answer和visualization_files的结果
        """
        # 1. 正常执行分析
        result = self.answer(question, max_iterations)

        # 2. 如果是Figure 4类型的分析，生成图表
        if generate_plots:
            analysis_type = self._detect_analysis_type_from_result(result)

            if analysis_type == 'figure4_mismatch':
                logger.info("\n" + "=" * 70)
                logger.info("🎨 GENERATING FIGURE 4 VISUALIZATIONS")
                logger.info("=" * 70)

                try:
                    visualization_files = generate_figure4_from_agent_result(
                        result,
                        output_dir
                    )

                    result['visualization_files'] = visualization_files
                    result['visualization_output_dir'] = output_dir

                    logger.info(f"\n✅ Generated {len(visualization_files)} figures:")
                    for name, path in visualization_files.items():
                        logger.info(f"   • {name}: {path}")

                except Exception as e:
                    logger.error(f"❌ Visualization generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    result['visualization_error'] = str(e)

        return result

    def _detect_analysis_type_from_result(self, result: Dict) -> str:
        """
        从结果中检测分析类型

        Returns:
            'figure4_mismatch' | 'figure3_focus' | 'other'
        """
        # 检查是否有mismatch计算
        has_mismatch = any(
            'mismatch' in step['purpose'].lower()
            for step in result.get('executed_steps', [])
        )

        # 检查是否是systematic screening
        has_screening = any(
            'systematic' in step['purpose'].lower() or
            'top' in step['purpose'].lower() and 'region' in step['purpose'].lower()
            for step in result.get('executed_steps', [])
        )

        if has_mismatch and has_screening:
            return 'figure4_mismatch'

        return 'other'

    # ==================== 辅助方法 ====================
    def _select_planner(self, state, question: str) -> str:
        """
        智能选择规划器（增强版 v2.0）

        🔧 关键修复：
        1. 增强systematic screening检测
        2. 添加更多比较关键词
        3. 改进日志

        Returns:
            'focus_driven' | 'comparative' | 'adaptive'
        """
        q_lower = question.lower()

        logger.info(f"   🎯 Selecting planner for: {question[:60]}...")

        # ====== Priority 1: 比较查询 → Comparative ======
        compare_keywords = [
            'compare', 'comparison', 'comparing',
            'versus', 'vs ', 'vs.', ' vs',
            'difference between', 'differences between',
            'contrast', 'contrasting',
            'distinguish', 'differentiate',
        ]

        for keyword in compare_keywords:
            if keyword in q_lower:
                logger.info(f"      Comparison keyword '{keyword}' detected → comparative")
                return 'comparative'

        # ====== Priority 2: Systematic screening → Comparative ======
        # 🔧 增强检测逻辑

        # 关键词检测
        systematic_keywords = [
            'which regions', 'which brain regions', 'which areas',
            'find all', 'identify all', 'list all',
            'screen', 'screening', 'systematic', 'systematically',
            'highest', 'top regions', 'top brain regions',
            'most', 'strongest', 'largest',
            'mismatch', 'discordant', 'inconsistent', 'divergent',
            'show', 'exhibit', 'demonstrate', 'display', 'have'
        ]

        # 检查是否包含systematic关键词
        has_systematic_keyword = any(kw in q_lower for kw in systematic_keywords)

        # 模式检测
        has_which = 'which' in q_lower
        has_superlative = any(w in q_lower for w in ['highest', 'top', 'most', 'strongest', 'largest', 'best', 'worst'])
        has_mismatch = any(w in q_lower for w in ['mismatch', 'discordant', 'inconsistent', 'divergent'])
        has_show_verb = any(w in q_lower for w in ['show', 'exhibit', 'demonstrate', 'display', 'have'])

        # 组合判断
        is_systematic = False
        reason = ""

        if has_which and has_superlative:
            is_systematic = True
            reason = "which + superlative pattern"
        elif has_which and has_mismatch:
            is_systematic = True
            reason = "which + mismatch pattern"
        elif has_which and has_show_verb:
            is_systematic = True
            reason = "which + show/exhibit pattern"
        elif has_systematic_keyword:
            is_systematic = True
            reason = f"systematic keyword detected"

        if is_systematic:
            logger.info(f"      Systematic screening detected ({reason}) → comparative")
            return 'comparative'

        # ====== Priority 3: Focus-driven → 有regions的深度查询 ======
        if 'Region' in state.discovered_entities:
            n_regions = len(state.discovered_entities.get('Region', []))
            if n_regions > 0:
                logger.info(f"      {n_regions} regions found → focus_driven")
                return 'focus_driven'

        # ====== Priority 4: Focus-driven → Gene查询且有深度意图 ======
        if 'GeneMarker' in state.discovered_entities:
            deep_intent_keywords = [
                'tell me about', 'about',
                'analyze', 'analysis', 'characterize', 'characterization',
                'comprehensive', 'detailed', 'in-depth'
            ]

            if any(kw in q_lower for kw in deep_intent_keywords):
                logger.info(f"      Gene query with deep intent → focus_driven")
                return 'focus_driven'

        # ====== Default: Adaptive ======
        logger.info(f"      Default → adaptive")
        return 'adaptive'

    def _classify_question_intent(self, question: str) -> str:
        """分类问题意图"""
        question_lower = question.lower()

        if any(w in question_lower for w in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(w in question_lower for w in ['comprehensive', 'detailed', 'everything']):
            return 'comprehensive'
        elif any(w in question_lower for w in ['why', 'explain', 'how']):
            return 'explanatory'
        elif any(w in question_lower for w in ['which', 'find', 'identify']):
            return 'screening'
        else:
            return 'simple_query'


    def _convert_candidate_to_reasoning(self, candidate, step_number, analysis_state):
        """转换CandidateStep (修复版)"""
        params = candidate.parameters.copy()

        # 🔧 智能判断action
        has_cypher = bool(candidate.cypher_template and candidate.cypher_template.strip())

        if not has_cypher:
            # 特殊步骤
            if 'statistical' in candidate.step_type.lower() or 'fdr' in candidate.step_id.lower():
                action = 'execute_statistical'
            elif 'multi-modal' in candidate.step_type.lower() or 'mismatch' in candidate.step_id.lower():
                action = 'execute_fingerprint'
            else:
                action = 'execute_cypher'
        else:
            action = 'execute_cypher'

        return ReasoningStep(
            step_number=step_number,
            purpose=candidate.purpose,
            action=action,  # 🔧 正确的action
            rationale=candidate.rationale,
            expected_result=candidate.expected_data,
            query_or_params={
                'query': candidate.cypher_template,
                'params': params
            },
            modality=candidate.step_type,
            depends_on=getattr(candidate, 'depends_on', [])
        )

    def _update_analysis_state(self,
                               analysis_state,
                               step: ReasoningStep,
                               result: Dict,
                               candidate):
        """
        更新分析状态（增强版 v2.0）

        🔧 关键修复：
        1. 多字段兼容的ProjectionTarget提取
        2. 智能fallback机制
        3. 增强日志
        """
        # 记录执行的步骤
        analysis_state.executed_steps.append({
            'purpose': step.purpose,
            'modality': step.modality,
            'row_count': len(result.get('data', [])),
            'step_id': candidate.step_id
        })

        # 更新modality覆盖
        if step.modality and step.modality not in analysis_state.modalities_covered:
            analysis_state.modalities_covered.append(step.modality)

        # 🆕 提取新发现的实体
        data = result.get('data', [])
        if not data:
            return

        first_row = data[0]

        # ====== 提取Regions ======
        if 'region' in first_row or 'acronym' in first_row:
            regions = list(set([
                row.get('region') or row.get('acronym')
                for row in data
                if row.get('region') or row.get('acronym')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Region', [])
            for r in regions:
                if r and r not in existing:
                    existing.append(r)

        # ====== 提取Clusters ======
        if 'cluster' in first_row or 'cluster_name' in first_row:
            clusters = list(set([
                row.get('cluster') or row.get('cluster_name')
                for row in data
                if row.get('cluster') or row.get('cluster_name')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Cluster', [])
            for c in clusters:
                if c and c not in existing:
                    existing.append(c)

        # ====== 提取Subclasses ======
        if 'subclass' in first_row or 'subclass_name' in first_row:
            subclasses = list(set([
                row.get('subclass') or row.get('subclass_name')
                for row in data
                if row.get('subclass') or row.get('subclass_name')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Subclass', [])
            for s in subclasses:
                if s and s not in existing:
                    existing.append(s)

        # ====== 🔧 增强: 提取Projection Targets (多策略) ======

        # 策略1: 检查常见字段名
        target_field_candidates = [
            'target', 'target_region', 'target_acronym', 'target_name',
            'tgt', 'tgt_region', 'projection_target',
            'downstream', 'downstream_region',
            'dest', 'destination', 'to_region'  # 添加更多可能的字段名
        ]

        targets_found = []
        matched_field = None

        for field in target_field_candidates:
            if field in first_row:
                matched_field = field
                targets_found = [
                    row.get(field)
                    for row in data
                    if row.get(field) and isinstance(row.get(field), str)
                ]
                if targets_found:
                    logger.info(f"   📍 Found targets via field: '{field}'")
                    break

        # 策略2: 如果step purpose包含"projection"但没找到标准字段，智能提取
        if not targets_found and 'projection' in step.purpose.lower():
            logger.info(f"   🔍 Fallback: Intelligent target extraction")

            # 尝试从所有字段中找region-like值
            for row in data[:20]:  # 检查前20行
                for key, value in row.items():
                    # 跳过明显的source字段
                    if key in ['source', 'source_region', 'region', 'acronym']:
                        continue

                    # 识别可能的region acronym (2-5个大写字母)
                    if isinstance(value, str) and 2 <= len(value) <= 5 and value.isupper():
                        targets_found.append(value)
                        logger.debug(f"      Found potential target: {value} (from field: {key})")

        # 策略3: 检查step的actual_result中是否有summary信息
        if not targets_found and hasattr(step, 'actual_result'):
            actual = step.actual_result
            if isinstance(actual, dict) and 'summary' in actual:
                summary = actual['summary']
                if 'targets' in summary:
                    targets_found = summary['targets']
                    logger.info(f"   📍 Found targets from step summary")

        # 去重并添加到discovered_entities
        if targets_found:
            targets_unique = list(set([t for t in targets_found if t]))

            existing = analysis_state.discovered_entities.setdefault('ProjectionTarget', [])

            new_targets = []
            for t in targets_unique:
                if t and t not in existing:
                    existing.append(t)
                    new_targets.append(t)

            if new_targets:
                logger.info(f"   📍 Discovered {len(new_targets)} NEW projection targets: {new_targets[:5]}")
                logger.info(f"      Total targets now: {len(existing)}")
            else:
                logger.info(f"   📍 Found {len(targets_unique)} targets (already known)")
        else:
            # 如果是projection步骤但没找到targets，警告
            if 'projection' in step.purpose.lower():
                logger.warning(f"   ⚠️ Projection step but no targets extracted!")
                logger.warning(f"      Available fields: {list(first_row.keys())}")
                logger.warning(f"      This may prevent closed-loop analysis")


    def _enhanced_planning_phase(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        增强的规划阶段

        步骤:
        1. 智能实体识别 (无hardcoded列表!)
        2. 实体聚类
        3. 动态Schema路径规划
        4. LLM精化
        """
        try:
            # Step 1: 实体识别
            logger.info("  [1/4] Intelligent entity recognition...")
            entity_matches = self.entity_recognizer.recognize_entities(state.question)
            state.entity_matches = entity_matches

            logger.info(f"     Found {len(entity_matches)} entity matches")
            for match in entity_matches[:5]:
                logger.info(f"       • {match.text} ({match.entity_type}) [{match.confidence:.2f}]")

            # Step 2: 实体聚类
            logger.info("  [2/4] Entity clustering...")
            entity_clusters = self.entity_clusterer.cluster_entities(
                entity_matches,
                state.question
            )
            state.entity_clusters = entity_clusters

            logger.info(f"     Created {len(entity_clusters)} entity clusters")
            for cluster in entity_clusters:
                logger.info(f"       • {cluster.cluster_type}: {cluster.primary_entity.text}")

            # Step 3: 动态Schema路径规划
            logger.info("  [3/4] Dynamic schema path planning...")
            query_plans = self.path_planner.generate_plan(entity_clusters, state.question)

            logger.info(f"     Generated {len(query_plans)} query plans")

            # 记录使用的schema路径
            for plan in query_plans:
                if plan.schema_path.hops:
                    state.schema_paths_used.append({
                        'start': plan.schema_path.start_label,
                        'end': plan.schema_path.end_label,
                        'hops': len(plan.schema_path.hops),
                        'score': plan.schema_path.score
                    })

            # Step 4: LLM精化
            logger.info("  [4/4] LLM plan refinement...")
            refined_steps = self._llm_refine_plans(query_plans, state)
            state.reasoning_plan = refined_steps

            # 保存实体到state (兼容原有格式)
            state.entities = [
                {
                    'text': m.text,
                    'type': m.entity_type,
                    'confidence': m.confidence
                }
                for m in entity_matches[:10]
            ]

            return {'success': True}

        except Exception as e:
            logger.error(f"Enhanced planning failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _llm_refine_plans(self,
                          query_plans: List,
                          state: EnhancedAgentState) -> List[ReasoningStep]:
        """
        LLM精化查询计划

        将动态生成的QueryPlan转换为ReasoningStep,并让LLM补充细节
        """
        # 转换为字典格式
        plans_dict = []
        for qp in query_plans:
            plans_dict.append({
                'step': qp.step_number,
                'purpose': qp.purpose,
                'action': qp.action,
                'query': qp.cypher_template,
                'parameters': qp.parameters,
                'modality': qp.modality,
                'depends_on': qp.depends_on,
                'schema_path_score': qp.schema_path.score if qp.schema_path else 0.0
            })

        prompt = f"""You are refining a reasoning plan for neuroscience knowledge graph analysis.

**Question:** {state.question}

**Recognized Entities:** {', '.join([e['text'] for e in state.entities])}

**Dynamically Generated Query Plans:**
{json.dumps(plans_dict, indent=2)}

Your task:
1. Review each query plan
2. Add detailed **expected_result** descriptions
3. Enhance **rationale** with domain knowledge
4. Verify Cypher query correctness
5. Add any missing steps if needed

Return a JSON object with key "steps" containing an array:
{{
  "steps": [
    {{
      "step_number": 1,
      "purpose": "...",
      "action": "execute_cypher",
      "rationale": "Detailed explanation",
      "expected_result": "Concrete prediction of what data will look like",
      "query_or_params": {{"query": "...", "params": {{}}}},
      "modality": "molecular/morphological/projection",
      "depends_on": []
    }},
    ...
  ]
}}

**Important:**
- Make rationale SPECIFIC and scientifically grounded
- Expected results should describe DATA PATTERNS (e.g., "10-20 clusters with neuron counts ranging 500-5000")
- Ensure query syntax is correct
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert neuroscientist and Neo4j query expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(response.choices[0].message.content)

            # 转换为ReasoningStep
            steps = []
            for step_dict in result.get('steps', []):
                query_or_params = step_dict.get('query_or_params', {})

                # 处理参数替换
                if isinstance(query_or_params, dict):
                    if 'query' not in query_or_params and 'query' in step_dict:
                        query_or_params = {'query': step_dict['query']}

                step = ReasoningStep(
                    step_number=step_dict.get('step_number', len(steps) + 1),
                    purpose=step_dict.get('purpose', ''),
                    action=step_dict.get('action', 'execute_cypher'),
                    rationale=step_dict.get('rationale', ''),
                    expected_result=step_dict.get('expected_result', ''),
                    query_or_params=query_or_params,
                    modality=step_dict.get('modality'),
                    depends_on=step_dict.get('depends_on', [])
                )
                steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")

            # Fallback: 直接转换QueryPlan
            fallback_steps = []
            for qp in query_plans:
                step = ReasoningStep(
                    step_number=qp.step_number,
                    purpose=qp.purpose,
                    action=qp.action,
                    rationale="Automatically generated from schema path",
                    expected_result="Data matching query criteria",
                    query_or_params={'query': qp.cypher_template, 'params': qp.parameters},
                    modality=qp.modality,
                    depends_on=qp.depends_on
                )
                fallback_steps.append(step)

            return fallback_steps

    def _characterize_top_pairs(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """
        深入分析top mismatch pairs (Case Study)

        🆕 新增功能:
        1. 提取top N pairs
        2. 查询每个pair的详细数据:
           - Morphological features
           - Projection targets
           - Molecular composition
        """
        n_top = params.get('n_top_pairs', 3)

        # 从FDR结果获取top pairs
        fdr_data = None
        for key, data in state.intermediate_data.items():
            if data and isinstance(data, list) and len(data) > 0:
                if 'fdr_significant' in data[0] and data[0].get('fdr_significant'):
                    fdr_data = data
                    break

        if not fdr_data:
            logger.warning("   No FDR significant pairs found, using top mismatch pairs")
            # Fallback: 使用top mismatch
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    if 'mismatch_combined' in data[0]:
                        fdr_data = sorted(data, key=lambda x: x['mismatch_combined'], reverse=True)
                        break

        if not fdr_data:
            return {'success': False, 'error': 'No mismatch data found', 'data': []}

        # 选择top N pairs
        top_pairs = fdr_data[:n_top]

        logger.info(f"   Analyzing top {len(top_pairs)} pairs:")
        for pair in top_pairs:
            logger.info(f"     • {pair['region1']} vs {pair['region2']}: mismatch={pair['mismatch_combined']:.3f}")

        # 详细分析每个pair
        detailed_results = []

        for pair in top_pairs:
            region1 = pair['region1']
            region2 = pair['region2']

            logger.info(f"   Deep characterization: {region1} vs {region2}")

            # 🔹 1. Morphological comparison
            morph_query = """
            MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
            WHERE r.acronym IN [$region1, $region2]
            RETURN r.acronym AS region,
                   count(n) AS neuron_count,
                   avg(n.axonal_length) AS avg_axon,
                   avg(n.dendritic_length) AS avg_dendrite,
                   avg(n.axonal_branches) AS avg_axon_branches,
                   avg(n.dendritic_branches) AS avg_dendrite_branches,
                   stdev(n.axonal_length) AS std_axon,
                   stdev(n.dendritic_length) AS std_dendrite
            """
            morph_result = self.db.run(morph_query, {'region1': region1, 'region2': region2})

            # 🔹 2. Projection targets comparison
            proj_query = """
            MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
            WHERE r.acronym IN [$region1, $region2]
            RETURN r.acronym AS source,
                   t.acronym AS target,
                   t.name AS target_name,
                   p.weight AS weight
            ORDER BY r.acronym, p.weight DESC
            LIMIT 30
            """
            proj_result = self.db.run(proj_query, {'region1': region1, 'region2': region2})

            # 🔹 3. Molecular composition
            mol_query = """
            MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
            WHERE r.acronym IN [$region1, $region2]
            RETURN r.acronym AS region,
                   c.name AS cluster,
                   c.markers AS markers,
                   c.number_of_neurons AS neurons
            ORDER BY r.acronym, c.number_of_neurons DESC
            LIMIT 20
            """
            mol_result = self.db.run(mol_query, {'region1': region1, 'region2': region2})

            # 整合结果
            detailed_results.append({
                'pair': f"{region1}_vs_{region2}",
                'region1': region1,
                'region2': region2,
                'mismatch_score': pair['mismatch_combined'],
                'p_value': pair.get('p_value', 1.0),
                'q_value': pair.get('q_value', 1.0),
                'morphology': morph_result.get('data', []),
                'projections': proj_result.get('data', []),
                'molecular': mol_result.get('data', [])
            })

        logger.info(f"   ✅ Detailed characterization complete for {len(detailed_results)} pairs")

        return {
            'success': True,
            'data': detailed_results,
            'rows': len(detailed_results),
            'analysis_type': 'case_study'
        }

    # ==================== Execution ====================

    def _execute_step(self, step: ReasoningStep, state: EnhancedAgentState) -> Dict[str, Any]:
        """执行单个步骤 (修复版 - 支持case study)"""
        start_time = time.time()

        try:
            query = step.query_or_params.get('query', '').strip()
            params = step.query_or_params.get('params', {})

            # 判断执行类型
            if not query:
                # 🆕 Case study检测
                if 'characterize' in step.purpose.lower() and 'top' in step.purpose.lower():
                    result = self._characterize_top_pairs(params, state)
                elif 'mismatch' in step.purpose.lower():
                    result = self._execute_fingerprint_step(step, state)
                elif 'statistical' in step.purpose.lower() or 'fdr' in step.purpose.lower():
                    result = self._execute_statistical_step(step, state)
                else:
                    result = {'success': False, 'error': 'Cannot determine execution type'}
            else:
                result = self._execute_cypher_step(step, state)

            step.actual_result = result
            step.execution_time = time.time() - start_time

            step_key = f"step_{step.step_number}"
            state.intermediate_data[step_key] = result.get('data', [])

            return result

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _execute_cypher_step(self, step: ReasoningStep, state: EnhancedAgentState) -> Dict[str, Any]:
        """执行Cypher查询步骤"""
        query = step.query_or_params.get('query', '').strip()
        params = step.query_or_params.get('params', {})

        # 🔧 空查询检查
        if not query:
            logger.warning(f"   Empty Cypher query - skipping")
            return {'success': False, 'error': 'Empty query', 'data': []}

        # 参数替换
        if step.depends_on:
            params = self._resolve_parameters(step, state, params)

        # 自动添加LIMIT
        import re
        if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            query = f"{query}\nLIMIT 100"

        return self.db.run(query, params)

    def _execute_statistical_step(self,
                                  step: ReasoningStep,
                                  state: EnhancedAgentState) -> Dict[str, Any]:
        """
        🆕 执行统计步骤
        """
        params = step.query_or_params.get('params', {})
        test_type = params.get('test_type', 'permutation')

        logger.info(f"   📊 Statistical test: {test_type}")

        try:
            if test_type == 'permutation':
                return self._permutation_test(params, state)

            elif test_type == 'fdr':
                return self._fdr_correction(params, state)

            elif test_type == 'correlation':
                return self._correlation_test(params, state)

            else:
                return {'success': False, 'error': f'Unknown test type: {test_type}'}

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _execute_fingerprint_step(self,
                                  step: ReasoningStep,
                                  state: EnhancedAgentState) -> Dict[str, Any]:
        """
        🆕 执行fingerprint计算步骤
        """
        params = step.query_or_params.get('params', {})
        analysis_type = params.get('analysis_type', 'cross_modal_mismatch')

        logger.info(f"   🔬 Fingerprint analysis: {analysis_type}")

        if analysis_type == 'cross_modal_mismatch':
            return self._compute_mismatch_matrix(params, state)
        else:
            return {'success': False, 'error': f'Unknown analysis type: {analysis_type}'}

    def _resolve_parameters(self,
                            step: ReasoningStep,
                            state: EnhancedAgentState,
                            params: Dict) -> Dict:
        """解析步骤依赖的参数"""
        resolved = params.copy()

        # 查找依赖步骤的数据
        for dep_num in step.depends_on:
            dep_key = f"step_{dep_num}"
            if dep_key in state.intermediate_data:
                dep_data = state.intermediate_data[dep_key]

                # 提取常用字段
                if dep_data:
                    # 提取region acronyms
                    regions = []
                    for row in dep_data:
                        if 'region' in row:
                            regions.append(row['region'])
                        elif 'acronym' in row:
                            regions.append(row['acronym'])

                    if regions:
                        resolved['enriched_regions'] = regions[:10]
                        resolved['target_regions'] = regions[:10]

        return resolved

    def _execute_cypher(self, query: str, params: Dict) -> Dict[str, Any]:
        """执行Cypher查询"""
        import re

        # 确保有LIMIT
        if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            query = f"{query}\nLIMIT 100"

        return self.db.run(query, params)

    def _permutation_test(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """Permutation test for morphological differences"""
        entity_a = params['entity_a']
        entity_b = params['entity_b']

        # 从之前的step获取数据
        morph_data = None
        for key, data in state.intermediate_data.items():
            if data and isinstance(data, list) and len(data) > 0:
                if 'region' in data[0] and ('avg_axon' in data[0] or 'avg_axon_length' in data[0]):
                    morph_data = data
                    break

        if not morph_data:
            return {'success': False, 'error': 'No morphological data found'}

        # 提取两组数据
        group_a = [row for row in morph_data if row.get('region') == entity_a]
        group_b = [row for row in morph_data if row.get('region') == entity_b]

        if not group_a or not group_b:
            return {'success': False,
                    'error': f'Insufficient data: {entity_a}={len(group_a)}, {entity_b}={len(group_b)}'}

        # 提取axon length
        import numpy as np
        axon_key = 'avg_axon' if 'avg_axon' in group_a[0] else 'avg_axon_length'
        axon_a = np.array([row.get(axon_key, 0) or 0 for row in group_a])
        axon_b = np.array([row.get(axon_key, 0) or 0 for row in group_b])

        # 移除零值
        axon_a = axon_a[axon_a > 0]
        axon_b = axon_b[axon_b > 0]

        if len(axon_a) == 0 or len(axon_b) == 0:
            return {'success': False, 'error': 'No valid morphology data'}

        # 计算observed difference
        observed_diff = float(np.mean(axon_a) - np.mean(axon_b))

        # 🎯 调用统计工具!
        result = self.stats.permutation_test(
            observed_stat=observed_diff,
            data1=axon_a,
            data2=axon_b,
            n_permutations=1000,
            seed=42
        )

        # 计算effect size
        effect_size = self.stats.cohens_d(axon_a, axon_b)

        # 格式化结果
        result_data = [{
            'comparison': f'{entity_a} vs {entity_b}',
            'feature': 'axonal_length',
            'mean_a': float(np.mean(axon_a)),
            'mean_b': float(np.mean(axon_b)),
            'observed_difference': observed_diff,
            'p_value': result['p_value'],
            'effect_size_cohens_d': effect_size,
            'significance': 'significant' if result['p_value'] < 0.05 else 'not significant',
            'interpretation': self._interpret_statistical_result(result, effect_size)
        }]

        logger.info(f"   ✅ Permutation test: p={result['p_value']:.4f}, d={effect_size:.2f}")

        return {
            'success': True,
            'data': result_data,
            'rows': len(result_data),
            'test_type': 'permutation'
        }

    def _fdr_correction(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """
        FDR correction (超强调试版)

        🔧 全面调试和容错
        """
        alpha = params.get('alpha', 0.05)

        logger.info(f"   === FDR Correction Debug ===")
        logger.info(f"   Available data keys: {list(state.intermediate_data.keys())}")

        # 🔧 增强数据查找
        mismatch_data = None
        mismatch_key = None

        # 策略1: 查找包含'mismatch_combined'和'p_value'的数据
        for key, data in state.intermediate_data.items():
            logger.debug(f"   Checking {key}: type={type(data)}, len={len(data) if isinstance(data, list) else 'N/A'}")

            if not data:
                continue

            if isinstance(data, list) and len(data) > 0:
                first_row = data[0]
                logger.debug(
                    f"     First row keys: {first_row.keys() if isinstance(first_row, dict) else 'Not a dict'}")

                # 检查必需字段
                has_mismatch = 'mismatch_combined' in first_row if isinstance(first_row, dict) else False
                has_pvalue = 'p_value' in first_row if isinstance(first_row, dict) else False

                logger.debug(f"     has_mismatch={has_mismatch}, has_pvalue={has_pvalue}")

                if has_mismatch and has_pvalue:
                    mismatch_data = data
                    mismatch_key = key
                    logger.info(f"   ✓ Found mismatch data in {key} ({len(data)} rows)")
                    break

        # 策略2: 如果没找到，尝试从最近的step获取
        if not mismatch_data:
            logger.warning("   Strategy 1 failed, trying strategy 2...")

            # 按key排序，找最近的step
            sorted_keys = sorted([k for k in state.intermediate_data.keys() if k.startswith('step_')],
                                 key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 and x.split('_')[
                                     1].isdigit() else 0,
                                 reverse=True)

            logger.debug(f"   Sorted keys: {sorted_keys}")

            for key in sorted_keys:
                data = state.intermediate_data[key]
                if data and isinstance(data, list) and len(data) > 0:
                    first_row = data[0]
                    if isinstance(first_row, dict) and 'mismatch_combined' in first_row:
                        logger.info(f"   ✓ Found mismatch data in {key} (strategy 2)")
                        mismatch_data = data
                        mismatch_key = key

                        # 🔧 如果没有p_value，添加默认值
                        if 'p_value' not in first_row:
                            logger.warning(f"   Adding default p_values")
                            for row in mismatch_data:
                                if 'p_value' not in row:
                                    row['p_value'] = 1.0 - min(0.99, row.get('mismatch_combined', 0))

                        break

        # 最终检查
        if not mismatch_data:
            logger.error("   ✗ No mismatch data found!")
            logger.error(f"   Available keys: {list(state.intermediate_data.keys())}")

            # 打印所有数据的样本
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    logger.error(
                        f"   {key} sample: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")

            return {
                'success': False,
                'error': 'No mismatch data with p-values found',
                'data': []
            }

        # 提取p-values
        p_values = []
        for row in mismatch_data:
            pval = row.get('p_value', None)
            if pval is not None:
                p_values.append(float(pval))
            else:
                logger.warning(
                    f"   Row missing p_value: {row.get('region1', 'unknown')}-{row.get('region2', 'unknown')}")
                p_values.append(1.0)

        logger.info(f"   FDR input: {len(p_values)} p-values")
        logger.info(f"   P-value range: [{min(p_values):.4f}, {max(p_values):.4f}]")
        logger.info(f"   P-values < 0.05: {sum(1 for p in p_values if p < 0.05)}")

        # 🎯 执行FDR correction
        try:
            q_values, significant = self.stats.fdr_correction(p_values, alpha)

            # 整合结果
            result_data = []
            for i, row in enumerate(mismatch_data):
                result_data.append({
                    **row,
                    'q_value': float(q_values[i]),
                    'fdr_significant': bool(significant[i])
                })

            # 筛选显著的
            significant_data = [r for r in result_data if r['fdr_significant']]

            logger.info(f"   ✅ FDR: {len(significant_data)}/{len(result_data)} significant (α={alpha})")

            if significant_data:
                top = significant_data[0]
                logger.info(f"   Top: {top['region1']}-{top['region2']}")
                logger.info(f"     Mismatch: {top['mismatch_combined']:.3f}")
                logger.info(f"     Q-value: {top['q_value']:.4f}")
            else:
                logger.warning(f"   No significant pairs after FDR correction")
                logger.warning(f"   Smallest q-value: {min(q_values):.4f}")
                logger.warning(f"   Consider: alpha={alpha} may be too stringent")

            return {
                'success': True,
                'data': significant_data,
                'rows': len(significant_data),
                'test_type': 'fdr',
                'alpha': alpha,
                'n_significant': len(significant_data),
                'n_total': len(result_data),
                'min_q_value': float(min(q_values)),
                'max_q_value': float(max(q_values))
            }

        except Exception as e:
            logger.error(f"   FDR correction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'data': []
            }

    def _correlation_test(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """Correlation test between modalities"""
        # 实现correlation (可选,暂时返回placeholder)
        logger.warning("Correlation test not yet implemented")
        return {'success': False, 'error': 'Not implemented'}

    # def _compute_mismatch_matrix(self, params: Dict, state: EnhancedAgentState) -> Dict:
    #     """
    #     计算cross-modal mismatch矩阵 (对齐Figure 4方法)
    #
    #     🎯 关键修复:
    #     1. 先计算所有pairs的距离矩阵
    #     2. 全局Min-Max归一化
    #     3. 然后计算mismatch
    #     """
    #     import time
    #     start_time = time.time()
    #
    #     # 获取regions
    #     regions = state.analysis_state.discovered_entities.get('Region', [])
    #
    #     if not regions:
    #         for key, data in state.intermediate_data.items():
    #             if data and isinstance(data, list) and len(data) > 0:
    #                 if 'region' in data[0]:
    #                     regions = list(set([row['region'] for row in data if row.get('region')]))
    #                     break
    #
    #     max_regions = params.get('max_regions', 15)
    #     regions = regions[:max_regions]
    #
    #     if len(regions) < 2:
    #         return {'success': False, 'error': 'Need at least 2 regions'}
    #
    #     n = len(regions)
    #     logger.info(f"   🚀 Computing mismatch (Figure 4 method) for {n} regions...")
    #
    #     # 🚀 Step 1: 批量获取fingerprints
    #     logger.info(f"   📊 Step 1/4: Batch fetching fingerprints...")
    #
    #     fingerprints = {}
    #     failed_regions = []
    #
    #     for region in regions:
    #         try:
    #             mol = self.fingerprint.compute_molecular_fingerprint(region)
    #             morph = self.fingerprint.compute_morphological_fingerprint(region)
    #             proj = self.fingerprint.compute_projection_fingerprint(region)
    #
    #             if mol is not None and morph is not None and proj is not None:
    #                 fingerprints[region] = {
    #                     'molecular': mol,
    #                     'morphological': morph,
    #                     'projection': proj
    #                 }
    #             else:
    #                 failed_regions.append(region)
    #
    #         except Exception as e:
    #             logger.warning(f"      Failed {region}: {e}")
    #             failed_regions.append(region)
    #
    #     valid_regions = [r for r in regions if r not in failed_regions]
    #     n_valid = len(valid_regions)
    #
    #     logger.info(f"      ✓ Got fingerprints: {len(fingerprints)}/{n}")
    #
    #     if n_valid < 2:
    #         return {'success': False, 'error': 'Insufficient valid regions'}
    #     # 🆕 Step 1.5: Z-score标准化形态指纹 (Figure 4方法)
    #     logger.info(f"   🔧 Step 1.5/4: Z-score standardization of morphology...")
    #     import numpy as np
    #     if len(fingerprints) >= 2:
    #         # 提取所有形态指纹
    #         all_morph = []
    #         for region in valid_regions:
    #             morph = fingerprints[region]['morphological']
    #             all_morph.append(morph)
    #
    #         all_morph = np.array(all_morph)  # (N_regions, 8)
    #
    #         logger.info(f"      Morphology array shape: {all_morph.shape}")
    #
    #         # 处理dendritic特征的0值 (索引4-7)
    #         dendritic_indices = [4, 5, 6, 7]
    #         for i in dendritic_indices:
    #             col = all_morph[:, i].copy()
    #             zero_mask = np.abs(col) < 1e-6
    #             n_zeros = zero_mask.sum()
    #             if n_zeros > 0:
    #                 logger.info(f"      Dendritic feature {i}: excluding {n_zeros}/{len(col)} zeros")
    #                 col[zero_mask] = np.nan
    #                 all_morph[:, i] = col
    #
    #         # 对每个特征维度进行z-score
    #         from scipy.stats import zscore
    #         for i in range(all_morph.shape[1]):
    #             col = all_morph[:, i]
    #             valid = ~np.isnan(col)
    #             if valid.sum() > 1:
    #                 col[valid] = zscore(col[valid])
    #                 all_morph[:, i] = col
    #
    #         # 更新fingerprints
    #         for idx, region in enumerate(valid_regions):
    #             fingerprints[region]['morphological'] = all_morph[idx]
    #
    #         logger.info(f"      ✓ Z-score standardization complete")
    #
    #     # 🚀 Step 2: 构建距离矩阵 (NxN)
    #     logger.info(f"   📏 Step 2/4: Building distance matrices...")
    #
    #     import numpy as np
    #     from scipy.spatial.distance import cosine, euclidean
    #
    #     mol_dist_matrix = np.zeros((n_valid, n_valid))
    #     morph_dist_matrix = np.zeros((n_valid, n_valid))
    #     proj_dist_matrix = np.zeros((n_valid, n_valid))
    #
    #     # 在Step 2: 构建距离矩阵中
    #     for i, region_a in enumerate(valid_regions):
    #         for j, region_b in enumerate(valid_regions):
    #             if i == j:
    #                 mol_dist_matrix[i, j] = 0
    #                 morph_dist_matrix[i, j] = 0
    #                 proj_dist_matrix[i, j] = 0
    #                 continue
    #
    #             fp_a = fingerprints[region_a]
    #             fp_b = fingerprints[region_b]
    #
    #             # 分子距离 (保持不变)
    #             try:
    #                 mol_dist_matrix[i, j] = cosine(fp_a['molecular'], fp_b['molecular'])
    #             except:
    #                 mol_dist_matrix[i, j] = np.nan
    #
    #             # 🔧 形态距离 (修复 - 使用Euclidean)
    #             try:
    #                 morph_a = fp_a['morphological']
    #                 morph_b = fp_b['morphological']
    #
    #                 # 检查NaN
    #                 valid_mask = ~(np.isnan(morph_a) | np.isnan(morph_b))
    #
    #                 if valid_mask.sum() >= 4:  # 至少4个有效维度
    #                     # 🎯 使用Euclidean距离（不是cosine）
    #                     morph_dist_matrix[i, j] = euclidean(
    #                         morph_a[valid_mask],
    #                         morph_b[valid_mask]
    #                     )
    #                 else:
    #                     morph_dist_matrix[i, j] = np.nan
    #             except Exception as e:
    #                 logger.debug(f"      Morph distance failed {region_a}-{region_b}: {e}")
    #                 morph_dist_matrix[i, j] = np.nan
    #
    #             # 投射距离 (保持不变)
    #             try:
    #                 proj_dist_matrix[i, j] = cosine(fp_a['projection'], fp_b['projection'])
    #             except:
    #                 proj_dist_matrix[i, j] = np.nan
    #
    #     print(f"      ✓ Distance matrices built")
    #     # 在 "✓ Distance matrices built" 后面添加
    #     print(
    #         f"      Molecular distance range: [{np.nanmin(mol_dist_matrix):.3f}, {np.nanmax(mol_dist_matrix):.3f}]")
    #     print(
    #         f"      Morphology distance range: [{np.nanmin(morph_dist_matrix):.3f}, {np.nanmax(morph_dist_matrix):.3f}]")
    #     print(
    #         f"      Projection distance range: [{np.nanmin(proj_dist_matrix):.3f}, {np.nanmax(proj_dist_matrix):.3f}]")
    #
    #     # 统计NaN数量
    #     n_total = mol_dist_matrix.size
    #     n_mol_nan = np.isnan(mol_dist_matrix).sum()
    #     n_morph_nan = np.isnan(morph_dist_matrix).sum()
    #     n_proj_nan = np.isnan(proj_dist_matrix).sum()
    #
    #     print(
    #         f"      NaN counts: mol={n_mol_nan}/{n_total}, morph={n_morph_nan}/{n_total}, proj={n_proj_nan}/{n_total}")
    #
    #     # 🚀 Step 3: Min-Max归一化 (全局)
    #     print(f"   🔧 Step 3/4: Normalizing distance matrices...")
    #
    #     def minmax_normalize(matrix):
    #         """Min-Max归一化到[0,1]"""
    #         valid = ~np.isnan(matrix)
    #         if valid.sum() == 0:
    #             return matrix
    #
    #         vmin = matrix[valid].min()
    #         vmax = matrix[valid].max()
    #
    #         if vmax - vmin < 1e-9:
    #             return np.zeros_like(matrix)
    #
    #         normalized = (matrix - vmin) / (vmax - vmin)
    #         return normalized
    #
    #     mol_norm = minmax_normalize(mol_dist_matrix)
    #     morph_norm = minmax_normalize(morph_dist_matrix)
    #     proj_norm = minmax_normalize(proj_dist_matrix)
    #
    #     print(f"      ✓ Normalization complete")
    #     print(f"      Normalized molecular range: [{np.nanmin(mol_norm):.3f}, {np.nanmax(mol_norm):.3f}]")
    #     print(f"      Normalized morphology range: [{np.nanmin(morph_norm):.3f}, {np.nanmax(morph_norm):.3f}]")
    #     print(f"      Normalized projection range: [{np.nanmin(proj_norm):.3f}, {np.nanmax(proj_norm):.3f}]")
    #
    #     # 🚀 Step 4: 计算Mismatch (归一化距离的差异)
    #     print(f"   🧮 Step 4/4: Computing mismatches...")
    #
    #     mismatch_results = []
    #
    #     from itertools import combinations
    #
    #     for i, region1 in enumerate(valid_regions):
    #         for j, region2 in enumerate(valid_regions):
    #             if i >= j:  # 只计算上三角
    #                 continue
    #
    #             # Mismatch = |normalized_distance_A - normalized_distance_B|
    #             mismatch_GM = abs(mol_norm[i, j] - morph_norm[i, j])
    #             mismatch_GP = abs(mol_norm[i, j] - proj_norm[i, j])
    #             mismatch_MP = abs(morph_norm[i, j] - proj_norm[i, j])
    #
    #             mismatch_combined = (mismatch_GM + mismatch_GP + mismatch_MP) / 3
    #
    #             # 相似度 (用于报告)
    #             sim_molecular = 1 - mol_dist_matrix[i, j]
    #             sim_morphological = 1 - morph_norm[i, j]  # 归一化后的
    #             sim_projection = 1 - proj_dist_matrix[i, j]
    #
    #             mismatch_results.append({
    #                 'region1': region1,
    #                 'region2': region2,
    #                 'mismatch_GM': float(mismatch_GM),
    #                 'mismatch_GP': float(mismatch_GP),
    #                 'mismatch_MP': float(mismatch_MP),
    #                 'mismatch_combined': float(mismatch_combined),
    #                 'sim_molecular': float(sim_molecular),
    #                 'sim_morphological': float(sim_morphological),
    #                 'sim_projection': float(sim_projection),
    #                 # 距离值 (调试用)
    #                 'dist_molecular': float(mol_dist_matrix[i, j]),
    #                 'dist_morphological': float(morph_dist_matrix[i, j]),
    #                 'dist_projection': float(proj_dist_matrix[i, j])
    #             })
    #
    #     # 统计检验
    #     all_mismatches = [r['mismatch_combined'] for r in mismatch_results]
    #     mean_m = np.mean(all_mismatches)
    #     std_m = np.std(all_mismatches)
    #
    #     for result in mismatch_results:
    #         m = result['mismatch_combined']
    #
    #         if std_m > 0:
    #             z_score = (m - mean_m) / std_m
    #             from scipy import stats
    #             p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    #         else:
    #             z_score = 0
    #             p_value = 1.0
    #
    #         result['p_value'] = float(p_value)
    #         result['z_score'] = float(z_score)
    #         result['effect_size'] = float(m)
    #         result['n_permutations'] = 0
    #
    #     mismatch_results.sort(key=lambda x: x['mismatch_combined'], reverse=True)
    #
    #     elapsed = time.time() - start_time
    #
    #     print(f"   ✅ Completed in {elapsed:.1f}s")
    #     print(f"      Total pairs: {len(mismatch_results)}")
    #
    #     if mismatch_results:
    #         top = mismatch_results[0]
    #         print(f"      Top: {top['region1']}-{top['region2']}")
    #         print(f"        Mismatch: {top['mismatch_combined']:.3f}")
    #         print(f"        P-value: {top['p_value']:.4f}")
    #
    #         # 🔍 显示top 5用于验证
    #         print(f"      Top 5 pairs:")
    #         for i, pair in enumerate(mismatch_results[:5], 1):
    #             print(f"        {i}. {pair['region1']}-{pair['region2']}: {pair['mismatch_combined']:.3f}")
    #
    #     return {
    #         'success': True,
    #         'data': mismatch_results,
    #         'rows': len(mismatch_results),
    #         'analysis_type': 'cross_modal_mismatch',
    #         'computation_time': elapsed,
    #         'method': 'figure4_compatible'
    #     }
    def _compute_mismatch_matrix(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """
        计算cross-modal mismatch矩阵 (完全对齐Ground Truth result4.py)

        🔧 关键修复：
        1. Step 0: 全局形态标准化（一次性，所有regions）
        2. Step 1: 使用缓存的标准化数据
        3. Step 2: 构建距离矩阵（形态距离用Euclidean）
        4. Step 3: Min-Max归一化
        5. Step 4: 计算Mismatch和相似度（统一用归一化距离）
        """
        import time
        start_time = time.time()

        # 获取regions
        regions = state.analysis_state.discovered_entities.get('Region', [])

        if not regions:
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    if 'region' in data[0]:
                        regions = list(set([row['region'] for row in data if row.get('region')]))
                        break

        max_regions = params.get('max_regions', 30)
        regions = regions[:max_regions]

        if len(regions) < 2:
            return {'success': False, 'error': 'Need at least 2 regions'}

        n = len(regions)
        logger.info(f"   🚀 Computing mismatch (Figure 4 method) for {n} regions...")

        # 🔧 Step 0: 全局形态标准化（只做一次）
        logger.info(f"   🔧 Step 0/4: Global morphology standardization...")
        if not hasattr(self.fingerprint, '_morph_cache'):
            self.fingerprint.standardize_morphology_globally(regions)

        # 🚀 Step 1: 批量获取fingerprints
        logger.info(f"   📊 Step 1/4: Batch fetching fingerprints...")

        fingerprints = {}
        failed_regions = []

        for region in regions:
            try:
                mol = self.fingerprint.compute_molecular_fingerprint(region)
                # 🔧 使用缓存的全局标准化形态数据
                morph = self.fingerprint._morph_cache.get(region)
                proj = self.fingerprint.compute_projection_fingerprint(region)

                if mol is not None and morph is not None and proj is not None:
                    fingerprints[region] = {
                        'molecular': mol,
                        'morphological': morph,  # 已经是z-scored的
                        'projection': proj
                    }
                else:
                    failed_regions.append(region)

            except Exception as e:
                logger.warning(f"      Failed {region}: {e}")
                failed_regions.append(region)

        valid_regions = [r for r in regions if r not in failed_regions]
        n_valid = len(valid_regions)

        logger.info(f"      ✓ Got fingerprints: {len(fingerprints)}/{n}")

        if n_valid < 2:
            return {'success': False, 'error': 'Insufficient valid regions'}

        # 🚀 Step 2: 构建距离矩阵 (NxN)
        logger.info(f"   📏 Step 2/4: Building distance matrices...")

        import numpy as np
        from scipy.spatial.distance import cosine, euclidean

        mol_dist_matrix = np.zeros((n_valid, n_valid))
        morph_dist_matrix = np.zeros((n_valid, n_valid))
        proj_dist_matrix = np.zeros((n_valid, n_valid))

        for i, region_a in enumerate(valid_regions):
            for j, region_b in enumerate(valid_regions):
                if i == j:
                    mol_dist_matrix[i, j] = 0
                    morph_dist_matrix[i, j] = 0
                    proj_dist_matrix[i, j] = 0
                    continue

                fp_a = fingerprints[region_a]
                fp_b = fingerprints[region_b]

                # 分子距离 (cosine)
                try:
                    mol_dist_matrix[i, j] = cosine(fp_a['molecular'], fp_b['molecular'])
                except:
                    mol_dist_matrix[i, j] = np.nan

                # 🔧 形态距离 (Euclidean on z-scored features)
                try:
                    morph_a = fp_a['morphological']
                    morph_b = fp_b['morphological']

                    # 检查NaN
                    valid_mask = ~(np.isnan(morph_a) | np.isnan(morph_b))

                    if valid_mask.sum() >= 4:  # 至少4个有效维度
                        morph_dist_matrix[i, j] = euclidean(
                            morph_a[valid_mask],
                            morph_b[valid_mask]
                        )
                    else:
                        morph_dist_matrix[i, j] = np.nan
                except Exception as e:
                    logger.debug(f"      Morph distance failed {region_a}-{region_b}: {e}")
                    morph_dist_matrix[i, j] = np.nan

                # 投射距离 (cosine)
                try:
                    proj_dist_matrix[i, j] = cosine(fp_a['projection'], fp_b['projection'])
                except:
                    proj_dist_matrix[i, j] = np.nan

        logger.info(f"      ✓ Distance matrices built")
        logger.info(
            f"      Molecular distance range: [{np.nanmin(mol_dist_matrix):.3f}, {np.nanmax(mol_dist_matrix):.3f}]")
        logger.info(
            f"      Morphology distance range: [{np.nanmin(morph_dist_matrix):.3f}, {np.nanmax(morph_dist_matrix):.3f}]")
        logger.info(
            f"      Projection distance range: [{np.nanmin(proj_dist_matrix):.3f}, {np.nanmax(proj_dist_matrix):.3f}]")

        # 统计NaN数量
        n_total = mol_dist_matrix.size
        n_mol_nan = np.isnan(mol_dist_matrix).sum()
        n_morph_nan = np.isnan(morph_dist_matrix).sum()
        n_proj_nan = np.isnan(proj_dist_matrix).sum()

        logger.info(
            f"      NaN counts: mol={n_mol_nan}/{n_total}, morph={n_morph_nan}/{n_total}, proj={n_proj_nan}/{n_total}")

        # 🚀 Step 3: Min-Max归一化 (全局)
        logger.info(f"   🔧 Step 3/4: Normalizing distance matrices...")

        def minmax_normalize(matrix):
            """Min-Max归一化到[0,1]"""
            valid = ~np.isnan(matrix)
            if valid.sum() == 0:
                return matrix

            vmin = matrix[valid].min()
            vmax = matrix[valid].max()

            if vmax - vmin < 1e-9:
                return np.zeros_like(matrix)

            normalized = (matrix - vmin) / (vmax - vmin)
            return normalized

        mol_norm = minmax_normalize(mol_dist_matrix)
        morph_norm = minmax_normalize(morph_dist_matrix)
        proj_norm = minmax_normalize(proj_dist_matrix)

        logger.info(f"      ✓ Normalization complete")
        logger.info(f"      Normalized molecular range: [{np.nanmin(mol_norm):.3f}, {np.nanmax(mol_norm):.3f}]")
        logger.info(f"      Normalized morphology range: [{np.nanmin(morph_norm):.3f}, {np.nanmax(morph_norm):.3f}]")
        logger.info(f"      Normalized projection range: [{np.nanmin(proj_norm):.3f}, {np.nanmax(proj_norm):.3f}]")

        # 🚀 Step 4: 计算Mismatch (归一化距离的差异)
        logger.info(f"   🧮 Step 4/4: Computing mismatches...")

        mismatch_results = []

        for i, region1 in enumerate(valid_regions):
            for j, region2 in enumerate(valid_regions):
                if i >= j:  # 只计算上三角
                    continue

                # Mismatch = |normalized_distance_A - normalized_distance_B|
                mismatch_GM = abs(mol_norm[i, j] - morph_norm[i, j])
                mismatch_GP = abs(mol_norm[i, j] - proj_norm[i, j])
                mismatch_MP = abs(morph_norm[i, j] - proj_norm[i, j])

                mismatch_combined = (mismatch_GM + mismatch_GP + mismatch_MP) / 3

                # 🔧 Fix: 相似度统一使用归一化距离
                sim_molecular = 1 - mol_norm[i, j]  # ← 修复：用归一化的
                sim_morphological = 1 - morph_norm[i, j]  # ← 已经对的
                sim_projection = 1 - proj_norm[i, j]  # ← 修复：用归一化的

                mismatch_results.append({
                    'region1': region1,
                    'region2': region2,
                    'mismatch_GM': float(mismatch_GM),
                    'mismatch_GP': float(mismatch_GP),
                    'mismatch_MP': float(mismatch_MP),
                    'mismatch_combined': float(mismatch_combined),
                    'sim_molecular': float(sim_molecular),
                    'sim_morphological': float(sim_morphological),
                    'sim_projection': float(sim_projection),
                    # 归一化距离（调试用）
                    'dist_molecular_norm': float(mol_norm[i, j]),
                    'dist_morphological_norm': float(morph_norm[i, j]),
                    'dist_projection_norm': float(proj_norm[i, j]),
                })

        # 统计检验
        all_mismatches = [r['mismatch_combined'] for r in mismatch_results]
        mean_m = np.mean(all_mismatches)
        std_m = np.std(all_mismatches)

        for result in mismatch_results:
            m = result['mismatch_combined']

            if std_m > 0:
                z_score = (m - mean_m) / std_m
                from scipy import stats
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0

            result['p_value'] = float(p_value)
            result['z_score'] = float(z_score)
            result['effect_size'] = float(m)
            result['n_permutations'] = 0

        mismatch_results.sort(key=lambda x: x['mismatch_combined'], reverse=True)

        elapsed = time.time() - start_time

        logger.info(f"   ✅ Completed in {elapsed:.1f}s")
        logger.info(f"      Total pairs: {len(mismatch_results)}")

        if mismatch_results:
            top = mismatch_results[0]
            logger.info(f"      Top: {top['region1']}-{top['region2']}")
            logger.info(f"        Mismatch: {top['mismatch_combined']:.3f}")
            logger.info(f"        P-value: {top['p_value']:.4f}")

            # 🔍 显示top 5用于验证
            logger.info(f"      Top 5 pairs:")
            for i, pair in enumerate(mismatch_results[:5], 1):
                logger.info(f"        {i}. {pair['region1']}-{pair['region2']}: {pair['mismatch_combined']:.3f}")

        return {
            'success': True,
            'data': mismatch_results,
            'rows': len(mismatch_results),
            'analysis_type': 'cross_modal_mismatch',
            'computation_time': elapsed,
            'method': 'figure4_fully_aligned'
        }

    def _compute_cosine_similarity(self, vec1, vec2):
        """
        快速计算余弦相似度

        🚀 优化: 使用NumPy向量化操作
        """
        import numpy as np

        if not vec1 or not vec2:
            return 0.0

        # 转换为NumPy数组
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)

        # 确保长度一致
        if len(v1) != len(v2):
            # Pad或truncate
            max_len = max(len(v1), len(v2))
            if len(v1) < max_len:
                v1 = np.pad(v1, (0, max_len - len(v1)))
            if len(v2) < max_len:
                v2 = np.pad(v2, (0, max_len - len(v2)))

        # 余弦相似度
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _interpret_statistical_result(self, test_result: Dict, effect_size: float) -> str:
        """解释统计结果"""
        p_value = test_result['p_value']

        if p_value < 0.001:
            sig_level = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            sig_level = "very significant (p < 0.01)"
        elif p_value < 0.05:
            sig_level = "significant (p < 0.05)"
        else:
            sig_level = "not significant (p ≥ 0.05)"

        if abs(effect_size) > 0.8:
            effect_desc = "large effect size"
        elif abs(effect_size) > 0.5:
            effect_desc = "medium effect size"
        elif abs(effect_size) > 0.2:
            effect_desc = "small effect size"
        else:
            effect_desc = "negligible effect size"

        return f"The difference is {sig_level} with a {effect_desc} (Cohen's d = {effect_size:.2f})"

    def _resolve_parameters(self,
                            step: ReasoningStep,
                            state: EnhancedAgentState,
                            params: Dict) -> Dict:
        """解析步骤依赖的参数"""
        resolved = params.copy()

        # 查找依赖步骤的数据
        for dep_num in step.depends_on:
            dep_key = f"step_{dep_num}"
            if dep_key in state.intermediate_data:
                dep_data = state.intermediate_data[dep_key]

                if not dep_data:
                    continue

                # 提取常用字段
                # 提取region acronyms
                regions = []
                for row in dep_data:
                    if 'region' in row:
                        regions.append(row['region'])
                    elif 'acronym' in row:
                        regions.append(row['acronym'])

                if regions:
                    resolved['enriched_regions'] = list(set(regions))[:10]
                    resolved['target_regions'] = list(set(regions))[:10]

                # 提取targets
                targets = []
                for row in dep_data:
                    if 'target' in row:
                        targets.append(row['target'])
                    elif 'target_region' in row:
                        targets.append(row['target_region'])

                if targets:
                    resolved['targets'] = list(set(targets))[:10]

        return resolved

    # ==================== Intelligent Replanning ====================

    def _intelligent_replan(self, state: EnhancedAgentState, from_step: int) -> bool:
        """
        智能重规划

        使用:
        - 结构化反思的建议
        - 替代假设
        - Schema中的替代路径
        """
        logger.info(f"🔄 Intelligent replanning from step {from_step}")
        state.replanning_count += 1

        # 获取最近的结构化反思
        if state.structured_reflections:
            last_reflection = state.structured_reflections[-1]

            # 使用反思中的建议
            logger.info(f"   Using reflection recommendations:")
            for rec in last_reflection.next_step_recommendations:
                logger.info(f"     • {rec}")

            # 如果有替代假设,尝试使用
            if last_reflection.alternative_hypotheses:
                logger.info(f"   Found {len(last_reflection.alternative_hypotheses)} alternative hypotheses")

        # 重新生成计划 (使用现有实体)
        try:
            query_plans = self.path_planner.generate_plan(
                state.entity_clusters,
                state.question
            )

            # 替换剩余步骤
            new_steps = self._llm_refine_plans(query_plans, state)

            # 更新plan,保留已执行的
            state.reasoning_plan = state.reasoning_plan[:from_step - 1] + new_steps

            logger.info(f"   ✅ Replanned with {len(new_steps)} new steps")
            return True

        except Exception as e:
            logger.error(f"   ❌ Replanning failed: {e}")
            return False

    # ==================== Answer Synthesis ====================

    def _synthesize_answer(self, state: EnhancedAgentState) -> str:
        """
        合成最终答案 (增强版 - 科学叙事)
        """
        # 准备证据摘要
        evidence = []
        for step in state.executed_steps:
            if step.actual_result and step.actual_result.get('success'):
                data_count = len(step.actual_result.get('data', []))
                evidence.append(f"- Step {step.step_number}: {step.purpose} ({data_count} results)")

        evidence_text = "\n".join(evidence)

        # 准备关键发现
        key_data = {}
        for step in state.executed_steps:
            if step.actual_result and step.actual_result.get('success'):
                data = step.actual_result.get('data', [])
                if data:
                    key_data[f"step_{step.step_number}"] = data[:5]  # Top 5

        # 准备结构化反思摘要
        reflection_summary = []
        for r in state.structured_reflections:
            reflection_summary.append(
                f"Step {r.step_number}: {r.validation_status.value} (confidence: {r.confidence_score:.2f})"
            )

        # 🆕 检测分析类型
        analysis_type = self._detect_analysis_type(state)

        # 🆕 准备PRIMARY FOCUS信息
        primary_focus_info = ""
        if hasattr(state.analysis_state, 'primary_focus') and state.analysis_state.primary_focus:
            focus = state.analysis_state.primary_focus
            supporting = focus.supporting_data
            primary_focus_info = f"""
    **PRIMARY FOCUS IDENTIFIED:**
    - Region: {focus.entity_id}
    - Enrichment: {supporting.get('total_neurons', 'N/A')} neurons across {supporting.get('cluster_count', 'N/A')} clusters
    - This region shows the highest enrichment and was selected for deep characterization
    """

        prompt = f"""Synthesize a comprehensive, publication-quality answer based on the multi-step analysis.

    **CRITICAL: Write as a SCIENTIFIC NARRATIVE, not a data report!**

    **Original Question:** {state.question}

    **Analysis Type Detected:** {analysis_type}

    **Entities Recognized:** {', '.join([e['text'] for e in state.entities[:5]])}

    {primary_focus_info}

    **Reasoning Steps Executed:**
    {chr(10).join([f"{i + 1}. {s.purpose}" for i, s in enumerate(state.executed_steps)])}

    **Evidence Collected:**
    {evidence_text}

    **Key Findings (quantitative data):**
    {json.dumps(key_data, indent=2, default=str)[:3000]}

    **Structured Reflections:**
    {chr(10).join(reflection_summary)}

    **Your Task:**

    Write a comprehensive answer with the following structure:

    ### [Title - Generate an engaging title]

    #### Introduction (1 paragraph)
    - Open with the biological significance
    - State the main finding concisely

    #### Multi-Modal Analysis Results

    **1. Molecular Characterization**
    - Cite SPECIFIC numbers (e.g., "18,474 neurons across 4 clusters")
    - Mention key markers and cell types
    - Use quantitative language

    **2. Spatial Distribution**
    - List regions with enrichment metrics
    - Highlight PRIMARY focus if identified
    - Use percentages and rankings

    **3. Morphological Features** (if available)
    - Report mean ± SD for axonal/dendritic measurements
    - Compare to baseline if applicable
    - Interpret structural specializations

    **4. Connectivity Patterns** (if available)
    - Describe projection targets with weights
    - Categorize by functional systems (sensory/motor/associative)
    - Mention top 3-5 targets quantitatively

    **5. Target Characterization (CLOSED LOOP)** (if available)
    - Describe cell type composition of projection targets
    - Connect back to molecular findings
    - Emphasize circuit-level integration

    **6. Statistical Validation** (if available)
    - Report p-values and effect sizes
    - Mention significance levels
    - Interpret biological meaning

    #### Integration and Implications
    - Connect molecular → morphological → projection findings
    - Propose functional hypotheses
    - Discuss circuit-level organization

    #### Limitations and Uncertainties
    - Acknowledge data gaps honestly
    - Cite confidence scores from reflections
    - Suggest validation approaches

    **Writing Style:**
    - Use ACTIVE voice ("Our analysis revealed..." not "It was found...")
    - Connect findings CAUSALLY ("Because X, we examined Y, which revealed Z")
    - Emphasize QUANTITATIVE data (numbers, percentages, statistics)
    - Make it VISUAL-READY (structure data for plotting)
    - Be HONEST about uncertainties

    **Avoid:**
    - Lists without narrative flow
    - Vague statements ("some regions", "several")
    - Overconfident claims
    - Jargon without explanation

    Generate a publication-quality narrative now.
    """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a neuroscience writer synthesizing research analysis results into publication-quality narratives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            answer = response.choices[0].message.content.strip()
            state.final_answer = answer

            # 估算置信度
            state.confidence_score = self._estimate_confidence(state)

            return answer

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")

            # Enhanced fallback: Generate structured answer from collected data
            return self._generate_fallback_answer(state)

    def _generate_fallback_answer(self, state: EnhancedAgentState) -> str:
        """
        Generate a structured answer from collected data when LLM synthesis fails.
        This provides useful output even without OpenAI API access.
        """
        lines = ["## Analysis Results (Auto-Generated Summary)", ""]

        # Question
        lines.append(f"**Question:** {state.question}")
        lines.append("")

        # Entities found
        if state.entities:
            lines.append("### Entities Identified")
            for e in state.entities[:5]:
                lines.append(f"- **{e.get('text', 'Unknown')}** ({e.get('type', 'Unknown')})")
            lines.append("")

        # Primary focus if identified
        if hasattr(state, 'analysis_state') and hasattr(state.analysis_state, 'primary_focus'):
            focus = state.analysis_state.primary_focus
            if focus:
                lines.append("### Primary Focus Region")
                lines.append(f"- **Region:** {focus.entity_id}")
                if focus.supporting_data:
                    sd = focus.supporting_data
                    lines.append(f"- **Total Neurons:** {sd.get('total_neurons', 'N/A'):,}")
                    lines.append(f"- **Cluster Count:** {sd.get('cluster_count', 'N/A')}")
                lines.append("")

        # Step results
        if state.executed_steps:
            lines.append("### Analysis Steps Completed")
            for step in state.executed_steps:
                result = step.actual_result or {}
                success = result.get('success', False)
                row_count = result.get('rows', 0)
                status = "✓" if success else "✗"
                lines.append(f"{status} **Step {step.step_number}:** {step.purpose}")
                if success and row_count > 0:
                    lines.append(f"  - Retrieved {row_count} data points")

                    # Extract key findings
                    data = result.get('data', [])
                    if data and isinstance(data, list) and len(data) > 0:
                        first_row = data[0]
                        if isinstance(first_row, dict):
                            # Show top results based on step type
                            if 'region' in first_row or 'acronym' in first_row:
                                regions = [r.get('region') or r.get('acronym') for r in data[:5]]
                                lines.append(f"  - Top regions: {', '.join(filter(None, regions))}")
                            if 'target' in first_row:
                                targets = [r.get('target') for r in data[:5]]
                                lines.append(f"  - Top targets: {', '.join(filter(None, targets))}")
                            if 'cluster' in first_row:
                                clusters = [r.get('cluster') for r in data[:3]]
                                lines.append(f"  - Clusters: {', '.join(filter(None, clusters))}")
                lines.append("")

        # Modalities covered
        if hasattr(state, 'analysis_state') and state.analysis_state.modalities_covered:
            lines.append("### Modalities Analyzed")
            for mod in state.analysis_state.modalities_covered:
                lines.append(f"- {mod.capitalize()}")
            lines.append("")

        # Discovered entities
        if hasattr(state, 'analysis_state') and state.analysis_state.discovered_entities:
            lines.append("### Entities Discovered During Analysis")
            for etype, entities in state.analysis_state.discovered_entities.items():
                if entities:
                    lines.append(f"- **{etype}:** {len(entities)} found")
                    if len(entities) <= 5:
                        lines.append(f"  - {', '.join(str(e) for e in entities)}")
            lines.append("")

        # Summary
        lines.append("### Summary")
        lines.append(f"- **Steps Executed:** {len(state.executed_steps)}")
        lines.append(f"- **Modalities:** {', '.join(state.analysis_state.modalities_covered) if hasattr(state, 'analysis_state') else 'N/A'}")

        state.confidence_score = self._estimate_confidence(state)
        lines.append(f"- **Confidence:** {state.confidence_score:.2f}")
        lines.append("")
        lines.append("*Note: This is an auto-generated summary. LLM-based narrative synthesis was unavailable.*")

        return "\n".join(lines)

    def _detect_analysis_type(self, state: EnhancedAgentState) -> str:
        """检测分析类型"""
        step_purposes = [s.purpose.lower() for s in state.executed_steps]

        if any('compare' in p or 'versus' in p for p in step_purposes):
            return "Comparative Analysis"
        elif any('mismatch' in p or 'screening' in p for p in step_purposes):
            return "Systematic Screening (Figure 4 type)"
        elif any('primary focus' in p or 'closed loop' in p for p in step_purposes):
            return "Focus-Driven Deep Analysis (Figure 3 type)"
        else:
            return "General Multi-Modal Analysis"

    # ==================== Utilities ====================

    def _step_to_dict(self, step: ReasoningStep) -> Dict:
        """转换步骤为字典 (修复版 - 保留完整actual_result)"""
        step_dict = {
            'step_number': step.step_number,
            'purpose': step.purpose,
            'action': step.action,
            'rationale': step.rationale,
            'expected_result': step.expected_result,
            'actual_result_summary': {
                'success': step.actual_result.get('success') if step.actual_result else False,
                'row_count': len(step.actual_result.get('data', [])) if step.actual_result else 0
            },
            'reflection': step.reflection,
            'validation_passed': step.validation_passed,
            'execution_time': step.execution_time,
            'modality': step.modality
        }

        # 🔧 关键修复：保留完整的actual_result用于绘图
        if step.actual_result:
            step_dict['actual_result'] = step.actual_result

        return step_dict

    # aipom_v10_production.py, Line ~1160
    def _estimate_confidence(self, state: EnhancedAgentState) -> float:
        """估算置信度（优化版）"""

        if not state.structured_reflections:
            return 0.5

        # 使用结构化反思的置信度
        confidences = [r.confidence_score for r in state.structured_reflections]
        avg_confidence = sum(confidences) / len(confidences)

        # 🔧 修复：不惩罚步骤少于计划
        # 因为adaptive可能合理地提前终止
        if state.reasoning_plan:
            completion_rate = len(state.executed_steps) / len(state.reasoning_plan)
        else:
            completion_rate = 1.0  # 没有计划，认为是完成的

        # 🔧 放宽completion_rate的影响
        completion_factor = 0.85 + 0.15 * completion_rate  # 原来是 0.7 + 0.3 * rate

        # Factor 2: 重规划惩罚
        replan_penalty = 0.95 ** state.replanning_count

        # 综合
        final_confidence = avg_confidence * completion_factor * replan_penalty

        return min(1.0, max(0.0, final_confidence))

    def _build_error_response(self, question: str, error: str, start_time: float) -> Dict:
        """构建错误响应"""
        return {
            'question': question,
            'answer': f"Analysis failed: {error}",
            'error': error,
            'execution_time': time.time() - start_time,
            'success': False,
            'entities_recognized': [],
            'reasoning_plan': [],
            'executed_steps': [],
            'reflections': [],
            'confidence_score': 0.0
        }

    def close(self):
        """关闭数据库连接"""
        self.db.close()


# ==================== Test ====================

def test_v10_agent():
    """测试V10 agent"""
    import os

    print("\n" + "=" * 80)
    print("AIPOM-CoT V10 PRODUCTION TEST")
    print("=" * 80)

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY", ''),
        model="gpt-4o"
    )

    # 测试问题
    test_questions = [
        "Tell me about Car3+ neurons",
        "Compare Pvalb and Sst interneurons in MOs",
        "What are the projection targets of the claustrum?"
    ]

    for q in test_questions:
        print(f"\n{'=' * 80}")
        print(f"Q: {q}")
        print('=' * 80)

        result = agent.answer(q, max_iterations=8)

        print(f"\n✅ Results:")
        print(f"   Entities: {len(result['entities_recognized'])}")
        print(f"   Steps: {result['total_steps']}")
        print(f"   Confidence: {result['confidence_score']:.3f}")
        print(f"   Time: {result['execution_time']:.2f}s")
        print(f"\n💡 Answer:\n{result['answer'][:300]}...\n")

    agent.close()


def test_car3_comprehensive():
    """测试Car3的完整分析"""

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY", ''),
        model="gpt-4o"
    )

    # 🎯 关键: 使用"comprehensive"触发深度分析
    # question = "Give me a comprehensive analysis of Car3+ neurons"
    question = "Which brain region pairs show the highest cross-modal mismatch between molecular fingerprints, morphological features, and projection patterns among the top 30 brain regions with most neurons?"
    # result = agent.answer(question, max_iterations=12)
    result = agent.answer_with_visualization(
        question,
        max_iterations=10,
        generate_plots=True,
        output_dir='./figure4_automatic_output'
    )
    print("\n" + "=" * 80)
    print("FIGURE 3 STORY ARC ANALYSIS")
    print("=" * 80)

    print(f"\nTarget Depth: {result['adaptive_planning']['target_depth']}")
    print(f"Steps Executed: {result['adaptive_planning']['final_depth']}")
    print(f"Modalities: {', '.join(result['adaptive_planning']['modalities_covered'])}")

    print("\n" + "-" * 80)
    print("STEP-BY-STEP NARRATIVE:")
    print("-" * 80)

    for i, step in enumerate(result['executed_steps'], 1):
        print(f"\n{i}. {step['purpose']}")
        print(f"   Modality: {step['modality']}")
        print(f"   Data: {step['actual_result_summary']['row_count']} rows")
        print(f"   Confidence: {step['reflection']}")

    print("\n" + "-" * 80)
    print("ENTITIES DISCOVERED:")
    print("-" * 80)
    for entity_type, count in result['adaptive_planning']['entities_discovered'].items():
        print(f"  • {entity_type}: {count}")

    print("\n" + "-" * 80)
    print("VALIDATION CHECKLIST:")
    print("-" * 80)

    modalities = result['adaptive_planning']['modalities_covered']
    entities = result['adaptive_planning']['entities_discovered']

    checks = {
        'Has molecular analysis': 'molecular' in modalities,
        'Has morphological analysis': 'morphological' in modalities,
        'Has projection analysis': 'projection' in modalities,
        'Found regions': 'Region' in entities and entities['Region'] > 0,
        'Found projection targets': 'ProjectionTarget' in entities and entities['ProjectionTarget'] > 0,
        'Analyzed target composition': any(
            'target' in s['purpose'].lower() and 'composition' in s['purpose'].lower() for s in
            result['executed_steps'])
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    # 计算完整性分数
    completeness = sum(checks.values()) / len(checks) * 100
    print(f"\n📊 Story Completeness: {completeness:.0f}%")

    if completeness >= 80:
        print("\n🎉 ✅ FIGURE 3 COMPLETE STORY ARC ACHIEVED!")
    else:
        print(f"\n⚠️  Story incomplete - missing {100 - completeness:.0f}% of elements")

    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result['answer'])

    agent.close()

    return result

# ==================== CLI Interface (CC_SPEC_MS) ====================

def create_agent_from_args(args) -> 'AIPOMCoTV10':
    """Create agent with env var priority for Neo4j connection"""
    return AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY", ''),
        model="gpt-4o"
    )


def run_chat_mode(args):
    """
    Chat mode with intent gating.
    SMALLTALK triggers zero KG queries; other intents are budgeted.
    """
    router = IntentRouter()
    query = args.query or input("Enter your query: ")

    # Classify intent
    intent = router.classify(query)
    logger.info(f"Intent classified: {intent.value}")

    # Handle SMALLTALK without KG queries
    if intent == IntentType.SMALLTALK:
        response = get_smalltalk_response(query)
        print(f"\n{response}")
        return {'intent': intent.value, 'response': response, 'kg_queries': 0}

    # Get budget limits
    budget_limits = get_budget_for_intent(intent, args.budget)
    logger.info(f"Budget limits: queries={budget_limits.max_kg_queries}, rows={budget_limits.row_limit}")

    # Initialize provenance
    prov = create_provenance_logger(run_id=f"chat_{int(time.time())}", seed=args.seed)
    prov.log_run_start(mode='chat', intent=intent.value, query=query, budget=args.budget)

    # Create agent and run
    agent = create_agent_from_args(args)

    try:
        # Set determinism
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Run with budget-aware max iterations
        result = agent.answer(query, max_iterations=budget_limits.max_plan_steps or 6)

        prov.log_run_end(
            termination_reason='completed',
            total_steps=result.get('total_steps', 0),
            total_kg_queries=len(result.get('executed_steps', [])),
            execution_time=result.get('execution_time', 0),
            success=True
        )

        print(f"\n{'='*80}")
        print("ANSWER:")
        print('='*80)
        print(result['answer'])

        return result
    finally:
        agent.close()


def _write_fail_fast_report(output_dir: Path, case_name: str, result: Dict,
                             evidence: EvidenceBuffer, prov: ProvenanceLogger, seed: int):
    """Write FAILED report with full diagnostics when evidence is missing.

    Called when kg_query_count==0, evidence_coverage==0, or execution errors occur.
    Never outputs numeric narrative without evidence — writes diagnostics instead.
    """
    lines = [
        f"# MS {case_name.upper()} Report — FAILED",
        "",
        f"**Status:** FAILED — No KG evidence collected",
        f"**Seed:** {seed}",
        f"**KG Queries:** {evidence.get_kg_query_count()}",
        f"**Evidence Coverage:** {evidence.get_coverage_rate():.0%}",
        "",
        "## Diagnostics",
        "",
        f"- Error: {result.get('error', 'Unknown')}",
        f"- Neo4j connected: {result.get('neo4j_status', {}).get('connected', 'Unknown')}",
        f"- Neo4j error: {result.get('neo4j_status', {}).get('error', 'N/A')}",
        "",
        "## Evidence Buffer State",
        "",
        evidence.to_markdown(),
        "",
        "## Traceback",
        "",
        "```",
        result.get('traceback', 'No traceback available'),
        "```",
    ]
    report_path = output_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # Also log to provenance
    prov.log_reflect(
        step_number=0,
        validation_status='fail_fast',
        confidence=0.0,
        should_replan=False,
        recommendations=[f'FAIL-FAST: {result.get("error", "No evidence")}']
    )


def run_analysis_mode(args):
    """
    Scientific analysis mode with deterministic execution and provenance tracing.
    Available analyses: reasoning, fingerprint, circuit.
    """
    if not args.analysis:
        print("Error: --analysis is required for analysis mode (reasoning, fingerprint, or circuit)")
        return None

    # Set determinism
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    output_base = Path(f"./outputs/{args.analysis}_analysis")
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "data").mkdir(exist_ok=True)

    # Initialize provenance
    prov = create_provenance_logger(run_id=f"{args.analysis}_analysis", seed=args.seed)
    prov.log_run_start(
        mode='analysis',
        intent='SCIENTIFIC_ANALYSIS',
        query=f"Analysis: {args.analysis}",
        budget='heavy',
        snapshot_id=args.snapshot_id
    )

    if args.analysis == 'reasoning':
        result = run_reasoning_demo(args, output_base, prov)
    elif args.analysis == 'fingerprint':
        result = run_fingerprint_analysis(args, output_base, prov)
    elif args.analysis == 'circuit':
        result = run_circuit_discovery(args, output_base, prov)
    else:
        print(f"Unknown analysis type: {args.analysis}")
        return None

    print(f"\nOutputs saved to: {output_base}")
    print(f"Provenance trace: {prov.get_trace_path()}")

    return result


def run_reasoning_demo(args, output_base: Path, prov: ProvenanceLogger) -> Dict:
    """
    Reasoning Demo: AIPOM-CoT reasoning workflow demonstration.
    Demonstrates the agent's multi-step reasoning and evidence gathering.
    Expected: >=4 plan steps, >=2 modalities, structured report.
    """
    prompt = "请展示你的AIPOM-CoT推理流程，并解释你将如何从知识图谱中获取证据。"

    prov.log_think("Starting reasoning demonstration", {'prompt': prompt})

    agent = create_agent_from_args(args)
    evidence = EvidenceBuffer()

    try:
        result = agent.answer(prompt, max_iterations=args.max_depth or 12)

        # Log plan
        prov.log_plan(
            plan_steps=[{'step': i+1, 'purpose': s['purpose']} for i, s in enumerate(result.get('executed_steps', []))],
            planner_type='adaptive'
        )

        # Record evidence for each step
        for i, step in enumerate(result.get('executed_steps', []), 1):
            modality = step.get('modality', 'general')
            if step.get('actual_result_summary', {}).get('row_count', 0) > 0:
                evidence.add_evidence(
                    modality=modality,
                    source_step=i,
                    query=f"Step {i}: {step['purpose']}",
                    data=[{'summary': step.get('actual_result_summary', {})}]
                )
            prov.log_act(
                step_number=i,
                action_type='reasoning_step',
                purpose=step['purpose'],
                result_summary=step.get('actual_result_summary', {})
            )

        # Generate report
        report_content = generate_analysis_report(
            analysis_type='reasoning',
            prompt=prompt,
            result=result,
            evidence=evidence,
            seed=args.seed
        )

        # Save outputs
        with open(output_base / "report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Save step details as JSON
        with open(output_base / "data" / "steps.json", 'w', encoding='utf-8') as f:
            json.dump(result.get('executed_steps', []), f, indent=2, ensure_ascii=False, default=str)

        prov.log_run_end(
            termination_reason='completed',
            total_steps=result.get('total_steps', 0),
            total_kg_queries=evidence.get_kg_query_count(),
            execution_time=result.get('execution_time', 0)
        )

        return result
    finally:
        agent.close()


def run_fingerprint_analysis(args, output_base: Path, prov: ProvenanceLogger) -> Dict:
    """
    Cross-Modal Brain Region Fingerprint Analysis.

    AGENT-DRIVEN IMPLEMENTATION: Uses FingerprintAgent with full TPAR workflow.
    The agent reasons about which fingerprints to compute and how to analyze them.

    Outputs:
    - data/similarity_molecule.csv
    - data/similarity_morphology.csv
    - data/similarity_projection.csv
    - data/mismatch_mol_morph.csv
    - data/mismatch_mol_proj.csv
    - figures/*.png
    - report.md
    """
    from fingerprint_agent import FingerprintAgent

    prov.log_think("Starting fingerprint analysis: Cross-modal mismatch analysis (agent-driven)", {
        'implementation': 'FingerprintAgent with TPAR workflow',
        'top_n_regions': args.top_n_regions if hasattr(args, 'top_n_regions') else 30
    })

    # Initialize agent with TPAR reasoning
    agent = FingerprintAgent(
        seed=args.seed,
        output_dir=str(output_base),
        top_n_regions=getattr(args, 'top_n_regions', 30)
    )

    # Run agent analysis (full TPAR workflow)
    result = agent.run()

    # Get evidence from agent
    evidence_summary = result.get('evidence', {})
    kg_query_count = evidence_summary.get('kg_query_count', 0)
    coverage_rate = evidence_summary.get('coverage_rate', 0)

    if not result.get('success'):
        # Create evidence buffer for fail report
        evidence = EvidenceBuffer()
        _write_fail_fast_report(output_base, 'fingerprint', result, evidence, prov, args.seed)
        print(f"FAIL: Fingerprint analysis agent execution failed: {result.get('error')}")
        return result

    if kg_query_count == 0 or coverage_rate == 0:
        result['success'] = False
        result['error'] = f'FAILED: NO EVIDENCE (kg_queries={kg_query_count}, coverage={coverage_rate:.0%})'
        evidence = EvidenceBuffer()
        _write_fail_fast_report(output_base, 'fingerprint', result, evidence, prov, args.seed)
        print(f"FAIL: Fingerprint analysis - {result['error']}")
        return result

    print(f"SUCCESS: Fingerprint analysis completed with {kg_query_count} KG queries, coverage={coverage_rate:.0%}")
    print(f"Files generated: {list(result.get('files', {}).keys())}")

    return result


def run_circuit_discovery(args, output_base: Path, prov: ProvenanceLogger) -> Dict:
    """
    Gene-Centric Neural Circuit Discovery (default gene: Car3).
    AGENT-DRIVEN IMPLEMENTATION: Uses CircuitAgent with full TPAR workflow.
    The agent reasons about gene circuit analysis and generates comprehensive panels.
    Outputs: subclass list, region enrichment, morphology counts, projections.
    """
    from circuit_agent import CircuitAgent

    gene = args.gene or "Car3"

    prov.log_think(f"Starting circuit discovery: {gene} neuron profiling (agent-driven)", {'gene': gene})

    # Initialize agent with TPAR reasoning
    agent = CircuitAgent(
        gene=gene,
        seed=args.seed,
        output_dir=str(output_base)
    )

    # Run agent analysis (full TPAR workflow including report generation)
    result = agent.run()

    # Get evidence from agent
    evidence_summary = result.get('evidence', {})
    kg_count = evidence_summary.get('kg_query_count', 0)
    coverage = evidence_summary.get('coverage_rate', 0)

    if not result.get('success'):
        # Create evidence buffer for fail report
        evidence = EvidenceBuffer()
        _write_fail_fast_report(output_base, 'circuit', result, evidence, prov, args.seed)
        print(f"FAIL: Circuit discovery agent execution failed: {result.get('error')}")
        return result

    if kg_count == 0 or coverage == 0:
        result['success'] = False
        result['error'] = f'FAILED: NO EVIDENCE (kg_queries={kg_count}, coverage={coverage:.0%})'
        evidence = EvidenceBuffer()
        _write_fail_fast_report(output_base, 'circuit', result, evidence, prov, args.seed)
        print(f"FAIL: Circuit discovery - {result['error']}")
        return result

    print(f"SUCCESS: Circuit discovery completed with {kg_count} KG queries, coverage={coverage:.0%}")
    print(f"Files generated: {list(result.get('files', {}).keys())}")

    return result


def run_kg_mode(args):
    """
    General KG query mode with entity/attribute resolution.
    Supports: entity lookup, attribute queries, constrained traversal.
    """
    query = args.query
    if not query:
        print("Error: --query is required for kg mode")
        return None

    router = IntentRouter()
    intent = router.classify(query)
    logger.info(f"Intent: {intent.value}")

    # Set determinism
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get budget
    budget_limits = get_budget_for_intent(intent, args.budget)

    # Initialize provenance
    prov = create_provenance_logger(run_id=f"kg_{int(time.time())}", seed=args.seed)
    prov.log_run_start(mode='kg', intent=intent.value, query=query, budget=args.budget)

    agent = create_agent_from_args(args)

    try:
        result = agent.answer(query, max_iterations=min(budget_limits.max_plan_steps, args.max_depth or 6))

        prov.log_run_end(
            termination_reason='completed',
            total_steps=result.get('total_steps', 0),
            total_kg_queries=len(result.get('executed_steps', [])),
            execution_time=result.get('execution_time', 0)
        )

        print(f"\n{'='*80}")
        print("ANSWER:")
        print('='*80)
        print(result['answer'])

        return result
    finally:
        agent.close()


def save_fingerprint_csvs(result: Dict, data_dir: Path, prov: ProvenanceLogger):
    """Extract and save CSV data from fingerprint analysis"""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try to extract mismatch data from executed steps
    for step in result.get('executed_steps', []):
        actual = step.get('actual_result', {})
        if actual.get('success') and actual.get('data'):
            data = actual['data']
            purpose = step.get('purpose', '').lower()

            # Detect data type and save
            if data and isinstance(data[0], dict):
                if 'mismatch_GM' in data[0] or 'mismatch_combined' in data[0]:
                    # Mismatch data
                    df = pd.DataFrame(data)
                    df.to_csv(data_dir / "mismatch_pairs.csv", index=False)
                    prov.log_act(
                        step_number=step.get('step_number', 0),
                        action_type='save_csv',
                        purpose='Save mismatch pairs',
                        result_summary={'file': 'mismatch_pairs.csv', 'rows': len(data)}
                    )

    # If visualization files were generated, log them
    if 'visualization_files' in result:
        for name, path in result['visualization_files'].items():
            logger.info(f"Generated: {name} -> {path}")


def save_circuit_csvs(result: Dict, data_dir: Path, gene: str, prov: ProvenanceLogger):
    """Extract and save CSV data from circuit analysis"""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Extract from intermediate_data
    intermediate = result.get('intermediate_data', {})

    for key, data in intermediate.items():
        if data and isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)

            # Determine filename based on content
            if 'region' in str(data[0].keys()).lower() and 'enrichment' in key.lower():
                filename = f"{gene}_region_enrichment.csv"
            elif 'subclass' in str(data[0].keys()).lower():
                filename = f"{gene}_subclass_list.csv"
            elif 'target' in str(data[0].keys()).lower():
                filename = f"{gene}_projection_targets.csv"
            else:
                filename = f"{gene}_{key}.csv"

            df.to_csv(data_dir / filename, index=False)
            prov.log_act(
                step_number=0,
                action_type='save_csv',
                purpose=f'Save {filename}',
                result_summary={'file': filename, 'rows': len(data)}
            )


def generate_analysis_report(analysis_type: str, prompt: str, result: Dict,
                       evidence: EvidenceBuffer, seed: int) -> str:
    """Generate markdown report for analysis cases"""
    lines = [
        f"# MS {analysis_type.upper()} Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seed:** {seed}",
        "",
        "## Query",
        "",
        f"> {prompt}",
        "",
        "## Execution Summary",
        "",
        f"- **Total Steps:** {result.get('total_steps', 0)}",
        f"- **Execution Time:** {result.get('execution_time', 0):.2f}s",
        f"- **Confidence:** {result.get('confidence_score', 0):.3f}",
        "",
        "## Modalities Covered",
        ""
    ]

    modalities = result.get('adaptive_planning', {}).get('modalities_covered', [])
    for mod in modalities:
        lines.append(f"- {mod}")

    lines.extend([
        "",
        "## Answer",
        "",
        result.get('answer', 'No answer generated'),
        "",
        "## Reasoning Steps",
        ""
    ])

    for i, step in enumerate(result.get('executed_steps', []), 1):
        lines.append(f"### Step {i}: {step.get('purpose', 'Unknown')}")
        lines.append("")
        lines.append(f"- **Modality:** {step.get('modality', 'N/A')}")
        summary = step.get('actual_result_summary', {})
        lines.append(f"- **Results:** {summary.get('row_count', 0)} rows")
        lines.append(f"- **Reflection:** {step.get('reflection', 'N/A')}")
        lines.append("")

    # Add evidence summary
    lines.append(evidence.to_markdown())

    return "\n".join(lines)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='AIPOM-CoT V10 Production - Neuroscience KG Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chat mode (default)
  python aipom_v10_production.py --mode chat --query "你好"
  python aipom_v10_production.py --mode chat --query "What is Car3?"

  # MS reproduction mode
  python aipom_v10_production.py --mode analysis --analysis reasoning --seed 42
  python aipom_v10_production.py --mode analysis --analysis fingerprint --seed 42
  python aipom_v10_production.py --mode analysis --analysis circuit --seed 42

  # General KG query mode
  python aipom_v10_production.py --mode kg --query "HIP 有什么属性"
        """
    )

    parser.add_argument('--mode', choices=['chat', 'analysis', 'kg'], default='chat',
                        help='Operation mode (default: chat)')
    parser.add_argument('--analysis', choices=['reasoning', 'fingerprint', 'circuit'],
                        help='Analysis type (required for analysis mode)')
    parser.add_argument('--query', type=str,
                        help='Natural language query')
    parser.add_argument('--gene', type=str, default='Car3',
                        help='Gene marker for circuit analysis (default: Car3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--snapshot-id', type=str, default=None,
                        help='Optional snapshot ID for logging')
    parser.add_argument('--max-depth', type=int, default=15,
                        help='Maximum planning depth (default: 15)')
    parser.add_argument('--budget', choices=['light', 'standard', 'heavy'], default='light',
                        help='Budget level for chat/kg modes (default: light)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Log startup info
    logger.info(f"AIPOM-CoT V10 Production")
    logger.info(f"Mode: {args.mode}, Seed: {args.seed}, Budget: {args.budget}")

    # Dispatch to appropriate mode
    if args.mode == 'chat':
        return run_chat_mode(args)
    elif args.mode == 'analysis':
        return run_analysis_mode(args)
    elif args.mode == 'kg':
        return run_kg_mode(args)


if __name__ == "__main__":
    main()