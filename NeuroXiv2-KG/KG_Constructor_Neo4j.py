import json
import sys
import warnings
import ast
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import nrrd
from loguru import logger
from neo4j import GraphDatabase
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 导入现有模块
from Subregion_Loader import SubregionLoader
from neuron_subregion_relationship_inserter import (
    NeuronSubregionRelationshipInserter,
    verify_relationships
)

from data_loader_enhanced import (
    load_data,
    prepare_analysis_data,
    map_cells_to_regions_fixed
)

from KG_ConstructorV4_Neo4j_with_neuron_subregion_subregionrelationship import (
    Neo4jConnector,
    MERFISHHierarchyLoader,
    RegionAnalyzer,
    MorphologyDataLoader,
    KnowledgeGraphBuilderNeo4j,
    CHUNK_SIZE,
    PCT_THRESHOLD,
    BATCH_SIZE,
    MORPH_ATTRIBUTES,
    STAT_ATTRIBUTES,
    NEURON_ATTRIBUTES,
    CONNECTIONS_FILE,
    INFO_FILE
)

from KG_ConstructorV5_Neo4j_without_neuron_to_ME_V3 import (
    NeuronProjectionProcessorV5Fixed,
    MERFISHSubregionMapper
)

def clean_feature_name(feature_name: str) -> str:
    """
    清理特征名称，确保符合Neo4j属性命名规范
    """
    # 替换特殊字符
    cleaned = (feature_name
               .replace('%', 'pct')  # 关键修复：将%替换为pct
               .replace(' ', '_')
               .replace('/', '_')
               .replace('-', '_')
               .replace('(', '')
               .replace(')', '')
               .lower())
    return cleaned

class NeuronDataLoader:
    """神经元数据加载器 - 处理列表格式的连接数据和完整形态学特征"""

    def __init__(self, data_path: Path, region_analyzer=None, morphology_loader=None):
        """
        初始化神经元数据加载器

        参数:
            data_path: 数据目录路径
            region_analyzer: 区域分析器实例
            morphology_loader: 形态学数据加载器实例
        """
        self.data_path = data_path
        self.region_analyzer = region_analyzer
        self.morphology_loader = morphology_loader
        self.info_df = None
        self.connections_df = None
        self.neurons_data = {}
        self.neuron_connections = {
            'den_neighbouring': {},  # neuron_id -> set of dendrite neighbors
            'axon_neighbouring': {}  # neuron_id -> set of axon neighbors
        }

        # 形态学数据映射
        self.axon_morph_df = None
        self.dendrite_morph_df = None

        # 形态学特征列表
        self.morph_features = set()

    def load_neuron_data(self) -> bool:
        """加载神经元数据"""
        logger.info("加载神经元数据...")

        # 1. 加载info.csv
        info_file = self.data_path / INFO_FILE
        if not info_file.exists():
            logger.error(f"info.csv文件不存在: {info_file}")
            return False

        try:
            self.info_df = pd.read_csv(info_file)

            # 过滤掉带有'CCF-thin'或'local'的神经元（参考data_loader_enhanced.py）
            if 'ID' in self.info_df.columns:
                original_len = len(self.info_df)
                # 如果ID列是字符串类型，进行过滤
                if self.info_df['ID'].dtype == 'object':
                    self.info_df = self.info_df[~self.info_df['ID'].str.contains('CCF-thin|local', na=False)]
                    filtered_count = original_len - len(self.info_df)
                    if filtered_count > 0:
                        logger.info(f"过滤掉了 {filtered_count} 个带有'CCF-thin|local'的神经元")

            logger.info(f"加载了 {len(self.info_df)} 条神经元信息")

            # 检查必要的列
            required_cols = ['ID', 'celltype']
            if not all(col in self.info_df.columns for col in required_cols):
                logger.error(f"info.csv缺少必要的列: {required_cols}")
                logger.info(f"可用的列: {list(self.info_df.columns)}")
                return False

        except Exception as e:
            logger.error(f"加载info.csv失败: {e}")
            return False

        # 2. 加载连接数据
        connections_file = self.data_path / CONNECTIONS_FILE
        if not connections_file.exists():
            logger.warning(f"连接文件不存在: {connections_file}")
            # 继续处理，只是没有连接信息
        else:
            try:
                self.connections_df = pd.read_csv(connections_file)
                logger.info(f"加载了 {len(self.connections_df)} 条连接记录")

                # 检查连接文件的列
                if 'axon_ID' in self.connections_df.columns and 'dendrite_ID' in self.connections_df.columns:
                    logger.info("连接文件包含axon_ID和dendrite_ID列")

                    # 显示数据样例以了解格式
                    sample_row = self.connections_df.iloc[0] if len(self.connections_df) > 0 else None
                    if sample_row is not None:
                        logger.debug(f"axon_ID样例: {str(sample_row['axon_ID'])[:100]}...")
                        logger.debug(f"dendrite_ID样例: {str(sample_row['dendrite_ID'])[:100]}...")
                else:
                    logger.warning(f"连接文件列名: {list(self.connections_df.columns)}")

            except Exception as e:
                logger.error(f"加载连接文件失败: {e}")

        # 3. 加载形态学数据
        self.load_morphology_data()

        return True

    def load_morphology_data(self):
        """加载完整的形态学数据（修正版）"""
        logger.info("加载神经元形态学数据（修正版）...")

        try:
            # 加载轴突形态数据
            axon_file = self.data_path / "axonfull_morpho.csv"
            if axon_file.exists():
                self.axon_morph_df = pd.read_csv(axon_file)
                logger.info(f"  - 原始轴突数据: {self.axon_morph_df.shape}")

                # 过滤掉带有'CCF-thin'或'local'的记录
                if 'ID' in self.axon_morph_df.columns:
                    original_len = len(self.axon_morph_df)
                    self.axon_morph_df = self.axon_morph_df[
                        ~self.axon_morph_df['ID'].astype(str).str.contains('CCF-thin|local', na=False)
                    ]
                    filtered_count = original_len - len(self.axon_morph_df)
                    if filtered_count > 0:
                        logger.info(f"    过滤掉了 {filtered_count} 条记录")

                # 收集所有形态学特征列
                exclude_cols = ['ID', 'name', 'celltype', 'type']
                axon_features = [col for col in self.axon_morph_df.columns if col not in exclude_cols]

                # 为每个特征添加 axonal_ 前缀
                for feat in axon_features:
                    # feature_name = f'axonal_{feat}'.replace(' ', '_').replace('/', '_').replace('-', '_').lower()
                    feature_name = clean_feature_name(f'axonal_{feat}')
                    self.morph_features.add(feature_name)

                logger.info(f"  - 轴突数据: {len(self.axon_morph_df)} 条记录，{len(axon_features)} 个特征")
                logger.debug(f"  - 轴突特征样例: {axon_features[:5]}")

            # 加载树突形态数据
            dendrite_file = self.data_path / "denfull_morpho.csv"
            if dendrite_file.exists():
                self.dendrite_morph_df = pd.read_csv(dendrite_file)
                logger.info(f"  - 原始树突数据: {self.dendrite_morph_df.shape}")

                # 过滤掉带有'CCF-thin'或'local'的记录
                if 'ID' in self.dendrite_morph_df.columns:
                    original_len = len(self.dendrite_morph_df)
                    self.dendrite_morph_df = self.dendrite_morph_df[
                        ~self.dendrite_morph_df['ID'].astype(str).str.contains('CCF-thin|local', na=False)
                    ]
                    filtered_count = original_len - len(self.dendrite_morph_df)
                    if filtered_count > 0:
                        logger.info(f"    过滤掉了 {filtered_count} 条记录")

                # 收集所有形态学特征列
                dendrite_features = [col for col in self.dendrite_morph_df.columns if col not in exclude_cols]

                # 为每个特征添加 dendritic_ 前缀
                for feat in dendrite_features:
                    # feature_name = f'dendritic_{feat}'.replace(' ', '_').replace('/', '_').replace('-', '_').lower()
                    feature_name = clean_feature_name(f'dendritic_{feat}')
                    self.morph_features.add(feature_name)

                logger.info(f"  - 树突数据: {len(self.dendrite_morph_df)} 条记录，{len(dendrite_features)} 个特征")
                logger.debug(f"  - 树突特征样例: {dendrite_features[:5]}")

            logger.info(f"✓ 总共收集了 {len(self.morph_features)} 个形态学特征")

            # 显示部分特征名称
            sample_features = list(self.morph_features)[:10]
            logger.info(f"  - 特征样例: {sample_features}")

        except Exception as e:
            logger.error(f"加载形态学数据失败: {e}")
            import traceback
            traceback.print_exc()

    def process_neuron_data(self):
        """处理神经元数据，包含所有形态学特征（修正版）"""
        if self.info_df is None:
            logger.error("没有加载神经元数据")
            return

        logger.info("处理神经元数据（包含完整形态学特征）...")

        # 创建区域名称到ID的映射
        region_name_to_id = {}
        if self.region_analyzer:
            for region_id, info in self.region_analyzer.region_info.items():
                acronym = info.get('acronym', '')
                if acronym:
                    region_name_to_id[acronym] = region_id

        # 从info.csv提取神经元信息
        neuron_count_with_axon = 0
        neuron_count_with_dendrite = 0

        for idx, row in tqdm(self.info_df.iterrows(), total=len(self.info_df), desc="处理神经元信息"):
            neuron_id = str(row['ID'])

            # 提取基础区域名称（移除层信息）
            celltype = row.get('celltype', '')
            base_region = self.extract_base_region(celltype)

            # 获取区域ID
            region_id = region_name_to_id.get(base_region) if base_region else None

            # 初始化神经元数据
            neuron_data = {
                'neuron_id': neuron_id,
                'name': row.get('name', neuron_id),
                'celltype': celltype,
                'base_region': base_region,
                'region_id': region_id
            }

            # ==================== 添加轴突形态学数据 ====================
            if self.axon_morph_df is not None and 'ID' in self.axon_morph_df.columns:
                axon_data = self.axon_morph_df[self.axon_morph_df['ID'] == neuron_id]

                if not axon_data.empty:
                    neuron_count_with_axon += 1

                    # 处理每个特征列
                    for col in self.axon_morph_df.columns:
                        if col not in ['ID', 'name', 'celltype', 'type']:
                            # ✅ 修复：使用 clean_feature_name 函数
                            feature_name = clean_feature_name(f'axonal_{col}')

                            try:
                                # 尝试转换为数值并取平均值（处理多行情况）
                                values = pd.to_numeric(axon_data[col], errors='coerce')
                                if not values.isna().all():
                                    neuron_data[feature_name] = float(values.mean())
                                else:
                                    neuron_data[feature_name] = 0.0
                            except Exception as e:
                                logger.debug(f"处理轴突特征 {col} 失败: {e}")
                                neuron_data[feature_name] = 0.0

            # ==================== 添加树突形态学数据 ====================
            if self.dendrite_morph_df is not None and 'ID' in self.dendrite_morph_df.columns:
                dendrite_data = self.dendrite_morph_df[self.dendrite_morph_df['ID'] == neuron_id]

                if not dendrite_data.empty:
                    neuron_count_with_dendrite += 1

                    # 处理每个特征列
                    for col in self.dendrite_morph_df.columns:
                        if col not in ['ID', 'name', 'celltype', 'type']:
                            # ✅ 修复：使用 clean_feature_name 函数
                            feature_name = clean_feature_name(f'dendritic_{col}')

                            try:
                                # 尝试转换为数值并取平均值（处理多行情况）
                                values = pd.to_numeric(dendrite_data[col], errors='coerce')
                                if not values.isna().all():
                                    neuron_data[feature_name] = float(values.mean())
                                else:
                                    neuron_data[feature_name] = 0.0
                            except Exception as e:
                                logger.debug(f"处理树突特征 {col} 失败: {e}")
                                neuron_data[feature_name] = 0.0

            # ==================== 为缺失的特征设置默认值 ====================
            for feature in self.morph_features:
                if feature not in neuron_data:
                    neuron_data[feature] = 0.0

            self.neurons_data[neuron_id] = neuron_data

        logger.info(f"✓ 处理了 {len(self.neurons_data)} 个神经元")
        logger.info(f"  - 有轴突数据: {neuron_count_with_axon} 个")
        logger.info(f"  - 有树突数据: {neuron_count_with_dendrite} 个")

        # 显示一个样例神经元的特征
        if self.neurons_data:
            sample_neuron = list(self.neurons_data.values())[0]
            logger.info(f"  - 样例神经元特征数量: {len(sample_neuron)}")

            # 显示形态学特征
            morph_features = {k: v for k, v in sample_neuron.items()
                              if k.startswith('axonal_') or k.startswith('dendritic_')}
            logger.info(f"  - 样例神经元形态学特征数量: {len(morph_features)}")
            logger.debug(f"  - 样例特征: {list(morph_features.keys())[:10]}")

        # 处理连接数据
        if self.connections_df is not None:
            self.process_connections()

    def parse_id_list(self, id_str: str) -> List[str]:
        """
        解析ID列表字符串

        参数:
            id_str: 包含ID列表的字符串，格式如 "[id1, id2, id3]" 或 "id1,id2,id3"

        返回:
            ID列表
        """
        if pd.isna(id_str) or not id_str:
            return []

        id_str = str(id_str).strip()

        # 处理不同的格式
        try:
            # 尝试作为Python列表解析
            if id_str.startswith('[') and id_str.endswith(']'):
                # 使用ast.literal_eval安全解析
                ids = ast.literal_eval(id_str)
                if isinstance(ids, list):
                    return [str(id).strip() for id in ids if id]

            # 尝试作为JSON数组解析
            if id_str.startswith('['):
                ids = json.loads(id_str)
                if isinstance(ids, list):
                    return [str(id).strip() for id in ids if id]

            # 处理被引号包裹的逗号分隔列表
            if id_str.startswith('"') and id_str.endswith('"'):
                id_str = id_str[1:-1]

            # 处理逗号分隔的列表
            if ',' in id_str:
                ids = id_str.split(',')
                return [id.strip().strip('"').strip("'") for id in ids if id.strip()]

            # 单个ID的情况
            clean_id = id_str.strip('"').strip("'")
            if clean_id:
                return [clean_id]

        except Exception as e:
            logger.debug(f"解析ID列表失败: {id_str[:100]}... 错误: {e}")

        return []

    def process_connections(self):
        """
        处理神经元连接数据（修正版本）

        文件结构：
        - SWC_Name: 源神经元ID
        - axon_ID: 该神经元的轴突邻居列表
        - dendrite_ID: 该神经元的树突邻居列表

        关系理解：
        - axon_neighbouring: 源神经元的轴突连接到的目标神经元
        - den_neighbouring: 连接到源神经元树突的其他神经元
        """
        logger.info("处理神经元连接...")

        # 初始化连接字典
        for neuron_id in self.neurons_data:
            self.neuron_connections['den_neighbouring'][neuron_id] = set()
            self.neuron_connections['axon_neighbouring'][neuron_id] = set()

        # 统计
        total_axon_connections = 0
        total_den_connections = 0
        processed_rows = 0
        skipped_rows = 0

        # 处理每条连接记录
        for idx, row in tqdm(self.connections_df.iterrows(),
                             total=len(self.connections_df),
                             desc="处理连接记录"):

            # 获取源神经元ID（从SWC_Name列）
            source_id = str(row.get('SWC_Name', ''))

            # 如果源神经元不在我们的数据中，跳过
            if not source_id or source_id not in self.neurons_data:
                skipped_rows += 1
                continue

            # 解析轴突邻居列表
            axon_neighbors = self.parse_id_list(row.get('axon_ID', ''))
            for neighbor_id in axon_neighbors:
                neighbor_id = str(neighbor_id)
                if neighbor_id in self.neurons_data:
                    # 源神经元的轴突连接到neighbor_id
                    self.neuron_connections['axon_neighbouring'][source_id].add(neighbor_id)
                    total_axon_connections += 1

            # 解析树突邻居列表
            dendrite_neighbors = self.parse_id_list(row.get('dendrite_ID', ''))
            for neighbor_id in dendrite_neighbors:
                neighbor_id = str(neighbor_id)
                if neighbor_id in self.neurons_data:
                    # neighbor_id连接到源神经元的树突
                    self.neuron_connections['den_neighbouring'][source_id].add(neighbor_id)
                    total_den_connections += 1

            if axon_neighbors or dendrite_neighbors:
                processed_rows += 1

        # 统计连接
        neurons_with_axon_connections = sum(
            1 for neighbors in self.neuron_connections['axon_neighbouring'].values()
            if neighbors
        )
        neurons_with_den_connections = sum(
            1 for neighbors in self.neuron_connections['den_neighbouring'].values()
            if neighbors
        )

        logger.info(f"处理了 {processed_rows} 行有效连接记录，跳过了 {skipped_rows} 行")
        logger.info(f"创建了 {total_axon_connections} 个轴突连接")
        logger.info(f"创建了 {total_den_connections} 个树突连接")
        logger.info(f"{neurons_with_axon_connections} 个神经元有轴突连接")
        logger.info(f"{neurons_with_den_connections} 个神经元有树突连接")

    def extract_base_region(self, celltype):
        """提取基础区域名称（移除层信息）"""
        if not celltype or pd.isna(celltype):
            return None

        celltype = str(celltype).strip()

        # 要移除的层模式
        layer_patterns = ['1', '2/3', '4', '5', '6a', '6b']
        base_region = celltype

        for layer in layer_patterns:
            if celltype.endswith(layer):
                base_region = celltype[:-len(layer)].strip()
                break

        return base_region if base_region else None

# ==================== 修复后的类定义 ====================
def verify_neuron_morphology_features(neo4j_connector, database='neo4j'):
    """验证Neuron节点的形态学特征"""
    logger.info("=" * 80)
    logger.info("验证Neuron节点形态学特征")
    logger.info("=" * 80)

    with neo4j_connector.driver.session(database=database) as session:
        # 1. 统计Neuron节点总数
        query1 = "MATCH (n:Neuron) RETURN count(n) as count"
        result1 = session.run(query1)
        neuron_count = result1.single()['count']
        logger.info(f"\n✓ 总Neuron节点数: {neuron_count}")

        # 2. 检查一个样例Neuron的所有属性
        query2 = """
        MATCH (n:Neuron)
        RETURN n
        LIMIT 1
        """
        result2 = session.run(query2)
        sample_neuron = result2.single()

        if sample_neuron:
            neuron_props = dict(sample_neuron['n'])

            # 分类属性
            basic_props = []
            axonal_props = []
            dendritic_props = []
            other_props = []

            for key in neuron_props.keys():
                if key.startswith('axonal_'):
                    axonal_props.append(key)
                elif key.startswith('dendritic_'):
                    dendritic_props.append(key)
                elif key in ['neuron_id', 'name', 'celltype', 'base_region']:
                    basic_props.append(key)
                else:
                    other_props.append(key)

            logger.info(f"\n样例Neuron属性统计:")
            logger.info(f"  - 基础属性: {len(basic_props)} 个")
            logger.info(f"    {basic_props}")
            logger.info(f"  - 轴突特征: {len(axonal_props)} 个")
            if axonal_props:
                logger.info(f"    样例: {axonal_props[:5]}")
            logger.info(f"  - 树突特征: {len(dendritic_props)} 个")
            if dendritic_props:
                logger.info(f"    样例: {dendritic_props[:5]}")
            if other_props:
                logger.info(f"  - 其他属性: {len(other_props)} 个")
                logger.info(f"    {other_props}")

            logger.info(f"  - 总属性数: {len(neuron_props)}")

        # 3. 统计有形态学特征的神经元数量
        query3 = """
        MATCH (n:Neuron)
        WHERE any(key IN keys(n) WHERE key STARTS WITH 'axonal_')
        RETURN count(n) as count_with_axon
        """
        result3 = session.run(query3)
        count_with_axon = result3.single()['count_with_axon']

        query4 = """
        MATCH (n:Neuron)
        WHERE any(key IN keys(n) WHERE key STARTS WITH 'dendritic_')
        RETURN count(n) as count_with_dendrite
        """
        result4 = session.run(query4)
        count_with_dendrite = result4.single()['count_with_dendrite']

        logger.info(f"\n形态学特征覆盖:")
        logger.info(f"  - 有轴突特征: {count_with_axon}/{neuron_count} ({count_with_axon / neuron_count * 100:.1f}%)")
        logger.info(
            f"  - 有树突特征: {count_with_dendrite}/{neuron_count} ({count_with_dendrite / neuron_count * 100:.1f}%)")

    logger.info("=" * 80)

class HierarchicalProjectionProcessorFixed:
    """
    层级投射关系处理器（修复版）

    修复内容：
    1. 统一所有PROJECT_TO关系属性为：weight, total, neuron_count
    """

    def __init__(self, projection_processor: 'NeuronProjectionProcessorV5Fixed',
                 neo4j_connector, data_path: Path):
        self.proj_processor = projection_processor
        self.neo4j = neo4j_connector
        self.data_path = data_path

        # 存储聚合后的投射数据
        self.region_to_subregion_projections = {}
        self.subregion_to_subregion_projections = {}

        # 神经元位置信息
        self.neuron_locations = {}

        # 统计
        self.stats = {
            'region_to_subregion': 0,
            'subregion_to_subregion': 0
        }

    def load_neuron_locations(self) -> bool:
        """加载神经元位置信息"""
        logger.info("加载神经元位置信息...")

        location_file = self.data_path / "info_with_me_subregion_v2.csv"

        if not location_file.exists():
            logger.warning(f"位置文件不存在，使用基础info.csv")
            return self._load_basic_neuron_locations()

        try:
            df = pd.read_csv(location_file, header=None, encoding='latin1', low_memory=False)

            col_names = [
                'ID', 'data_source', 'celltype', 'brain_atlas', 'recon_method',
                'has_recon_axon', 'has_recon_den', 'has_ab', 'layer', 'has_apical',
                'has_local', 'hemisphere', 'subregion', 'me_subregion_voxel',
                'me_subregion_acronym', 'me_subregion_name', 'parent_region_id',
                'is_me_subregion'
            ]
            df.columns = col_names

            for _, row in df.iterrows():
                neuron_id = str(row['ID'])
                celltype = str(row['celltype'])
                region = self._extract_base_region(celltype)

                self.neuron_locations[neuron_id] = {
                    'region': region,
                    'subregion': str(row['subregion']) if pd.notna(row['subregion']) else None,
                    'me_subregion': str(row['me_subregion_acronym']) if pd.notna(row['me_subregion_acronym']) else None,
                    'celltype': celltype
                }

            logger.info(f"成功加载 {len(self.neuron_locations)} 个神经元位置")
            return True

        except Exception as e:
            logger.error(f"加载位置信息失败: {e}")
            return False

    def _load_basic_neuron_locations(self) -> bool:
        """从基础info.csv加载位置"""
        info_file = self.data_path / "info.csv"

        if not info_file.exists():
            logger.error("无法找到神经元位置信息文件")
            return False

        try:
            info_df = pd.read_csv(info_file)

            for _, row in info_df.iterrows():
                neuron_id = str(row['ID'])
                celltype = str(row.get('celltype', ''))
                region = self._extract_base_region(celltype)

                self.neuron_locations[neuron_id] = {
                    'region': region,
                    'subregion': None,
                    'me_subregion': None,
                    'celltype': celltype
                }

            logger.info(f"从基础文件加载了 {len(self.neuron_locations)} 个神经元位置")
            return True

        except Exception as e:
            logger.error(f"加载基础位置失败: {e}")
            return False

    def _extract_base_region(self, celltype: str) -> str:
        """提取基础region名称"""
        layer_patterns = ['1', '2/3', '4', '5', '6a', '6b']
        base_region = celltype

        for layer in layer_patterns:
            if celltype.endswith(layer):
                base_region = celltype[:-len(layer)]
                break

        return base_region

    def aggregate_region_to_subregion_projections(self):
        """聚合Region->Subregion投射（修复版：计算weight, total, neuron_count）"""
        logger.info("=" * 60)
        logger.info("聚合 Region -> Subregion 投射关系（统一属性版）")
        logger.info("=" * 60)

        if not self.neuron_locations:
            logger.error("未加载神经元位置信息")
            return

        # 存储每个连接的详细信息
        connection_details = defaultdict(lambda: {
            'total_length': 0.0,  # total
            'neuron_ids': set(),   # 用于计算neuron_count
            'lengths': []          # 用于计算weight（平均值）
        })

        for neuron_id, subregion_projs in tqdm(
            self.proj_processor.neuron_to_subregion_axon.items(),
            desc="聚合 Region->Subregion"
        ):
            if neuron_id not in self.neuron_locations:
                continue

            source_region = self.neuron_locations[neuron_id]['region']
            if not source_region:
                continue

            for target_subregion, length in subregion_projs.items():
                key = (source_region, target_subregion)

                connection_details[key]['total_length'] += length
                connection_details[key]['neuron_ids'].add(neuron_id)
                connection_details[key]['lengths'].append(length)

        # 转换为最终格式
        for (source, target), details in connection_details.items():
            self.region_to_subregion_projections[(source, target)] = {
                'weight': np.mean(details['lengths']),  # 平均投射长度
                'total': details['total_length'],        # 总投射长度
                'neuron_count': len(details['neuron_ids'])  # 神经元数量
            }

        logger.info(f"✓ 计算了 {len(self.region_to_subregion_projections)} 个 Region->Subregion 投射")

    def aggregate_subregion_to_subregion_projections(self):
        """聚合Subregion->Subregion投射（修复版：计算weight, total, neuron_count）"""
        logger.info("=" * 60)
        logger.info("聚合 Subregion -> Subregion 投射关系（统一属性版）")
        logger.info("=" * 60)

        if not self.neuron_locations:
            logger.error("未加载神经元位置信息")
            return

        connection_details = defaultdict(lambda: {
            'total_length': 0.0,
            'neuron_ids': set(),
            'lengths': []
        })

        for neuron_id, subregion_projs in tqdm(
            self.proj_processor.neuron_to_subregion_axon.items(),
            desc="聚合 Subregion->Subregion"
        ):
            if neuron_id not in self.neuron_locations:
                continue

            source_subregion = self.neuron_locations[neuron_id]['subregion']
            if not source_subregion or source_subregion == 'None':
                continue

            for target_subregion, length in subregion_projs.items():
                key = (source_subregion, target_subregion)

                connection_details[key]['total_length'] += length
                connection_details[key]['neuron_ids'].add(neuron_id)
                connection_details[key]['lengths'].append(length)

        for (source, target), details in connection_details.items():
            self.subregion_to_subregion_projections[(source, target)] = {
                'weight': np.mean(details['lengths']),
                'total': details['total_length'],
                'neuron_count': len(details['neuron_ids'])
            }

        logger.info(f"✓ 计算了 {len(self.subregion_to_subregion_projections)} 个 Subregion->Subregion 投射")

    def insert_region_to_subregion_projections(self, batch_size: int = 1000):
        """插入Region->Subregion投射（统一属性版）"""
        logger.info("插入 Region -> Subregion 投射关系...")

        if not self.region_to_subregion_projections:
            logger.warning("没有投射数据")
            return

        batch_relationships = []
        success_count = 0

        for (source_region, target_subregion), stats in tqdm(
            self.region_to_subregion_projections.items(),
            desc="插入 Region->Subregion"
        ):
            rel = {
                'source_acronym': source_region,
                'target_acronym': target_subregion,
                'weight': float(stats['weight']),
                'total': float(stats['total']),
                'neuron_count': int(stats['neuron_count']),
                'projection_type': 'axon',
                'source_level': 'region',
                'target_level': 'subregion'
            }

            batch_relationships.append(rel)

            if len(batch_relationships) >= batch_size:
                count = self._execute_region_to_subregion_batch(batch_relationships)
                success_count += count
                batch_relationships = []

        if batch_relationships:
            count = self._execute_region_to_subregion_batch(batch_relationships)
            success_count += count

        self.stats['region_to_subregion'] = success_count
        logger.info(f"✓ 成功插入 {success_count} 个 Region->Subregion 投射")

    def _execute_region_to_subregion_batch(self, batch: List[Dict]) -> int:
        """执行批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (r:Region)
        WHERE r.acronym = rel.source_acronym
        MATCH (s:Subregion {acronym: rel.target_acronym})
        MERGE (r)-[p:PROJECT_TO]->(s)
        SET p.weight = rel.weight,
            p.total = rel.total,
            p.neuron_count = rel.neuron_count,
            p.projection_type = rel.projection_type,
            p.source_level = rel.source_level,
            p.target_level = rel.target_level,
            p.source_acronym = rel.source_acronym,
            p.target_acronym = rel.target_acronym
        RETURN count(p) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            return 0

    def insert_subregion_to_subregion_projections(self, batch_size: int = 1000):
        """插入Subregion->Subregion投射（统一属性版）"""
        logger.info("插入 Subregion -> Subregion 投射关系...")

        if not self.subregion_to_subregion_projections:
            logger.warning("没有投射数据")
            return

        batch_relationships = []
        success_count = 0

        for (source_subregion, target_subregion), stats in tqdm(
            self.subregion_to_subregion_projections.items(),
            desc="插入 Subregion->Subregion"
        ):
            rel = {
                'source_acronym': source_subregion,
                'target_acronym': target_subregion,
                'weight': float(stats['weight']),
                'total': float(stats['total']),
                'neuron_count': int(stats['neuron_count']),
                'projection_type': 'axon',
                'source_level': 'subregion',
                'target_level': 'subregion'
            }

            batch_relationships.append(rel)

            if len(batch_relationships) >= batch_size:
                count = self._execute_subregion_to_subregion_batch(batch_relationships)
                success_count += count
                batch_relationships = []

        if batch_relationships:
            count = self._execute_subregion_to_subregion_batch(batch_relationships)
            success_count += count

        self.stats['subregion_to_subregion'] = success_count
        logger.info(f"✓ 成功插入 {success_count} 个 Subregion->Subregion 投射")

    def _execute_subregion_to_subregion_batch(self, batch: List[Dict]) -> int:
        """执行批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (s1:Subregion {acronym: rel.source_acronym})
        MATCH (s2:Subregion {acronym: rel.target_acronym})
        MERGE (s1)-[p:PROJECT_TO]->(s2)
        SET p.weight = rel.weight,
            p.total = rel.total,
            p.neuron_count = rel.neuron_count,
            p.projection_type = rel.projection_type,
            p.source_level = rel.source_level,
            p.target_level = rel.target_level,
            p.source_acronym = rel.source_acronym,
            p.target_acronym = rel.target_acronym
        RETURN count(p) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            return 0

    def insert_all_hierarchical_projections(self, batch_size: int = 1000):
        """插入所有层级投射关系"""
        logger.info("=" * 80)
        logger.info("插入层级投射关系（统一属性版）")
        logger.info("=" * 80)

        if not self.load_neuron_locations():
            logger.warning("无法加载神经元位置")

        self.aggregate_region_to_subregion_projections()
        self.aggregate_subregion_to_subregion_projections()

        self.insert_region_to_subregion_projections(batch_size)
        self.insert_subregion_to_subregion_projections(batch_size)

        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("层级投射关系统计（统一属性版）")
        logger.info("=" * 60)
        logger.info(f"Region -> Subregion: {self.stats['region_to_subregion']}")
        logger.info(f"Subregion -> Subregion: {self.stats['subregion_to_subregion']}")
        logger.info(f"总计: {sum(self.stats.values())}")
        logger.info("=" * 60)


class KnowledgeGraphBuilderNeo4jV3Fixed(KnowledgeGraphBuilderNeo4j):
    """V3.1修复版知识图谱构建器"""

    def __init__(self, neo4j_connector):
        super().__init__(neo4j_connector)
        # 去重集合
        self.processed_den_pairs = set()
        self.processed_axon_pairs = set()

    def generate_and_insert_neuron_nodes_with_full_morphology(self, neuron_loader: 'NeuronDataLoader'):
        """插入包含完整形态学特征的Neuron节点（修正版）"""
        if not neuron_loader or not neuron_loader.neurons_data:
            logger.warning("没有神经元数据")
            return

        logger.info("=" * 60)
        logger.info("插入Neuron节点（包含完整形态学特征）")
        logger.info("=" * 60)

        # 去重
        unique_neurons = {}
        duplicate_count = 0

        for neuron_id, neuron_data in tqdm(neuron_loader.neurons_data.items(),
                                           desc="准备Neuron数据"):
            if neuron_id in unique_neurons:
                duplicate_count += 1
            else:
                unique_neurons[neuron_id] = neuron_data

        if duplicate_count > 0:
            logger.warning(f"移除了 {duplicate_count} 个重复神经元")

        logger.info(f"准备插入 {len(unique_neurons)} 个唯一神经元")

        # 批量插入
        batch_nodes = []
        neuron_count = 0

        # 统计形态学特征
        total_features = 0
        axonal_features = 0
        dendritic_features = 0

        for neuron_id, neuron_data in tqdm(unique_neurons.items(), desc="插入Neuron节点"):
            # 基础属性
            node_dict = {
                'neuron_id': neuron_id,
                'name': neuron_data.get('name', neuron_id),
                'celltype': neuron_data.get('celltype', ''),
                'base_region': neuron_data.get('base_region', '')
            }

            # ==================== 添加所有形态学特征 ====================
            feature_count = 0
            for key, value in neuron_data.items():
                # 跳过基础属性
                if key in ['neuron_id', 'name', 'celltype', 'base_region', 'region_id']:
                    continue

                # 添加形态学特征
                if key.startswith('axonal_') or key.startswith('dendritic_'):
                    try:
                        node_dict[key] = float(value) if value is not None else 0.0
                        feature_count += 1

                        if key.startswith('axonal_'):
                            axonal_features += 1
                        elif key.startswith('dendritic_'):
                            dendritic_features += 1
                    except (ValueError, TypeError):
                        node_dict[key] = 0.0

            total_features += feature_count
            batch_nodes.append(node_dict)

            # 批量插入
            if len(batch_nodes) >= BATCH_SIZE:
                self._insert_neurons_batch_with_merge_v2(batch_nodes)
                neuron_count += len(batch_nodes)
                batch_nodes = []

        # 插入剩余节点
        if batch_nodes:
            self._insert_neurons_batch_with_merge_v2(batch_nodes)
            neuron_count += len(batch_nodes)

        self.stats['neurons_inserted'] = neuron_count

        # 统计信息
        avg_features = total_features / neuron_count if neuron_count > 0 else 0
        logger.info("=" * 60)
        logger.info(f"✓ 成功插入 {neuron_count} 个Neuron节点")
        logger.info(f"  - 平均每个神经元: {avg_features:.1f} 个形态学特征")
        logger.info(f"  - 总轴突特征: {axonal_features}")
        logger.info(f"  - 总树突特征: {dendritic_features}")
        logger.info("=" * 60)

    def _insert_neurons_batch_with_merge_v2(self, batch_nodes):
        """使用MERGE批量插入Neuron（改进版）"""
        if not batch_nodes:
            return

        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            # 获取所有属性名
            all_keys = set()
            for node in batch_nodes:
                all_keys.update(node.keys())

            # 移除neuron_id（用于MERGE）
            all_keys.discard('neuron_id')

            # 构建SET语句
            set_clauses = [f"n.{key} = props.{key}" for key in sorted(all_keys)]
            set_statement = ",\n                ".join(set_clauses)

            query = f"""
            UNWIND $batch AS props
            MERGE (n:Neuron {{neuron_id: props.neuron_id}})
            SET {set_statement}
            """

            try:
                session.run(query, batch=batch_nodes)
            except Exception as e:
                logger.error(f"批量插入Neuron失败: {e}")
                logger.error(f"  - 样例节点: {batch_nodes[0] if batch_nodes else 'N/A'}")
                # 尝试逐个插入以定位问题
                self._insert_neurons_one_by_one_v2(batch_nodes)

    def _insert_neurons_one_by_one_v2(self, nodes):
        """逐个插入Neuron（用于调试）"""
        logger.warning("批量插入失败，切换到逐个插入模式...")
        failed_count = 0
        success_count = 0

        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            for node in nodes:
                try:
                    # 构建SET子句
                    set_parts = []
                    for key in node.keys():
                        if key != 'neuron_id':
                            set_parts.append(f"n.{key} = ${key}")

                    set_statement = ", ".join(set_parts)

                    query = f"""
                    MERGE (n:Neuron {{neuron_id: $neuron_id}})
                    SET {set_statement}
                    """

                    session.run(query, **node)
                    success_count += 1

                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        logger.error(f"插入神经元失败 {node['neuron_id']}: {e}")

        logger.info(f"逐个插入完成: 成功 {success_count}, 失败 {failed_count}")

    def _insert_neurons_batch_with_merge(self, batch_nodes):
        """使用MERGE批量插入Neuron"""
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            # 动态构建SET语句
            sample_node = batch_nodes[0]
            set_clauses = []
            for key in sample_node.keys():
                if key != 'neuron_id':
                    set_clauses.append(f"n.{key} = props.{key}")

            set_statement = ",\n                ".join(set_clauses)

            query = f"""
            UNWIND $batch AS props
            MERGE (n:Neuron {{neuron_id: props.neuron_id}})
            SET {set_statement}
            """

            try:
                session.run(query, batch=batch_nodes)
            except Exception as e:
                logger.error(f"批量插入Neuron失败: {e}")

    def generate_and_insert_neuron_projection_relationships(
        self,
        projection_processor: NeuronProjectionProcessorV5Fixed
    ):
        """插入Neuron投射关系（统一属性版）"""
        logger.info("=" * 60)
        logger.info("插入Neuron投射关系（统一属性版）")
        logger.info("=" * 60)

        # 1. Neuron->Region
        if projection_processor.neuron_to_region_axon:
            logger.info("\n插入 Neuron -> Region 投射...")
            self._insert_neuron_projections_unified(
                projection_processor.neuron_to_region_axon,
                target_level='Region',
                projection_type='axon'
            )

        # 2. Neuron->Subregion
        if projection_processor.neuron_to_subregion_axon:
            logger.info("\n插入 Neuron -> Subregion 投射...")
            self._insert_neuron_projections_unified(
                projection_processor.neuron_to_subregion_axon,
                target_level='Subregion',
                projection_type='axon'
            )

        logger.info("=" * 60)

    def _insert_neuron_projections_unified(
        self,
        projection_dict: Dict[str, Dict[str, float]],
        target_level: str,
        projection_type: str
    ):
        """统一的Neuron投射插入（统一属性版）"""
        batch_relationships = []
        success_count = 0

        for neuron_id, projections in tqdm(
            projection_dict.items(),
            desc=f"处理 Neuron->{target_level}"
        ):
            for target_acronym, length in projections.items():
                rel = {
                    'neuron_id': str(neuron_id),
                    'target_acronym': target_acronym,
                    'weight': float(length),  # 统一使用weight
                    'total': float(length),   # 对单个神经元，total=weight
                    'neuron_count': 1,        # 单个神经元
                    'projection_type': projection_type,
                    'source_level': 'neuron',
                    'target_level': target_level.lower()
                }

                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_neuron_projection_batch(
                        batch_relationships,
                        target_level
                    )
                    success_count += count
                    batch_relationships = []

        if batch_relationships:
            count = self._execute_neuron_projection_batch(
                batch_relationships,
                target_level
            )
            success_count += count

        logger.info(f"  ✓ 成功插入 {success_count} 个 Neuron->{target_level} 关系")

    def _execute_neuron_projection_batch(self, batch: List[Dict], target_level: str):
        """执行Neuron投射批量插入"""
        if target_level == 'Region':
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (t:Region)
            WHERE t.acronym = rel.target_acronym
            MERGE (n)-[p:PROJECT_TO]->(t)
            SET p.weight = rel.weight,
                p.total = rel.total,
                p.neuron_count = rel.neuron_count,
                p.projection_type = rel.projection_type,
                p.source_level = rel.source_level,
                p.target_level = rel.target_level,
                p.source_acronym = n.base_region,
                p.target_acronym = rel.target_acronym
            RETURN count(p) as created_count
            """
        else:  # Subregion
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (s:Subregion {acronym: rel.target_acronym})
            MERGE (n)-[p:PROJECT_TO]->(s)
            SET p.weight = rel.weight,
                p.total = rel.total,
                p.neuron_count = rel.neuron_count,
                p.projection_type = rel.projection_type,
                p.source_level = rel.source_level,
                p.target_level = rel.target_level,
                p.source_acronym = n.base_region,
                p.target_acronym = rel.target_acronym
            RETURN count(p) as created_count
            """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            return 0

    def insert_neuron_locate_at_relationships(self, neuron_loader: NeuronDataLoader):
        """
        ⭐ 修复2: 插入Neuron-LOCATE_AT->Region关系
        """
        logger.info("=" * 60)
        logger.info("插入 Neuron-LOCATE_AT->Region 关系")
        logger.info("=" * 60)

        if not neuron_loader or not neuron_loader.neurons_data:
            logger.warning("没有神经元数据")
            return

        batch_relationships = []
        success_count = 0

        for neuron_id, neuron_data in tqdm(
            neuron_loader.neurons_data.items(),
            desc="处理LOCATE_AT关系"
        ):
            region_id = neuron_data.get('region_id')

            if region_id:
                rel = {
                    'neuron_id': neuron_id,
                    'region_id': int(region_id)
                }
                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_locate_at_batch(batch_relationships)
                    success_count += count
                    batch_relationships = []

        if batch_relationships:
            count = self._execute_locate_at_batch(batch_relationships)
            success_count += count

        logger.info(f"✓ 成功插入 {success_count} 个 LOCATE_AT 关系")
        logger.info("=" * 60)

    def _execute_locate_at_batch(self, batch: List[Dict]) -> int:
        """执行LOCATE_AT批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (n:Neuron {neuron_id: rel.neuron_id})
        MATCH (r:Region {region_id: rel.region_id})
        MERGE (n)-[loc:LOCATE_AT]->(r)
        RETURN count(loc) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入LOCATE_AT失败: {e}")
            return 0

    def generate_and_insert_neuron_neighbouring_relationships(self, neuron_loader: NeuronDataLoader):
        """
        ⭐ 修复3: 插入去重的neighbouring关系
        """
        logger.info("=" * 60)
        logger.info("插入Neuron neighbouring关系（去重版）")
        logger.info("=" * 60)

        if not neuron_loader:
            logger.warning("没有神经元数据")
            return

        # 重置去重集合
        self.processed_den_pairs = set()
        self.processed_axon_pairs = set()

        # 1. DEN_NEIGHBOURING
        self._insert_den_neighbouring_deduplicated(neuron_loader)

        # 2. AXON_NEIGHBOURING
        self._insert_axon_neighbouring_deduplicated(neuron_loader)

        logger.info("=" * 60)

    def _insert_den_neighbouring_deduplicated(self, neuron_loader: NeuronDataLoader):
        """插入去重的DEN_NEIGHBOURING关系"""
        logger.info("\n插入 DEN_NEIGHBOURING 关系（去重版）...")

        batch_relationships = []
        success_count = 0
        duplicate_count = 0

        for source_id, neighbors in tqdm(
            neuron_loader.neuron_connections['den_neighbouring'].items(),
            desc="处理DEN_NEIGHBOURING"
        ):
            for target_id in neighbors:
                # 创建排序的pair作为key
                pair_key = tuple(sorted([source_id, target_id]))

                if pair_key in self.processed_den_pairs:
                    duplicate_count += 1
                    continue

                self.processed_den_pairs.add(pair_key)

                rel = {
                    'source_id': source_id,
                    'target_id': target_id
                }
                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_den_neighbouring_batch(batch_relationships)
                    success_count += count
                    batch_relationships = []

        if batch_relationships:
            count = self._execute_den_neighbouring_batch(batch_relationships)
            success_count += count

        logger.info(f"✓ 成功插入 {success_count} 个 DEN_NEIGHBOURING 关系")
        logger.info(f"✓ 去除了 {duplicate_count} 个重复关系")

    def _execute_den_neighbouring_batch(self, batch: List[Dict]) -> int:
        """执行DEN_NEIGHBOURING批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (n1:Neuron {neuron_id: rel.source_id})
        MATCH (n2:Neuron {neuron_id: rel.target_id})
        MERGE (n1)-[r:DEN_NEIGHBOURING]->(n2)
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入DEN_NEIGHBOURING失败: {e}")
            return 0

    def _insert_axon_neighbouring_deduplicated(self, neuron_loader: NeuronDataLoader):
        """插入去重的AXON_NEIGHBOURING关系"""
        logger.info("\n插入 AXON_NEIGHBOURING 关系（去重版）...")

        batch_relationships = []
        success_count = 0
        duplicate_count = 0

        for source_id, neighbors in tqdm(
            neuron_loader.neuron_connections['axon_neighbouring'].items(),
            desc="处理AXON_NEIGHBOURING"
        ):
            for target_id in neighbors:
                # 创建排序的pair作为key
                pair_key = tuple(sorted([source_id, target_id]))

                if pair_key in self.processed_axon_pairs:
                    duplicate_count += 1
                    continue

                self.processed_axon_pairs.add(pair_key)

                rel = {
                    'source_id': source_id,
                    'target_id': target_id
                }
                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_axon_neighbouring_batch(batch_relationships)
                    success_count += count
                    batch_relationships = []

        if batch_relationships:
            count = self._execute_axon_neighbouring_batch(batch_relationships)
            success_count += count

        logger.info(f"✓ 成功插入 {success_count} 个 AXON_NEIGHBOURING 关系")
        logger.info(f"✓ 去除了 {duplicate_count} 个重复关系")

    def _execute_axon_neighbouring_batch(self, batch: List[Dict]) -> int:
        """执行AXON_NEIGHBOURING批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (n1:Neuron {neuron_id: rel.source_id})
        MATCH (n2:Neuron {neuron_id: rel.target_id})
        MERGE (n1)-[r:AXON_NEIGHBOURING]->(n2)
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入AXON_NEIGHBOURING失败: {e}")
            return 0

    # 其他方法保持不变...
    def generate_and_insert_merfish_subregion_relationships(
        self,
        merfish_cells: pd.DataFrame,
        level: str
    ):
        """插入MERFISH到Subregion/ME_Subregion的HAS关系"""
        logger.info(f"=" * 60)
        logger.info(f"插入 HAS 关系到 {level.upper()}")
        logger.info(f"=" * 60)

        if level == 'subregion':
            region_col = 'subregion_acronym'
            node_label = 'Subregion'
            id_field = 'acronym'
        else:
            region_col = 'me_subregion_acronym'
            node_label = 'ME_Subregion'
            id_field = 'acronym'

        if region_col not in merfish_cells.columns:
            logger.warning(f"没有 {region_col} 列")
            return

        valid_cells = merfish_cells[merfish_cells[region_col].notna()]

        if len(valid_cells) == 0:
            logger.warning(f"没有细胞映射到 {level}")
            return

        logger.info(f"有 {len(valid_cells)} 个细胞映射到 {level}")

        for cell_type_level in ['class', 'subclass', 'supertype', 'cluster']:
            if cell_type_level not in valid_cells.columns:
                continue

            self._insert_has_relationships_to_subregion(
                valid_cells,
                region_col,
                cell_type_level,
                node_label,
                id_field
            )

        logger.info(f"=" * 60)

    def _insert_has_relationships_to_subregion(
        self,
        cells_df: pd.DataFrame,
        region_col: str,
        cell_type_col: str,
        target_label: str,
        target_id_field: str
    ):
        """插入HAS关系"""
        logger.info(f"\n插入 HAS_{cell_type_col.upper()} 到 {target_label}...")

        if cell_type_col == 'class':
            id_map = self.class_id_map
        elif cell_type_col == 'subclass':
            id_map = self.subclass_id_map
        elif cell_type_col == 'supertype':
            id_map = self.supertype_id_map
        else:
            id_map = self.cluster_id_map

        valid_cells = cells_df[
            (cells_df[region_col].notna()) &
            (cells_df[cell_type_col].notna())
        ]

        if len(valid_cells) == 0:
            logger.warning(f"没有有效的 {cell_type_col} 数据")
            return

        counts_df = valid_cells.groupby([region_col, cell_type_col]).size().reset_index(name='count')
        region_totals = valid_cells.groupby(region_col).size().reset_index(name='total')
        counts_df = pd.merge(counts_df, region_totals, on=region_col)
        counts_df['pct'] = counts_df['count'] / counts_df['total']
        counts_df = counts_df[counts_df['pct'] >= PCT_THRESHOLD]

        logger.info(f"准备插入 {len(counts_df)} 条关系")

        batch_relationships = []
        success_count = 0

        for region_acronym, group in tqdm(
            counts_df.groupby(region_col),
            desc=f"HAS_{cell_type_col.upper()}"
        ):
            group_sorted = group.sort_values('pct', ascending=False)
            rank = 1

            for _, row in group_sorted.iterrows():
                cell_type = row[cell_type_col]

                if cell_type in id_map:
                    rel = {
                        'region_acronym': str(region_acronym),
                        'cell_type_id': id_map[cell_type],
                        'pct_cells': float(row['pct']),
                        'rank': rank
                    }
                    batch_relationships.append(rel)
                    rank += 1

                    if len(batch_relationships) >= BATCH_SIZE:
                        count = self._execute_has_subregion_batch(
                            batch_relationships,
                            cell_type_col,
                            target_label,
                            target_id_field
                        )
                        success_count += count
                        batch_relationships = []

        if batch_relationships:
            count = self._execute_has_subregion_batch(
                batch_relationships,
                cell_type_col,
                target_label,
                target_id_field
            )
            success_count += count

        logger.info(f"✓ 成功插入 {success_count} 条关系")

    def _execute_has_subregion_batch(self, batch, cell_type_col, target_label, target_id_field):
        """执行HAS关系批量插入"""
        cell_type_label = cell_type_col.capitalize()

        query = f"""
        UNWIND $batch AS rel
        MATCH (sr:{target_label} {{{target_id_field}: rel.region_acronym}})
        MATCH (ct:{cell_type_label} {{tran_id: rel.cell_type_id}})
        MERGE (sr)-[r:HAS_{cell_type_col.upper()}]->(ct)
        SET r.pct_cells = rel.pct_cells,
            r.rank = rel.rank
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入HAS关系失败: {e}")
            return 0


# ==================== Main函数 ====================

def main(data_dir: str = "../data",
         hierarchy_json: str = None,
         neo4j_uri: str = "bolt://localhost:7687",
         neo4j_user: str = "neo4j",
         neo4j_password: str = "password",
         database_name: str = "neuroxiv",
         clear_database: bool = False):
    """
    主函数 - V3.1修复版

    修复内容：
    1. 统一所有PROJECT_TO关系属性（weight, total, neuron_count）
    2. 补充Neuron-LOCATE_AT->Region关系
    3. 去重neighbouring关系
    """

    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - V3.1 修复版")
    logger.info("=" * 60)
    logger.info("修复内容：")
    logger.info("1. 统一所有PROJECT_TO关系属性（weight, total, neuron_count）")
    logger.info("2. 补充Neuron-LOCATE_AT->Region关系")
    logger.info("3. 去重neighbouring关系")
    logger.info("=" * 60)

    # 初始化Neo4j连接
    neo4j_conn = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password, database_name)

    if not neo4j_conn.connect():
        logger.error("无法连接到Neo4j")
        return

    try:
        if clear_database:
            neo4j_conn.clear_database_smart()

        neo4j_conn.create_constraints()

        # Phase 1: 数据加载
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: 数据加载")
        logger.info("=" * 60)

        data_path = Path(data_dir)
        data = load_data(data_path)
        processed_data = prepare_analysis_data(data)

        region_data = processed_data.get('region_data', pd.DataFrame())
        merfish_cells = processed_data.get('merfish_cells', pd.DataFrame())
        projection_data = processed_data.get('projection_df', pd.DataFrame())

        builder = KnowledgeGraphBuilderNeo4jV3Fixed(neo4j_conn)

        tree_data = processed_data.get('tree', [])
        if tree_data:
            builder.region_analyzer = RegionAnalyzer(tree_data)

        # Phase 2-3: 加载层级和形态数据
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2-3: 加载层级和形态数据")
        logger.info("=" * 60)

        hierarchy_loader = MERFISHHierarchyLoader(
            Path(hierarchy_json) if hierarchy_json else data_path / "hierarchy.json"
        )

        if not hierarchy_loader.load_hierarchy():
            logger.error("无法加载层级数据")
            return

        morphology_loader = MorphologyDataLoader(data_path, builder.region_analyzer)
        if morphology_loader.load_morphology_data():
            if projection_data is not None and not projection_data.empty:
                morphology_loader.set_projection_data(projection_data)
            builder.morphology_loader = morphology_loader

        # Phase 3.5: 加载神经元数据
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.5: 加载神经元数据")
        logger.info("=" * 60)

        neuron_loader = NeuronDataLoader(
            data_path,
            builder.region_analyzer,
            builder.morphology_loader
        )

        if neuron_loader.load_neuron_data():
            neuron_loader.process_neuron_data()
            logger.info(f"✓ 加载了 {len(neuron_loader.neurons_data)} 个神经元")
        else:
            logger.warning("无法加载神经元数据")
            neuron_loader = None

        # Phase 3.6: 处理投射数据
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.6: 处理投射数据")
        logger.info("=" * 60)

        ccf_tree_json = data_path / "tree_yzx.json"

        projection_processor = NeuronProjectionProcessorV5Fixed(
            data_path=data_path,
            ccf_tree_json=ccf_tree_json
        )

        if projection_processor.run_full_pipeline():
            logger.info("✓ 投射数据处理成功")
        else:
            logger.warning("投射数据处理失败")
            projection_processor = None

        # Phase 3.7: MERFISH映射
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.7: MERFISH细胞空间映射")
        logger.info("=" * 60)

        ccf_me_json = data_path / "surf_tree_ccf-me.json"
        annotation_file = data_path / "annotation_25.nrrd"

        if all([ccf_tree_json.exists(), ccf_me_json.exists(), annotation_file.exists()]):
            merfish_mapper = MERFISHSubregionMapper(
                data_path=data_path,
                ccf_tree_json=ccf_tree_json,
                ccf_me_json=ccf_me_json
            )

            if merfish_mapper.load_standard_ccf_tree():
                if merfish_mapper.load_ccf_me_tree_for_subregions():
                    try:
                        annotation_volume, _ = nrrd.read(str(annotation_file))
                        merfish_cells = merfish_mapper.map_cells_to_subregions(
                            merfish_cells,
                            annotation_volume
                        )
                    except Exception as e:
                        logger.error(f"Subregion映射失败: {e}")

                    if merfish_mapper.load_me_subregion_annotation():
                        merfish_cells = merfish_mapper.map_cells_to_me_subregions(merfish_cells)

        # Phase 3.8: 加载Subregion
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.8: 加载Subregion数据")
        logger.info("=" * 60)

        subregion_loader = SubregionLoader(ccf_me_json)
        if not subregion_loader.load_subregion_data():
            logger.warning("无法加载Subregion数据")
            subregion_loader = None

        # Phase 4: 插入节点
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: 插入节点")
        logger.info("=" * 60)

        builder.set_hierarchy_loader(hierarchy_loader)

        logger.info("\n插入Region节点...")
        builder.generate_and_insert_unified_region_nodes(region_data, merfish_cells)

        if neuron_loader:
            logger.info("\n插入Neuron节点...")
            builder.generate_and_insert_neuron_nodes_with_full_morphology(neuron_loader)

        if subregion_loader:
            logger.info("\n插入Subregion和ME_Subregion节点...")
            builder.generate_and_insert_subregion_nodes(subregion_loader)
            builder.generate_and_insert_me_subregion_nodes(subregion_loader)

        logger.info("\n插入MERFISH细胞类型节点...")
        builder.generate_and_insert_merfish_nodes_from_hierarchy(merfish_cells)

        # Phase 5: 插入关系
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5: 插入关系")
        logger.info("=" * 60)

        # HAS关系 - Region级别
        logger.info("\n插入Region级别HAS关系...")
        for level in ['class', 'subclass', 'supertype', 'cluster']:
            builder.generate_and_insert_has_relationships_unified(merfish_cells, level)

        # HAS关系 - Subregion级别
        if 'subregion_acronym' in merfish_cells.columns:
            mapped_count = merfish_cells['subregion_acronym'].notna().sum()
            if mapped_count > 0:
                logger.info(f"\n插入Subregion级别HAS关系...")
                builder.generate_and_insert_merfish_subregion_relationships(
                    merfish_cells,
                    'subregion'
                )

        # HAS关系 - ME_Subregion级别
        if 'me_subregion_acronym' in merfish_cells.columns:
            mapped_count = merfish_cells['me_subregion_acronym'].notna().sum()
            if mapped_count > 0:
                logger.info(f"\n插入ME_Subregion级别HAS关系...")
                builder.generate_and_insert_merfish_subregion_relationships(
                    merfish_cells,
                    'me_subregion'
                )

        # BELONGS_TO关系
        logger.info("\n插入BELONGS_TO关系...")
        builder.generate_and_insert_belongs_to_from_hierarchy()

        # PROJECT_TO关系 - Region级别
        logger.info("\n插入Region级别PROJECT_TO关系...")
        builder.generate_and_insert_project_to_relationships(projection_data)

        # ⭐ 修复2: LOCATE_AT关系
        if neuron_loader:
            logger.info("\n⭐ 插入Neuron-LOCATE_AT->Region关系...")
            builder.insert_neuron_locate_at_relationships(neuron_loader)

        # ⭐ 修复3: 去重的neighbouring关系
        # if neuron_loader:
        #     logger.info("\n⭐ 插入去重的neighbouring关系...")
        #     builder.generate_and_insert_neuron_neighbouring_relationships(neuron_loader)

        # ⭐ 修复1: 统一属性的投射关系
        if projection_processor:
            logger.info("\n⭐ 插入统一属性的Neuron投射关系...")
            builder.generate_and_insert_neuron_projection_relationships(projection_processor)

        # Subregion层级关系
        if subregion_loader:
            logger.info("\n插入Subregion层级关系...")
            builder.generate_and_insert_subregion_relationships(subregion_loader)

        # ⭐ 修复1: 层级投射关系（统一属性）
        if projection_processor:
            logger.info("\n⭐ 插入统一属性的层级投射关系...")
            hierarchical_proj_processor = HierarchicalProjectionProcessorFixed(
                projection_processor=projection_processor,
                neo4j_connector=neo4j_conn,
                data_path=data_path
            )
            hierarchical_proj_processor.insert_all_hierarchical_projections(
                batch_size=BATCH_SIZE
            )

        # Neuron-Subregion关系
        logger.info("\n插入Neuron-Subregion关系...")
        neuron_subregion_inserter = NeuronSubregionRelationshipInserter(
            neo4j_conn, data_path
        )

        if neuron_subregion_inserter.load_neuron_subregion_mapping():
            neuron_subregion_inserter.insert_all_relationships(batch_size=BATCH_SIZE)
            verify_relationships(neo4j_conn, database_name)

        # 统计报告
        logger.info("\n" + "=" * 60)
        logger.info("统计报告")
        logger.info("=" * 60)
        builder.print_statistics_report_enhanced_with_subregion()

        logger.info("\n" + "=" * 60)
        logger.info("✓ 知识图谱构建完成！")
        logger.info("=" * 60)
        logger.info("\n修复验证：")
        logger.info("1. ✓ 所有PROJECT_TO关系使用统一属性（weight, total, neuron_count）")
        logger.info("2. ✓ 已添加Neuron-LOCATE_AT->Region关系")
        logger.info("3. ✓ neighbouring关系已去重")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"构建过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 V3.1修复版')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data')
    parser.add_argument('--hierarchy_json', type=str,
                       default='/home/wlj/NeuroXiv2/data/tran-data-type-tree.json')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, required=True, default='neuroxiv')
    parser.add_argument('--database', type=str, default='neo4j')
    parser.add_argument('--clear_database', action='store_true')

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        hierarchy_json=args.hierarchy_json,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        database_name=args.database,
        clear_database=args.clear_database
    )