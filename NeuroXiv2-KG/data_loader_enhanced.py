"""
数据加载模块 - 包含用于加载和预处理各种数据源的函数
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import json
from pathlib import Path
import h5py
from loguru import logger
from typing import Dict, Any, Union, Tuple, Optional, List, Set
import datetime

# 从setup导入常量
from setup import (
    MORPH_FEATURES,
    RANDOM_STATE,
    FDR_THRESHOLD
)

# 当前时间和用户 - 使用提供的时间和用户名
CURRENT_TIME = "2025-07-30 16:06:26"
CURRENT_USER = "PrometheusTT"

# 正确的形态特征名称映射
MORPH_FEATURE_NAMES = {
    "axonal_length": "Total Length",
    "axonal_branches": "Number of Bifurcations",
    "axonal_bifurcation_remote_angle": "Average Bifurcation Angle Remote",
    "axonal_maximum_branch_order": "Max Branch Order",
    "dendritic_length": "Total Length",
    "dendritic_branches": "Number of Bifurcations",
    "dendritic_bifurcation_remote_angle": "Average Bifurcation Angle Remote",
    "dendritic_maximum_branch_order": "Max Branch Order",
}

# 记录运行信息
logger.info(f"数据加载器初始化时间: {CURRENT_TIME}")
logger.info(f"运行用户: {CURRENT_USER}")

# 定义数据文件路径
MERFISH_COORDINATE_FILES = [
    "ccf_coordinates_1.csv",
    "ccf_coordinates_2.csv",
    "ccf_coordinates_3.csv",
    "ccf_coordinates_4.csv"
]

MERFISH_METADATA_FILES = [
    "cell_metadata_with_cluster_annotation_1.csv",
    "cell_metadata_with_cluster_annotation_2.csv",
    "cell_metadata_with_cluster_annotation_3.csv",
    "cell_metadata_with_cluster_annotation_4.csv"
]

MERFISH_H5AD_FILES = [
    "Zhuang-ABCA-1-raw.h5ad",
    "Zhuang-ABCA-2-raw.h5ad",
    "Zhuang-ABCA-3-raw.h5ad",
    "Zhuang-ABCA-4-raw.h5ad"
]

MORPH_FILES = {
    "axon": "axonfull_morpho.csv",
    "dendrite": "denfull_morpho.csv"
}

PROJECTION_FILE = "Proj_Axon_Final.csv"
ANNOTATION_NRRD = "annotation_25.nrrd"
TREE_JSON = "tree_yzx.json"
INFO_FILE = "info.csv"
SOMA_FILE = "soma.csv"

def load_gene_panel_mapping(gene_panel_file: str) -> Dict[str, str]:
    """
    从基因面板文件加载Ensembl ID到基因符号的映射

    参数:
        gene_panel_file: 基因面板CSV文件路径

    返回:
        Ensembl ID到基因符号的映射字典
    """
    try:
        logger.info(f"加载基因ID映射文件: {gene_panel_file}")
        gene_panel = pd.read_csv(gene_panel_file)

        # 检查必要的列是否存在
        if 'ensemble_id' in gene_panel.columns and 'gene_symbol' in gene_panel.columns:
            mapping = dict(zip(gene_panel['ensemble_id'], gene_panel['gene_symbol']))
            column_name = 'ensemble_id'
        elif 'ensembl_id' in gene_panel.columns and 'gene_symbol' in gene_panel.columns:
            mapping = dict(zip(gene_panel['ensembl_id'], gene_panel['gene_symbol']))
            column_name = 'ensembl_id'
        else:
            logger.warning(f"基因面板文件格式不正确，列名: {list(gene_panel.columns)}")
            return {}

        logger.info(f"创建了{len(mapping)}个基因ID映射 (从列 '{column_name}' 到 'gene_symbol')")

        # 示例映射
        sample_items = list(mapping.items())[:5]
        logger.debug(f"映射示例: {sample_items}")

        return mapping

    except Exception as e:
        logger.error(f"加载基因面板映射失败: {e}")
        return {}


def convert_ensembl_to_symbols(adata: ad.AnnData, gene_mapping: Dict[str, str]) -> ad.AnnData:
    """
    将AnnData中的Ensembl ID转换为基因符号

    参数:
        adata: 包含Ensembl ID作为基因名的AnnData对象
        gene_mapping: Ensembl ID到基因符号的映射字典

    返回:
        更新了基因名的AnnData对象
    """
    logger.info("转换Ensembl ID到基因符号...")

    if len(gene_mapping) == 0:
        logger.warning("基因映射为空，无法转换")
        return adata

    # 创建副本以避免修改原始数据
    adata_copy = adata.copy()

    # 获取当前的基因名（Ensembl IDs）
    current_genes = adata_copy.var_names.tolist()

    # 转换为基因符号
    new_names = []
    converted_count = 0
    missing_count = 0

    for ensembl_id in current_genes:
        if ensembl_id in gene_mapping:
            gene_symbol = gene_mapping[ensembl_id]
            new_names.append(gene_symbol)
            converted_count += 1
        else:
            # 对于缺失的映射，保留原始ID
            new_names.append(ensembl_id)
            missing_count += 1

    # 检查是否有重复的基因名
    duplicates = set([x for x in new_names if new_names.count(x) > 1])
    if duplicates:
        logger.warning(f"发现{len(duplicates)}个重复的基因符号")

        # 处理重复的基因名
        unique_names = []
        seen = set()
        for i, name in enumerate(new_names):
            if name in seen:
                # 对于重复的基因名，添加原始Ensembl ID作为后缀
                unique_names.append(f"{name}_{current_genes[i]}")
            else:
                unique_names.append(name)
                seen.add(name)

        new_names = unique_names

    # 更新基因名称
    adata_copy.var_names = new_names

    # 添加原始Ensembl ID作为var的一个列
    adata_copy.var['ensembl_id'] = current_genes

    logger.info(f"转换了{converted_count}/{len(current_genes)}个基因ID (缺失映射: {missing_count})")
    logger.info(f"基因符号示例: {new_names[:5]}")

    return adata_copy


def update_adata_gene_symbols(adata: ad.AnnData, gene_panel_file: str) -> ad.AnnData:
    """
    从基因面板文件加载映射并更新AnnData对象中的基因名

    参数:
        adata: 包含Ensembl ID作为基因名的AnnData对象
        gene_panel_file: 基因面板CSV文件路径

    返回:
        更新了基因名的AnnData对象
    """
    # 加载映射
    gene_mapping = load_gene_panel_mapping(gene_panel_file)

    # 转换基因名
    if gene_mapping:
        return convert_ensembl_to_symbols(adata, gene_mapping)
    else:
        logger.warning("未加载基因映射，保持原始基因名不变")
        return adata

def diagnose_coordinate_mapping(cells_df: pd.DataFrame, annotation_volume: np.ndarray) -> None:
    """
    诊断坐标映射问题
    """
    logger.info("诊断坐标映射...")

    # 检查坐标范围
    logger.info("细胞坐标范围（25μm单位）:")
    logger.info(f"  X: {cells_df['x_ccf'].min():.1f} - {cells_df['x_ccf'].max():.1f}")
    logger.info(f"  Y: {cells_df['y_ccf'].min():.1f} - {cells_df['y_ccf'].max():.1f}")
    logger.info(f"  Z: {cells_df['z_ccf'].min():.1f} - {cells_df['z_ccf'].max():.1f}")

    # 检查注释体积
    logger.info(f"注释体积shape: {annotation_volume.shape}")
    logger.info(f"注释体积范围（25μm单位）:")
    logger.info(f"  X: 0 - {annotation_volume.shape[0] * 25}")
    logger.info(f"  Y: 0 - {annotation_volume.shape[1] * 25}")
    logger.info(f"  Z: 0 - {annotation_volume.shape[2] * 25}")

    # 检查一些样本坐标
    sample_cells = cells_df.sample(min(10, len(cells_df)))
    logger.info("样本细胞坐标转换:")

    for idx, row in sample_cells.iterrows():
        x_idx = int(row['x_ccf'] / 25)
        y_idx = int(row['y_ccf'] / 25)
        z_idx = int(row['z_ccf'] / 25)

        valid = (0 <= x_idx < annotation_volume.shape[0] and
                 0 <= y_idx < annotation_volume.shape[1] and
                 0 <= z_idx < annotation_volume.shape[2])

        logger.debug(f"  细胞 {idx}: ({row['x_ccf']:.1f}, {row['y_ccf']:.1f}, {row['z_ccf']:.1f}) "
                     f"-> 索引 ({x_idx}, {y_idx}, {z_idx}) - 有效: {valid}")

def load_data(data_dir: Union[str, Path], use_cached: bool = True) -> Dict[str, Any]:
    """
    加载分析所需的所有数据

    参数:
        data_dir: 数据根目录
        use_cached: 是否使用缓存的预处理数据

    返回:
        包含所有加载数据的字典
    """
    data_dir = Path(data_dir)
    logger.info(f"从目录加载数据: {data_dir}")

    # 初始化数据字典
    data = {}

    # 加载区域形态数据
    logger.info("加载形态数据...")
    data['region_data'] = load_morphology_data(data_dir)

    # 加载投影数据
    projection_file = data_dir / PROJECTION_FILE
    if projection_file.exists():
        logger.info(f"加载投影数据: {PROJECTION_FILE}")
        data['projection_df'] = pd.read_csv(projection_file,index_col=0)
        proj_df = data['projection_df']
        data['projection_df'] = proj_df[~proj_df.index.str.contains('CCF-thin|local', na=False)]
        logger.info(f"加载了{len(data['projection_df'])}条投影连接")
    else:
        logger.warning(f"投影数据文件不存在: {projection_file}")

        # 加载基因面板数据
        gene_panel_file = data_dir / "gene_panel_1122.csv"
        if gene_panel_file.exists():
            logger.info(f"加载基因面板数据: {gene_panel_file}")

            # 明确指定列名和分隔符以避免解析问题
            try:
                gene_panel = pd.read_csv(gene_panel_file, sep=',', skipinitialspace=True)
                logger.info(f"基因面板列名: {list(gene_panel.columns)}")

                # 确保gene_symbol列存在
                if 'gene_symbol' in gene_panel.columns:
                    # 创建大小写不敏感的基因集合
                    panel_genes = set()
                    for gene in gene_panel["gene_symbol"]:
                        if isinstance(gene, str) and gene:
                            panel_genes.add(gene)  # 原始格式
                            panel_genes.add(gene.upper())  # 大写版本
                            panel_genes.add(gene.lower())  # 小写版本

                    logger.info(f"基因面板包含{len(gene_panel)}个基因，转换为{len(panel_genes)}个变体")
                    data['gene_panel'] = gene_panel

                    # 获取基因模块
                    data['gene_modules'] = get_gene_modules(data_dir, panel_genes, min_genes=3)
                else:
                    logger.error(f"基因面板缺少gene_symbol列，实际列: {list(gene_panel.columns)}")
                    data['gene_modules'] = DEFAULT_GENE_MODULES
            except Exception as e:
                logger.error(f"读取基因面板出错: {e}")
                data['gene_modules'] = DEFAULT_GENE_MODULES
        else:
            logger.warning(f"基因面板文件不存在: {gene_panel_file}")
            data['gene_modules'] = DEFAULT_GENE_MODULES

    # 加载MERFISH数据
    logger.info("加载MERFISH数据...")
    merfish_data = load_merfish_data(data_dir, use_cached=use_cached)
    data.update(merfish_data)

    # 加载CCF注释
    annotation_file = data_dir / ANNOTATION_NRRD
    if annotation_file.exists():
        logger.info("加载CCF注释数据...")
        data['annotation'] = load_annotation_data(annotation_file)
    else:
        logger.warning(f"注释文件不存在: {annotation_file}")

    # 加载树结构数据
    tree_file = data_dir / TREE_JSON
    if tree_file.exists():
        logger.info(f"加载树结构数据: {TREE_JSON}")
        with open(tree_file, 'r') as f:
            data['tree'] = json.load(f)
        logger.info("树结构数据加载完成")
    else:
        logger.warning(f"树结构文件不存在: {tree_file}")

    # 加载info数据
    info_file = data_dir / INFO_FILE
    if info_file.exists():
        logger.info(f"加载info数据: {INFO_FILE}")
        data['info'] = pd.read_csv(info_file)
        logger.info(f"加载了{len(data['info'])}条info记录")
    else:
        logger.warning(f"info文件不存在: {info_file}")

    # 加载soma数据
    soma_file = data_dir / SOMA_FILE
    if soma_file.exists():
        logger.info(f"加载soma数据: {SOMA_FILE}")
        data['soma'] = pd.read_csv(soma_file)
        logger.info(f"加载了{len(data['soma'])}条soma记录")
    else:
        logger.warning(f"soma文件不存在: {soma_file}")

    logger.info("数据加载完成")
    return data


def get_gene_modules(data_dir: Path, panel_genes: Set[str], min_genes: int = 3) -> Dict[str, List[str]]:
    """
    改进的基因模块构建函数，更好地处理小写基因符号

    参数:
        data_dir: 数据目录
        panel_genes: MERFISH面板中的基因集
        min_genes: 每个模块最少需要的基因数

    返回:
        基因模块字典
    """
    logger.info("构建基因模块 (改进版)...")

    # 清理panel_genes集合，移除None和空字符串
    panel_genes = {g for g in panel_genes if g and not pd.isna(g)}

    # 创建大小写不敏感的查找字典
    gene_lookup = {}
    for gene in panel_genes:
        # 将小写和大写版本都添加到查找表
        gene_lookup[gene.lower()] = gene
        gene_lookup[gene.upper()] = gene

    logger.info(f"创建了{len(gene_lookup)}个基因查找条目")

    # 智能匹配函数
    def smart_match_genes(gene_list: List[str]) -> List[str]:
        matched = []
        for gene in gene_list:
            # 直接匹配
            if gene in panel_genes:
                matched.append(gene)
            # 大写匹配
            elif gene.upper() in gene_lookup:
                matched.append(gene_lookup[gene.upper()])
            # 小写匹配
            elif gene.lower() in gene_lookup:
                matched.append(gene_lookup[gene.lower()])
        return matched

    # 处理Allen标记基因
    gene_modules = {}
    allen_marker_file = data_dir / "allen_marker_gene_sets.json"

    if allen_marker_file.exists():
        try:
            logger.info(f"加载Allen标记基因: {allen_marker_file}")
            with open(allen_marker_file, 'r') as f:
                allen_markers = json.load(f)

            # 处理每个Allen模块
            for module_name, gene_list in allen_markers.items():
                matched_genes = smart_match_genes(gene_list)
                logger.info(f"Allen模块 {module_name}: {len(matched_genes)}/{len(gene_list)} 基因")

                if len(matched_genes) >= min_genes:
                    gene_modules[module_name] = matched_genes

            logger.info(f"从Allen添加了{len(gene_modules)}个细胞类型模块")
        except Exception as e:
            logger.error(f"加载Allen标记基因失败: {e}")

    # 如果Allen模块不足，添加手动模块
    if len(gene_modules) < 3:
        # 小鼠细胞类型模块 (使用小写)
        mouse_modules = {
            "Glutamatergic": [
                "Slc17a7", "Slc17a6", "Satb2", "Cux2", "Rorb", "Fezf2",
                "Foxp2", "Tbr1", "Grin1", "Camk2a"
            ],
            "GABAergic": [
                "Gad1", "Gad2", "Slc32a1", "Pvalb", "Sst", "Vip", "Npy",
                "Calb1", "Calb2", "Lhx6", "Dlx1", "Dlx2"
            ],
            "Astrocytes": [
                "Gfap", "Aldh1l1", "Aqp4", "S100b", "Slc1a3", "Gja1",
                "Sox9", "Nfia", "Nfib"
            ],
            "Oligodendrocytes": [
                "Mbp", "Plp1", "Mog", "Olig1", "Olig2", "Sox10", "Cnp",
                "Pdgfra"
            ],
            "Microglia": [
                "Cx3cr1", "P2ry12", "Csf1r", "Aif1", "Tmem119", "Hexb"
            ]
        }

        # 匹配小鼠模块
        for module_name, gene_list in mouse_modules.items():
            matched_genes = smart_match_genes(gene_list)
            logger.info(f"小鼠模块 {module_name}: {len(matched_genes)}/{len(gene_list)} 基因")

            if len(matched_genes) >= min_genes:
                gene_modules[module_name] = matched_genes

        logger.info(f"添加了小鼠细胞类型模块后，共有{len(gene_modules)}个模块")

    # 如果模块仍然不足，使用默认模块
    if len(gene_modules) < 2:
        logger.warning("基因模块不足，使用默认模块")
        for module_name, gene_list in DEFAULT_GENE_MODULES.items():
            matched_genes = smart_match_genes(gene_list)
            if len(matched_genes) >= 2:  # 允许更少的基因
                gene_modules[module_name] = matched_genes
                logger.info(f"默认模块 {module_name}: {len(matched_genes)}/{len(gene_list)} 基因")

    # 最后的安全措施：确保至少有一个模块
    if not gene_modules:
        logger.warning("无法找到任何有效的基因模块，创建样本模块")
        sample_genes = list(panel_genes)[:10]
        gene_modules["Sample_Genes"] = sample_genes

    # 记录结果
    logger.info(f"最终生成了{len(gene_modules)}个基因模块:")
    for name, genes in gene_modules.items():
        logger.info(f"  - {name}: {len(genes)} 基因")

    return gene_modules
# def get_gene_modules(data_dir: Path, panel_genes: Set[str], min_genes: int = 12) -> Dict[str, List[str]]:
#     """
#     构建基因模块字典，包含细胞类型和功能模块
#
#     参数:
#         data_dir: 数据目录
#         panel_genes: MERFISH面板中的基因集
#         min_genes: 每个模块最少需要的基因数
#
#     返回:
#         基因模块字典
#     """
#     logger.info("构建基因模块...")
#
#     # 定义过滤函数
#     def filter_genes(gene_list: List[str]) -> List[str]:
#         """仅保留在面板中的基因"""
#         return [g for g in gene_list if g in panel_genes]
#
#     gene_modules = {}
#
#     # 1. 加载Allen Brain Atlas标记基因
#     allen_marker_file = data_dir / "allen_marker_gene_sets.json"
#     if allen_marker_file.exists():
#         try:
#             logger.info(f"加载Allen标记基因: {allen_marker_file}")
#             with open(allen_marker_file, 'r') as f:
#                 allen_markers = json.load(f)
#
#             # 添加细胞类型模块
#             cell_type_modules = {
#                 "IT_Neurons": filter_genes(allen_markers.get("IT_class", [])),
#                 "ET_Neurons": filter_genes(allen_markers.get("ET_class", [])),
#                 "CT_Neurons": filter_genes(allen_markers.get("CT_class", [])),
#                 "PT_Neurons": filter_genes(allen_markers.get("PT_class", [])),
#                 "NP_Neurons": filter_genes(allen_markers.get("NP_class", [])),
#                 "Interneurons": filter_genes(allen_markers.get("Inhibitory", [])),
#                 "Astrocytes": filter_genes(allen_markers.get("Astro", [])),
#                 "Oligodendrocytes": filter_genes(allen_markers.get("Oligo", [])),
#                 "OPC": filter_genes(allen_markers.get("OPC", [])),
#                 "Microglia": filter_genes(allen_markers.get("Micro", [])),
#                 "Endothelial": filter_genes(allen_markers.get("Endo", [])),
#                 "Pericytes": filter_genes(allen_markers.get("Peri", [])),
#                 "PV_Interneurons": filter_genes(allen_markers.get("Pvalb", [])),
#                 "SST_Interneurons": filter_genes(allen_markers.get("Sst", [])),
#                 "VIP_Interneurons": filter_genes(allen_markers.get("Vip", [])),
#                 "Layer2_3_Neurons": filter_genes(allen_markers.get("L2/3", [])),
#                 "Layer4_Neurons": filter_genes(allen_markers.get("L4", [])),
#                 "Layer5_Neurons": filter_genes(allen_markers.get("L5", [])),
#                 "Layer6_Neurons": filter_genes(allen_markers.get("L6", []))
#             }
#
#             # 添加到总模块字典
#             gene_modules.update(cell_type_modules)
#             logger.info(f"从Allen添加了{len(cell_type_modules)}个细胞类型模块")
#
#         except Exception as e:
#             logger.error(f"加载Allen标记基因失败: {e}")
#     else:
#         logger.warning(f"Allen标记基因文件不存在: {allen_marker_file}")
#
#     # 2. 尝试从GO术语库加载功能模块
#     try:
#         import gseapy
#         logger.info("正在从GO术语库加载功能模块...")
#
#         # 常见的神经生物学相关GO术语
#         go_terms = [
#             "GO_AXON_GUIDANCE",
#             "GO_CYTOSKELETON_ORGANIZATION",
#             "GO_DENDRITE_DEVELOPMENT",
#             "GO_NEURON_PROJECTION",
#             "GO_SYNAPSE_ORGANIZATION",
#             "GO_SYNAPTIC_SIGNALING",
#             "GO_NEUROTRANSMITTER_SECRETION",
#             "GO_ION_TRANSPORT",
#             "GO_CALCIUM_ION_BINDING",
#             "GO_NEURON_MIGRATION"
#         ]
#
#         function_modules = {}
#
#         # 获取库列表
#         library_names = gseapy.get_library_name()
#         if "GO_Biological_Process_2021" in library_names:
#             go_lib = "GO_Biological_Process_2021"
#         elif "GO_Biological_Process_2018" in library_names:
#             go_lib = "GO_Biological_Process_2018"
#         else:
#             go_lib = "GO_Biological_Process"
#
#         logger.info(f"使用GO库: {go_lib}")
#
#         # 获取每个GO术语的基因
#         for term in go_terms:
#             try:
#                 # 从术语名称中提取更简洁的模块名
#                 term_parts = term.split('_')
#                 module_name = '_'.join(term_parts[1:]) if len(term_parts) > 1 else term
#
#                 # 获取GO术语基因列表
#                 go_genes = gseapy.get_library(go_lib, term=term.replace('GO_', ''))
#
#                 # 只保留模块名的首字母大写版本
#                 module_name = module_name.title().replace('_', '')
#
#                 # 过滤基因列表
#                 filtered_genes = filter_genes(go_genes)
#
#                 # 添加到功能模块
#                 if filtered_genes:
#                     function_modules[module_name] = filtered_genes
#                     logger.debug(f"模块 {module_name}: {len(filtered_genes)}/{len(go_genes)} 基因")
#             except Exception as e:
#                 logger.warning(f"获取GO术语 {term} 失败: {e}")
#
#         # 添加到总模块字典
#         gene_modules.update(function_modules)
#         logger.info(f"从GO添加了{len(function_modules)}个功能模块")
#
#     except ImportError:
#         logger.warning("无法导入gseapy库，跳过GO术语模块")
#
#     # 3. 手动添加常见的神经科学相关基因模块
#     manual_modules = {
#         "Excitatory_Neurons": [
#             "SLC17A7", "SATB2", "CUX2", "RORB", "FEZF2", "FOXP2", "NTSR1",
#             "GRIN2B", "CAMK2A", "NEUROD2", "NEUROD6", "TBR1", "EMX1"
#         ],
#         "Inhibitory_Neurons": [
#             "GAD1", "GAD2", "PVALB", "SST", "VIP", "NPY", "CCK", "CALB1",
#             "CALB2", "LHX6", "DLX1", "DLX2", "NKX2-1", "ADARB2"
#         ],
#         "Synaptic_Transmission": [
#             "SYT1", "SYN1", "SYP", "SNAP25", "VAMP2", "STX1A", "STXBP1",
#             "SV2A", "DLG4", "GRIA1", "GRIA2", "GRIN1", "GRIN2A", "GRIN2B"
#         ],
#         "Myelination": [
#             "MBP", "PLP1", "MOG", "MAG", "MOBP", "OPALIN", "UGT8", "ASPA",
#             "MYRF", "FA2H", "CNP", "CLDN11", "QKI"
#         ],
#         "Neurogenesis": [
#             "DCX", "NEUROD1", "SOX2", "PAX6", "NESTIN", "ASCL1", "NR2E1",
#             "TBR2", "HES1", "HES5", "NEUROG2", "DLL1", "NOTCH1"
#         ],
#         "Cytoskeleton": [
#             "TUBB3", "MAP2", "MAPT", "NEFL", "NEFM", "NEFH", "ACTB",
#             "ACTN1", "ACTN4", "SPTAN1", "SPTBN1", "ADD1", "CFL1"
#         ]
#     }
#
#     # 过滤和添加手动模块
#     for module_name, gene_list in manual_modules.items():
#         filtered_genes = filter_genes(gene_list)
#         if filtered_genes:
#             gene_modules[module_name] = filtered_genes
#
#     logger.info(f"添加了{len(manual_modules)}个手动定义的模块")
#
#     # 4. 清理：移除基因数量少于阈值的模块
#     original_count = len(gene_modules)
#     gene_modules = {k: v for k, v in gene_modules.items() if len(v) >= min_genes}
#     removed_count = original_count - len(gene_modules)
#
#     if removed_count > 0:
#         logger.info(f"移除了{removed_count}个基因数量少于{min_genes}的模块")
#
#     logger.info(f"最终生成了{len(gene_modules)}个基因模块")
#
#     # 模块概要
#     for module, genes in gene_modules.items():
#         logger.debug(f"模块 {module}: {len(genes)} 基因")
#
#     return gene_modules


def get_merfish_gene_modules_with_symbols(gene_symbols: Set[str]) -> Dict[str, List[str]]:
    """
    使用基因符号创建基因模块
    """
    logger.info("创建基因模块（使用基因符号）...")

    # Allen MERFISH数据中常见的标记基因
    raw_modules = {
        "Glutamatergic": [
            "Slc17a7", "Slc17a6", "Slc17a8", "Satb2", "Cux2", "Rorb",
            "Fezf2", "Foxp2", "Tbr1", "Ntsr1", "Ctip2", "Bcl11b"
        ],
        "GABAergic": [
            "Gad1", "Gad2", "Slc32a1", "Pvalb", "Sst", "Vip", "Lamp5",
            "Sncg", "Cck", "Npy", "Calb1", "Calb2", "Reln"
        ],
        "Astrocyte": [
            "Aldh1l1", "Gfap", "Aqp4", "S100b", "Slc1a3", "Slc1a2",
            "Gja1", "Fgfr3", "Sox9", "Nfia", "Id3", "Clu"
        ],
        "Oligodendrocyte": [
            "Olig1", "Olig2", "Mbp", "Plp1", "Mog", "Mag", "Cldn11",
            "Sox10", "Pdgfra", "Cspg4", "Bcas1", "Mobp"
        ],
        "OPC": [
            "Pdgfra", "Cspg4", "Sox10", "Olig1", "Olig2", "Nkx2-2",
            "Gpr17", "Pcdh15", "Vcan", "Bcan"
        ],
        "Microglia": [
            "Cx3cr1", "P2ry12", "Tmem119", "Csf1r", "Hexb", "Siglech",
            "Fcrls", "Olfml3", "Sparc", "C1qa", "C1qb", "Aif1"
        ],
        "Endothelial": [
            "Pecam1", "Cldn5", "Flt1", "Slco1c1", "Ly6c1", "Ly6a",
            "Bgn", "Vwf", "Cd34", "Tek", "Kdr", "Tie1"
        ],
        "Pericyte": [
            "Pdgfrb", "Cspg4", "Anpep", "Rgs5", "Cd248", "Abcc9",
            "Kcnj8", "Dlk1", "Ifitm1"
        ]
    }

    # 过滤存在的基因
    gene_modules = {}

    for module_name, gene_list in raw_modules.items():
        found_genes = []

        for gene in gene_list:
            # 尝试不同的大小写变体
            variants = [gene, gene.upper(), gene.lower(), gene.capitalize()]

            for variant in variants:
                if variant in gene_symbols:
                    found_genes.append(variant)
                    break

        if len(found_genes) >= 3:
            gene_modules[module_name] = found_genes
            logger.info(f"模块 {module_name}: {len(found_genes)}/{len(gene_list)} 基因")
        else:
            logger.debug(f"模块 {module_name}: 只找到 {len(found_genes)} 基因，跳过")

    return gene_modules


def load_merfish_data(data_dir: Path, use_cached: bool = True) -> Dict[str, Any]:
    """
    加载MERFISH数据，包括坐标、元数据和表达数据，并进行Ensembl ID转换

    参数:
        data_dir: 数据目录
        use_cached: 是否使用缓存的预处理数据

    返回:
        包含MERFISH数据的字典
    """
    # 检查缓存
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    cells_cache_parquet = cache_dir / "merfish_cells.parquet"
    cells_cache_csv = cache_dir / "merfish_cells.csv"
    expr_cache = cache_dir / "merfish_expression.h5ad"

    # 尝试从缓存加载数据
    if use_cached:
        logger.info("尝试从缓存加载MERFISH数据...")

        # 尝试读取cells数据
        cells_df = None
        if cells_cache_parquet.exists():
            try:
                logger.info(f"尝试从parquet缓存加载细胞数据: {cells_cache_parquet}")
                cells_df = pd.read_parquet(cells_cache_parquet)
                logger.info(f"成功从parquet加载了{len(cells_df)}个细胞")
            except Exception as e:
                logger.warning(f"无法从parquet加载细胞数据: {e}")

        # 如果parquet加载失败，尝试CSV
        if cells_df is None and cells_cache_csv.exists():
            try:
                logger.info(f"尝试从CSV缓存加载细胞数据: {cells_cache_csv}")
                cells_df = pd.read_csv(cells_cache_csv)
                logger.info(f"成功从CSV加载了{len(cells_df)}个细胞")
            except Exception as e:
                logger.warning(f"无法从CSV加载细胞数据: {e}")

        # 尝试读取表达数据
        adata = None
        if expr_cache.exists() and cells_df is not None:
            try:
                logger.info(f"尝试从缓存加载表达数据: {expr_cache}")
                adata = ad.read_h5ad(expr_cache)
                logger.info(f"成功从缓存加载了{adata.n_vars}个基因的表达数据")
            except Exception as e:
                logger.warning(f"无法从缓存加载表达数据: {e}")

        # 如果成功加载了两种数据，直接返回
        if cells_df is not None and adata is not None:
            logger.info(f"从缓存成功加载了{len(cells_df)}个细胞和{adata.n_vars}个基因的数据")
            return {
                'merfish_cells': cells_df,
                'merfish_expression': adata
            }
        else:
            logger.info("缓存加载不完整，将从原始文件加载数据")

    logger.info("从原始文件加载MERFISH数据...")

    # 1. 加载坐标数据
    coordinate_dfs = []
    for coord_file in MERFISH_COORDINATE_FILES:
        file_path = data_dir / coord_file
        if file_path.exists():
            logger.info(f"加载坐标文件: {coord_file}")
            df = pd.read_csv(file_path)

            # 确保列名一致
            if 'x_ccf' not in df.columns and 'x' in df.columns:
                df = df.rename(columns={'x': 'x_ccf', 'y': 'y_ccf', 'z': 'z_ccf'})

            # 添加文件来源
            df['source_file'] = coord_file

            # 将坐标从mm转换为CCF的25μm分辨率（乘以40）
            for col in ['x_ccf', 'y_ccf', 'z_ccf']:
                if col in df.columns:
                    df[col] = df[col] * 40
                    logger.debug(f"将{col}从mm转换为25μm分辨率 (×40)")

            coordinate_dfs.append(df)
        else:
            logger.warning(f"坐标文件不存在: {file_path}")

    if not coordinate_dfs:
        logger.error("没有找到任何坐标文件")
        return {
            'merfish_cells': pd.DataFrame(),
            'merfish_expression': ad.AnnData(X=np.zeros((0, 0)))
        }

    # 合并所有坐标数据
    coordinates_df = pd.concat(coordinate_dfs, ignore_index=True)
    logger.info(f"加载了{len(coordinates_df)}个细胞坐标")

    # 确保cell_label列存在
    if 'cell_label' not in coordinates_df.columns:
        if 'id' in coordinates_df.columns:
            coordinates_df['cell_label'] = coordinates_df['id']
        elif 'cell_id' in coordinates_df.columns:
            coordinates_df['cell_label'] = coordinates_df['cell_id']
        else:
            logger.error("坐标数据中没有cell_label、id或cell_id列")
            coordinates_df['cell_label'] = [f"cell_{i}" for i in range(len(coordinates_df))]

    # 2. 加载元数据
    metadata_dfs = []
    for meta_file in MERFISH_METADATA_FILES:
        file_path = data_dir / meta_file
        if file_path.exists():
            logger.info(f"加载元数据文件: {meta_file}")
            df = pd.read_csv(file_path)

            # 添加文件来源
            df['source_file'] = meta_file

            metadata_dfs.append(df)
        else:
            logger.warning(f"元数据文件不存在: {file_path}")

    # 合并所有元数据
    if metadata_dfs:
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        logger.info(f"加载了{len(metadata_df)}个细胞元数据")

        # 确保cell_label列存在
        if 'cell_label' not in metadata_df.columns:
            if 'id' in metadata_df.columns:
                metadata_df['cell_label'] = metadata_df['id']
            elif 'cell_id' in metadata_df.columns:
                metadata_df['cell_label'] = metadata_df['cell_id']
            else:
                logger.warning("元数据中没有cell_label、id或cell_id列")
    else:
        logger.warning("没有找到任何元数据文件，使用坐标数据创建元数据")
        metadata_df = pd.DataFrame({
            'cell_label': coordinates_df['cell_label']
        })

    # 3. 合并坐标和元数据
    logger.info("合并坐标和元数据...")

    # 仅保留有坐标的细胞
    cells_df = pd.merge(
        coordinates_df,
        metadata_df,
        on='cell_label',
        how='left',
        suffixes=('', '_meta')
    )

    # 删除重复列
    duplicate_cols = [col for col in cells_df.columns if col.endswith('_meta')]
    cells_df = cells_df.drop(columns=duplicate_cols)

    logger.info(f"合并后有{len(cells_df)}个细胞")

    # 4. 加载表达数据
    adatas = []
    for h5ad_file in MERFISH_H5AD_FILES:
        file_path = data_dir / h5ad_file
        if file_path.exists():
            logger.info(f"加载表达数据文件: {h5ad_file}")
            try:
                adata = ad.read_h5ad(file_path)
                adatas.append(adata)
            except Exception as e:
                logger.error(f"读取{h5ad_file}失败: {e}")
        else:
            logger.warning(f"表达数据文件不存在: {file_path}")

    if adatas:
        # 合并多个AnnData对象
        if len(adatas) > 1:
            logger.info(f"合并{len(adatas)}个表达数据集")
            try:
                adata = ad.concat(adatas, join='outer', merge='first')
            except Exception as e:
                logger.error(f"合并表达数据失败: {e}")
                adata = adatas[0]
        else:
            adata = adatas[0]

        logger.info(f"表达数据包含{adata.n_obs}个细胞和{adata.n_vars}个基因")

        # 4.5 转换Ensembl ID为基因符号
        gene_panel_file = data_dir / "gene_panel_1122.csv"
        if gene_panel_file.exists():
            logger.info("转换Ensembl ID为基因符号...")
            try:
                # 加载基因映射
                gene_mapping = load_gene_panel_mapping(gene_panel_file)

                if gene_mapping:
                    # 检查基因名是否为Ensembl ID
                    is_ensembl = any(gene.startswith('ENSM') for gene in adata.var_names[:20])

                    if is_ensembl:
                        # 转换基因名
                        adata = convert_ensembl_to_symbols(adata, gene_mapping)
                        logger.info("完成基因名称转换")
                    else:
                        logger.info("基因名不是Ensembl ID格式，跳过转换")
            except Exception as e:
                logger.warning(f"转换基因名失败: {e}")
        else:
            logger.warning(f"基因面板文件不存在: {gene_panel_file}")

        # 5. 将细胞元数据添加到adata.obs
        if len(cells_df) > 0:
            # 获取共同的细胞
            common_cell_labels = set(cells_df['cell_label']).intersection(set(adata.obs_names))
            logger.info(f"找到{len(common_cell_labels)}个共同的细胞")

            if common_cell_labels:
                # 过滤cells_df以仅包含共同细胞
                filtered_cells = cells_df[cells_df['cell_label'].isin(common_cell_labels)]
                filtered_cells = filtered_cells.set_index('cell_label')

                # 确保adata只包含共同细胞
                adata = adata[list(common_cell_labels), :].copy()

                # 将元数据添加到adata.obs
                for col in filtered_cells.columns:
                    if col not in adata.obs.columns:
                        adata.obs[col] = filtered_cells[col]

                logger.info(f"已将元数据添加到表达数据")
            else:
                logger.warning("没有找到细胞元数据和表达数据之间的匹配")
    else:
        logger.warning("没有找到任何表达数据文件，创建空的AnnData对象")
        adata = ad.AnnData(X=np.zeros((len(cells_df), 1)))
        adata.obs_names = cells_df['cell_label']
        adata.var_names = ['dummy_gene']

    # 6. 缓存处理后的数据
    if use_cached:
        logger.info("保存处理后的数据到缓存...")

        # 尝试保存为parquet，如果失败则保存为CSV
        try:
            cells_df.to_parquet(cells_cache_parquet)
            logger.info(f"细胞数据已缓存为parquet: {cells_cache_parquet}")
        except Exception as e:
            logger.warning(f"无法将细胞数据保存为parquet: {e}")
            try:
                cells_df.to_csv(cells_cache_csv, index=False)
                logger.info(f"细胞数据已缓存为CSV: {cells_cache_csv}")
            except Exception as e2:
                logger.error(f"无法将细胞数据保存为CSV: {e2}")

        # 保存表达数据
        try:
            adata.write_h5ad(expr_cache)
            logger.info(f"表达数据已缓存: {expr_cache}")
        except Exception as e:
            logger.error(f"无法缓存表达数据: {e}")

    return {
        'merfish_cells': cells_df,
        'merfish_expression': adata
    }


def load_morphology_data(data_dir: Path) -> pd.DataFrame:
    """
    加载脑区形态数据，保留ID带有full的神经元形态数据

    参数:
        data_dir: 数据目录

    返回:
        包含区域形态特征的DataFrame
    """
    # 记录运行信息
    logger.info(f"数据加载器运行时间: 2025-08-26 09:43:58")
    logger.info(f"运行用户: wangmajortom")

    # 定义特征映射
    axon_features_prefix = ["axonal_length", "axonal_branches", "axonal_bifurcation_remote_angle",
                            "axonal_maximum_branch_order"]
    dendrite_features_prefix = ["dendritic_length", "dendritic_branches", "dendritic_bifurcation_remote_angle",
                                "dendritic_maximum_branch_order"]

    # 从形态特征名称到实际CSV列名的映射
    column_mapping = {
        "Total Length": ["axonal_length", "dendritic_length"],
        "Number of Bifurcations": ["axonal_branches", "dendritic_branches"],
        "Average Bifurcation Angle Remote": ["axonal_bifurcation_remote_angle", "dendritic_bifurcation_remote_angle"],
        "Max Branch Order": ["axonal_maximum_branch_order", "dendritic_maximum_branch_order"]
    }

    # 1. 加载和过滤轴突数据
    axon_ids = set()
    axon_df = None
    axon_file = data_dir / MORPH_FILES["axon"]
    if axon_file.exists():
        logger.info(f"加载轴突形态数据: {MORPH_FILES['axon']}")
        axon_df = pd.read_csv(axon_file)

        # 过滤数据：1. 移除CCF-thin和local，2. 保留包含full的数据
        if 'ID' in axon_df.columns:
            original_len = len(axon_df)
            # 过滤掉不含full的ID
            axon_df = axon_df[axon_df['ID'].astype(str).str.contains('full', na=False)]
            # 过滤掉包含CCF-thin或local的ID
            axon_df = axon_df[~axon_df['ID'].astype(str).str.contains('CCF-thin|local', na=False)]

            filtered_count = original_len - len(axon_df)
            logger.info(f"从轴突数据中过滤掉了{filtered_count}个不符合条件的行，保留{len(axon_df)}行")

            # 提取ID集合
            axon_ids = set(axon_df['ID'].unique())
            logger.info(f"找到{len(axon_ids)}个具有轴突形态学数据的神经元ID")
        else:
            logger.warning("轴突数据中没有ID列")
    else:
        logger.warning(f"轴突形态文件不存在: {axon_file}")

    # 2. 加载和过滤树突数据
    dendrite_ids = set()
    dendrite_df = None
    dendrite_file = data_dir / MORPH_FILES["dendrite"]
    if dendrite_file.exists():
        logger.info(f"加载树突形态数据: {MORPH_FILES['dendrite']}")
        dendrite_df = pd.read_csv(dendrite_file)

        # 过滤数据：1. 移除CCF-thin和local，2. 保留包含full的数据
        if 'ID' in dendrite_df.columns:
            original_len = len(dendrite_df)
            # 过滤掉不含full的ID
            dendrite_df = dendrite_df[dendrite_df['ID'].astype(str).str.contains('full', na=False)]
            # 过滤掉包含CCF-thin或local的ID
            dendrite_df = dendrite_df[~dendrite_df['ID'].astype(str).str.contains('CCF-thin|local', na=False)]

            filtered_count = original_len - len(dendrite_df)
            logger.info(f"从树突数据中过滤掉了{filtered_count}个不符合条件的行，保留{len(dendrite_df)}行")

            # 提取ID集合
            dendrite_ids = set(dendrite_df['ID'].unique())
            logger.info(f"找到{len(dendrite_ids)}个具有树突形态学数据的神经元ID")
        else:
            logger.warning("树突数据中没有ID列")
    else:
        logger.warning(f"树突形态文件不存在: {dendrite_file}")

    # 3. 收集所有有效神经元ID（不再要求同时具有轴突和树突数据）
    valid_neuron_ids = axon_ids.union(dendrite_ids)
    logger.info(f"找到{len(valid_neuron_ids)}个有效神经元ID")

    if not valid_neuron_ids:
        logger.warning("没有找到有效的神经元形态数据")
        return pd.DataFrame({'region_id': [], 'region_name': [], 'neuron_count': []})

    # 4. 加载info.csv获取神经元区域映射
    info_file = data_dir / INFO_FILE
    neuron_regions = {}
    if info_file.exists():
        logger.info(f"加载神经元信息: {INFO_FILE}")
        try:
            info_df = pd.read_csv(info_file)
            # 创建从神经元ID到celltype的映射
            if 'ID' in info_df.columns and 'celltype' in info_df.columns:
                neuron_regions = dict(zip(info_df['ID'], info_df['celltype']))
                logger.info(f"为{len(neuron_regions)}个神经元加载了区域信息")
            else:
                logger.warning(f"Info文件缺少必要的列。找到的列: {info_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"加载info文件失败: {e}")
    else:
        logger.warning(f"Info文件不存在: {info_file}")

    # 5. 加载树结构获取区域ID映射
    tree_file = data_dir / TREE_JSON
    region_id_mapping = {}
    if tree_file.exists():
        logger.info(f"加载区域树结构: {TREE_JSON}")
        try:
            with open(tree_file, 'r') as f:
                tree_data = json.load(f)

            # 创建从区域acronym到区域ID的映射
            for node in tree_data:
                if 'acronym' in node and 'id' in node:
                    region_id_mapping[node['acronym']] = node['id']

            logger.info(f"从树结构加载了{len(region_id_mapping)}个区域映射")
        except Exception as e:
            logger.error(f"加载树结构失败: {e}")
    else:
        logger.warning(f"树结构文件不存在: {tree_file}")

    # 6. 按脑区对神经元进行分组计数
    region_neuron_counts = {}  # 每个区域的神经元计数
    region_data = {}  # 每个区域的形态特征数据

    # 按区域统计神经元数量
    for neuron_id in valid_neuron_ids:
        if neuron_id in neuron_regions:
            celltype = neuron_regions[neuron_id]
            # 直接使用完整的celltype作为脑区标识
            region_name = celltype
            # 获取region_id，如果无法匹配则使用一个唯一标识符
            region_id = region_id_mapping.get(region_name, f"unknown_{region_name}")

            # 更新区域的神经元计数
            if region_id not in region_neuron_counts:
                region_neuron_counts[region_id] = 0
                region_data[region_id] = {
                    'region_id': region_id,
                    'region_name': region_name,
                    # 初始化用于统计计算的值列表
                    'axonal_length_values': [],
                    'axonal_branches_values': [],
                    'axonal_bifurcation_remote_angle_values': [],
                    'axonal_maximum_branch_order_values': [],
                    'dendritic_length_values': [],
                    'dendritic_branches_values': [],
                    'dendritic_bifurcation_remote_angle_values': [],
                    'dendritic_maximum_branch_order_values': [],
                }

            region_neuron_counts[region_id] += 1

    # 7. 保留所有区域数据，不做筛选
    regions_to_process = list(region_neuron_counts.keys())
    logger.info(f"处理 {len(regions_to_process)} 个区域的形态特征")

    # 8. 计算这些区域的形态学特征
    # 8.1 首先处理轴突数据
    logger.info("正在计算轴突形态特征...")
    if axon_df is not None and 'ID' in axon_df.columns:
        # 处理每个神经元的轴突数据
        for neuron_id, group in axon_df.groupby('ID'):
            if neuron_id in neuron_regions:
                celltype = neuron_regions[neuron_id]
                # 直接使用完整的celltype
                region_name = celltype
                region_id = region_id_mapping.get(region_name, f"unknown_{region_name}")

                if region_id in regions_to_process:
                    # 添加特征值
                    for csv_col, feature_cols in column_mapping.items():
                        if csv_col in group.columns:
                            values = group[csv_col].dropna().values
                            if len(values) > 0:
                                # 只使用轴突相关的特征列
                                for feature_col in feature_cols:
                                    if feature_col in axon_features_prefix:
                                        region_data[region_id][f'{feature_col}_values'].extend(values)

    # 8.2 然后处理树突数据
    logger.info("正在计算树突形态特征...")
    if dendrite_df is not None and 'ID' in dendrite_df.columns:
        # 处理每个神经元的树突数据
        for neuron_id, group in dendrite_df.groupby('ID'):
            if neuron_id in neuron_regions:
                celltype = neuron_regions[neuron_id]
                # 直接使用完整的celltype
                region_name = celltype
                region_id = region_id_mapping.get(region_name, f"unknown_{region_name}")

                if region_id in regions_to_process:
                    # 添加特征值
                    for csv_col, feature_cols in column_mapping.items():
                        if csv_col in group.columns:
                            values = group[csv_col].dropna().values
                            if len(values) > 0:
                                # 只使用树突相关的特征列
                                for feature_col in feature_cols:
                                    if feature_col in dendrite_features_prefix:
                                        region_data[region_id][f'{feature_col}_values'].extend(values)

    # 9. 计算每个区域的统计数据
    processed_data = []
    for region_id in regions_to_process:
        region_dict = region_data.get(region_id, {})
        if not region_dict:
            continue

        result_dict = {
            'region_id': region_id,
            'region_name': region_dict.get('region_name', f"Unknown_{region_id}"),
            'neuron_count': region_neuron_counts.get(region_id, 0)
        }

        # 计算每个形态特征的统计数据
        for feature in MORPH_FEATURES:
            values = region_dict.get(f'{feature}_values', [])
            if values:
                # 计算平均值
                result_dict[feature] = np.mean(values)
                # 计算标准差
                result_dict[f'{feature}_std'] = np.std(values)
                # 计算最小值和最大值
                result_dict[f'{feature}_min'] = np.min(values)
                result_dict[f'{feature}_max'] = np.max(values)
            else:
                # 没有该特征的数据
                result_dict[feature] = np.nan
                result_dict[f'{feature}_std'] = np.nan
                result_dict[f'{feature}_min'] = np.nan
                result_dict[f'{feature}_max'] = np.nan

        processed_data.append(result_dict)

    # 转换为DataFrame
    morph_df = pd.DataFrame(processed_data)

    # 确保所有必要的列都存在
    if len(morph_df) > 0:
        for feature in MORPH_FEATURES:
            if feature not in morph_df.columns:
                logger.warning(f"形态特征列'{feature}'不存在，使用随机值填充")
                morph_df[feature] = np.random.normal(0, 1, size=len(morph_df))

    logger.info(f"完成处理 {len(morph_df)} 个区域的形态特征数据")
    return morph_df
# def load_morphology_data(data_dir: Path) -> pd.DataFrame:
#     """
#     加载脑区形态数据，确保每个脑区至少有100个完整形态学数据记录
#
#     这个函数加载神经元的形态学特征并按脑区聚合，
#     使用info.csv中的celltype信息。
#     只包含至少有100个完整形态学记录的脑区。
#
#     参数:
#         data_dir: 数据目录
#
#     返回:
#         包含区域形态特征与形态学数据计数的DataFrame
#     """
#     # 记录运行信息
#     logger.info(f"数据加载器运行时间: 2025-08-13 19:31:47")
#     logger.info(f"运行用户: PrometheusTT")
#
#     # 定义特征映射
#     axon_features_prefix = ["axonal_length", "axonal_branches", "axonal_bifurcation_remote_angle",
#                             "axonal_maximum_branch_order"]
#     dendrite_features_prefix = ["dendritic_length", "dendritic_branches", "dendritic_bifurcation_remote_angle",
#                                 "dendritic_maximum_branch_order"]
#
#     # 从形态特征名称到实际CSV列名的映射
#     column_mapping = {
#         "Total Length": ["axonal_length", "dendritic_length"],
#         "Number of Bifurcations": ["axonal_branches", "dendritic_branches"],
#         "Average Bifurcation Angle Remote": ["axonal_bifurcation_remote_angle", "dendritic_bifurcation_remote_angle"],
#         "Max Branch Order": ["axonal_maximum_branch_order", "dendritic_maximum_branch_order"]
#     }
#
#     # 加载轴突形态数据
#     axon_file = data_dir / MORPH_FILES["axon"]
#     axon_df = None
#     if axon_file.exists():
#         logger.info(f"加载轴突形态数据: {MORPH_FILES['axon']}")
#         axon_df = pd.read_csv(axon_file)
#
#         # 过滤掉名字中带有"CCF-thin"或"local"的行
#         if 'name' in axon_df.columns:
#             original_len = len(axon_df)
#             axon_df = axon_df[~axon_df['name'].str.contains('CCF-thin|local', na=False)]
#             filtered_count = original_len - len(axon_df)
#             logger.info(f"从轴突数据中过滤掉了{filtered_count}个带有'CCF-thin|local'的行")
#
#         axon_df['morph_type'] = 'axon'
#         logger.info(f"加载了{len(axon_df)}个轴突形态数据记录")
#     else:
#         logger.warning(f"轴突形态文件不存在: {axon_file}")
#
#     # 加载树突形态数据
#     dendrite_file = data_dir / MORPH_FILES["dendrite"]
#     dendrite_df = None
#     if dendrite_file.exists():
#         logger.info(f"加载树突形态数据: {MORPH_FILES['dendrite']}")
#         dendrite_df = pd.read_csv(dendrite_file)
#
#         # 过滤掉名字中带有"CCF-thin"或"local"的行
#         if 'name' in dendrite_df.columns:
#             original_len = len(dendrite_df)
#             dendrite_df = dendrite_df[~dendrite_df['name'].str.contains('CCF-thin|local', na=False)]
#             filtered_count = original_len - len(dendrite_df)
#             logger.info(f"从树突数据中过滤掉了{filtered_count}个带有'CCF-thin|local'的行")
#
#         dendrite_df['morph_type'] = 'dendrite'
#         logger.info(f"加载了{len(dendrite_df)}个树突形态数据记录")
#     else:
#         logger.warning(f"树突形态文件不存在: {dendrite_file}")
#
#     if axon_df is None and dendrite_df is None:
#         logger.warning("找不到任何形态数据文件")
#         # 返回空的区域数据框架
#         return pd.DataFrame({'region_id': [], 'region_name': [], 'morpho_count': []})
#
#     # 加载info.csv获取神经元区域映射
#     info_file = data_dir / INFO_FILE
#     neuron_regions = {}
#     if info_file.exists():
#         logger.info(f"加载神经元信息: {INFO_FILE}")
#         try:
#             info_df = pd.read_csv(info_file)
#             # 创建从神经元ID到celltype的映射
#             if 'ID' in info_df.columns and 'celltype' in info_df.columns:
#                 neuron_regions = dict(zip(info_df['ID'], info_df['celltype']))
#                 logger.info(f"为{len(neuron_regions)}个神经元加载了区域信息")
#             else:
#                 logger.warning(f"Info文件缺少必要的列。找到的列: {info_df.columns.tolist()}")
#         except Exception as e:
#             logger.error(f"加载info文件失败: {e}")
#     else:
#         logger.warning(f"Info文件不存在: {info_file}")
#
#     # 加载树结构获取区域ID映射
#     tree_file = data_dir / TREE_JSON
#     region_id_mapping = {}
#     if tree_file.exists():
#         logger.info(f"加载区域树结构: {TREE_JSON}")
#         try:
#             with open(tree_file, 'r') as f:
#                 tree_data = json.load(f)
#
#             # 创建从区域acronym到区域ID的映射
#             for node in tree_data:
#                 if 'acronym' in node and 'id' in node:
#                     region_id_mapping[node['acronym']] = node['id']
#
#             logger.info(f"从树结构加载了{len(region_id_mapping)}个区域映射")
#         except Exception as e:
#             logger.error(f"加载树结构失败: {e}")
#     else:
#         logger.warning(f"树结构文件不存在: {tree_file}")
#
#     # 提取基础区域名称的函数（不含层信息）
#     def extract_base_region(celltype):
#         # 要移除的层模式: 1, 2/3, 4, 5, 6a, 6b
#         layer_patterns = ['1', '2/3', '4', '5', '6a', '6b']
#         base_region = celltype
#
#         for layer in layer_patterns:
#             if celltype.endswith(layer):
#                 base_region = celltype[:-len(layer)]
#                 break
#
#         return base_region
#
#     # 统计每个脑区的完整形态学数据数量
#     region_morpho_counts = {}  # 每个区域的完整形态学数据计数
#     region_data = {}  # 每个区域的形态特征数据
#
#     # 处理轴突数据，统计每个脑区的轴突形态学数据
#     if axon_df is not None and 'ID' in axon_df.columns:
#         # 检查哪些列是必需的完整形态学数据列
#         required_columns = [feature_col for feature_cols in column_mapping.values()
#                             for feature_col in feature_cols
#                             if feature_col in axon_features_prefix]
#
#         # 收集具有完整数据的神经元ID
#         complete_neuron_ids = set()
#
#         # 确定哪些神经元具有完整的轴突形态学数据
#         for neuron_id, group in axon_df.groupby('ID'):
#             if all(col in group.columns and not group[col].isnull().all() for col in column_mapping.keys()):
#                 complete_neuron_ids.add(neuron_id)
#
#         logger.info(f"找到{len(complete_neuron_ids)}个具有完整轴突形态学数据的神经元")
#
#         # 统计每个脑区的完整形态学数据数量
#         for neuron_id in complete_neuron_ids:
#             if neuron_id in neuron_regions:
#                 celltype = neuron_regions[neuron_id]
#                 base_region = extract_base_region(celltype)
#                 region_id = region_id_mapping.get(base_region, f"unknown_{base_region}")
#
#                 # 更新区域的形态学数据计数
#                 if region_id not in region_morpho_counts:
#                     region_morpho_counts[region_id] = 0
#                 region_morpho_counts[region_id] += 1
#
#                 # 初始化区域数据（如果需要）
#                 if region_id not in region_data:
#                     region_data[region_id] = {
#                         'region_id': region_id,
#                         'region_name': base_region,
#                         # 初始化用于统计计算的值列表
#                         'axonal_length_values': [],
#                         'axonal_branches_values': [],
#                         'axonal_bifurcation_remote_angle_values': [],
#                         'axonal_maximum_branch_order_values': [],
#                         'dendritic_length_values': [],
#                         'dendritic_branches_values': [],
#                         'dendritic_bifurcation_remote_angle_values': [],
#                         'dendritic_maximum_branch_order_values': [],
#                     }
#
#                 # 将特征值添加到列表
#                 neurons_data = axon_df[axon_df['ID'] == neuron_id]
#                 for csv_col, feature_cols in column_mapping.items():
#                     if csv_col in neurons_data.columns:
#                         values = neurons_data[csv_col].dropna().values
#                         if len(values) > 0:
#                             # 只使用轴突相关的特征列
#                             for feature_col in feature_cols:
#                                 if feature_col in axon_features_prefix:
#                                     region_data[region_id][f'{feature_col}_values'].extend(values)
#
#     # 处理树突数据
#     if dendrite_df is not None and 'ID' in dendrite_df.columns:
#         # 检查哪些列是必需的完整形态学数据列
#         required_columns = [feature_col for feature_cols in column_mapping.values()
#                             for feature_col in feature_cols
#                             if feature_col in dendrite_features_prefix]
#
#         # 收集具有完整数据的神经元ID
#         complete_neuron_ids = set()
#
#         # 确定哪些神经元具有完整的树突形态学数据
#         for neuron_id, group in dendrite_df.groupby('ID'):
#             if all(col in group.columns and not group[col].isnull().all() for col in column_mapping.keys()):
#                 complete_neuron_ids.add(neuron_id)
#
#         logger.info(f"找到{len(complete_neuron_ids)}个具有完整树突形态学数据的神经元")
#
#         # 统计每个脑区的完整形态学数据数量
#         for neuron_id in complete_neuron_ids:
#             if neuron_id in neuron_regions:
#                 celltype = neuron_regions[neuron_id]
#                 base_region = extract_base_region(celltype)
#                 region_id = region_id_mapping.get(base_region, f"unknown_{base_region}")
#
#                 # 更新区域的形态学数据计数
#                 if region_id not in region_morpho_counts:
#                     region_morpho_counts[region_id] = 0
#                 region_morpho_counts[region_id] += 1
#
#                 # 初始化区域数据（如果需要）
#                 if region_id not in region_data:
#                     region_data[region_id] = {
#                         'region_id': region_id,
#                         'region_name': base_region,
#                         # 初始化用于统计计算的值列表
#                         'axonal_length_values': [],
#                         'axonal_branches_values': [],
#                         'axonal_bifurcation_remote_angle_values': [],
#                         'axonal_maximum_branch_order_values': [],
#                         'dendritic_length_values': [],
#                         'dendritic_branches_values': [],
#                         'dendritic_bifurcation_remote_angle_values': [],
#                         'dendritic_maximum_branch_order_values': [],
#                     }
#
#                 # 将特征值添加到列表
#                 neurons_data = dendrite_df[dendrite_df['ID'] == neuron_id]
#                 for csv_col, feature_cols in column_mapping.items():
#                     if csv_col in neurons_data.columns:
#                         values = neurons_data[csv_col].dropna().values
#                         if len(values) > 0:
#                             # 只使用树突相关的特征列
#                             for feature_col in feature_cols:
#                                 if feature_col in dendrite_features_prefix:
#                                     region_data[region_id][f'{feature_col}_values'].extend(values)
#
#     # 筛选出至少有100个完整形态学数据的区域
#     regions_with_sufficient_data = []
#     for region_id, count in region_morpho_counts.items():
#         if count >= 100:
#             regions_with_sufficient_data.append(region_id)
#
#     logger.info(
#         f"找到 {len(regions_with_sufficient_data)} 个至少有100个完整形态学数据的区域，总区域数：{len(region_morpho_counts)}")
#
#     # 如果没有足够的区域，返回空DataFrame
#     if not regions_with_sufficient_data:
#         logger.warning("没有区域包含至少100个完整形态学数据")
#         return pd.DataFrame({'region_id': [], 'region_name': [], 'morpho_count': []})
#
#     # 计算每个有足够形态学数据的区域的统计数据
#     processed_data = []
#     for region_id in regions_with_sufficient_data:
#         region_dict = region_data.get(region_id, {})
#         if not region_dict:
#             continue
#
#         result_dict = {
#             'region_id': region_id,
#             'region_name': region_dict.get('region_name', f"Unknown_{region_id}"),
#             'morpho_count': region_morpho_counts.get(region_id, 0)
#         }
#
#         # 计算每个形态特征的统计数据
#         for feature in MORPH_FEATURES:
#             values = region_dict.get(f'{feature}_values', [])
#             if values:
#                 # 计算平均值
#                 result_dict[feature] = np.mean(values)
#                 # 计算标准差
#                 result_dict[f'{feature}_std'] = np.std(values)
#                 # 计算最小值和最大值
#                 result_dict[f'{feature}_min'] = np.min(values)
#                 result_dict[f'{feature}_max'] = np.max(values)
#             else:
#                 # 没有该特征的数据
#                 result_dict[feature] = np.nan
#                 result_dict[f'{feature}_std'] = np.nan
#                 result_dict[f'{feature}_min'] = np.nan
#                 result_dict[f'{feature}_max'] = np.nan
#
#         processed_data.append(result_dict)
#
#     # 转换为DataFrame
#     morph_df = pd.DataFrame(processed_data)
#
#     # 确保所有必要的列都存在
#     if len(morph_df) > 0:
#         for feature in MORPH_FEATURES:
#             if feature not in morph_df.columns:
#                 logger.warning(f"形态特征列'{feature}'不存在，使用随机值填充")
#                 morph_df[feature] = np.random.normal(0, 1, size=len(morph_df))
#
#     logger.info(f"处理了 {len(morph_df)} 个有足够形态学数据的区域的形态特征")
#     return morph_df


def load_annotation_data(annotation_file: Path) -> Dict[str, Any]:
    """
    加载CCF注释数据

    参数:
        annotation_file: 注释文件路径

    返回:
        包含注释数据的字典
    """
    try:
        volume, header = load_annotation_volume(annotation_file)
        logger.info(f"加载了注释体积，形状: {volume.shape}")

        return {
            'volume': volume,
            'header': header
        }
    except Exception as e:
        logger.error(f"加载注释数据时出错: {e}")
        # 返回空的注释数据
        return {
            'volume': np.zeros((528, 320, 456), dtype=np.int32),
            'header': {'spacing': [25, 25, 25], 'origin': [0, 0, 0]}
        }


def load_annotation_volume(annotation_file: Path) -> Tuple[np.ndarray, Dict]:
    """
    加载CCF注释体积

    参数:
        annotation_file: 注释文件路径

    返回:
        volume: 注释体积
        header: 注释头信息
    """
    # 尝试多种方法加载注释文件
    try:
        # 尝试使用pynrrd
        try:
            import nrrd
            volume, header = nrrd.read(annotation_file)
            logger.info(f"使用pynrrd加载了注释文件")
            return volume, header
        except ImportError:
            logger.debug("找不到pynrrd包")

        # 尝试使用SimpleITK
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(str(annotation_file))
            volume = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            header = {'spacing': spacing, 'origin': origin}
            logger.info(f"使用SimpleITK加载了注释文件")
            return volume, header
        except ImportError:
            logger.debug("找不到SimpleITK包")

        # 如果以上方法都失败，尝试使用nibabel
        try:
            import nibabel as nib
            img = nib.load(str(annotation_file))
            volume = img.get_fdata()
            header = {
                'spacing': img.header.get_zooms(),
                'origin': [0, 0, 0]  # nibabel可能需要手动设置原点
            }
            logger.info(f"使用nibabel加载了注释文件")
            return volume, header
        except ImportError:
            logger.debug("找不到nibabel包")

    except Exception as e:
        logger.error(f"加载注释文件失败: {e}")

    # 如果所有方法都失败，创建一个空的注释体积
    logger.warning(f"无法加载注释文件，创建空的注释体积")
    volume = np.zeros((528, 320, 456), dtype=np.int32)
    header = {'spacing': [25, 25, 25], 'origin': [0, 0, 0]}

    return volume, header


def map_cells_to_regions_fixed(cells_df: pd.DataFrame,
                               annotation_volume: np.ndarray,
                               annotation_header: Dict) -> pd.DataFrame:
    """
    修正后的细胞到区域映射函数 - 直接使用坐标作为索引
    """
    logger.info("将细胞映射到脑区...")

    # 创建副本以避免修改原始数据
    cells = cells_df.copy()

    # 获取坐标和体积形状
    x_coords = cells['x_ccf'].values
    y_coords = cells['y_ccf'].values
    z_coords = cells['z_ccf'].values
    volume_shape = annotation_volume.shape

    # 打印诊断信息
    logger.info(f"注释体积形状: {volume_shape}")
    logger.info(f"坐标范围: X [{x_coords.min():.1f}, {x_coords.max():.1f}], "
                f"Y [{y_coords.min():.1f}, {y_coords.max():.1f}], "
                f"Z [{z_coords.min():.1f}, {z_coords.max():.1f}]")

    # 直接使用坐标作为索引 (不除以25)
    x_indices = x_coords.astype(int)
    y_indices = y_coords.astype(int)
    z_indices = z_coords.astype(int)

    # 检查有效索引
    valid_indices = (
            (0 <= x_indices) & (x_indices < volume_shape[0]) &
            (0 <= y_indices) & (y_indices < volume_shape[1]) &
            (0 <= z_indices) & (z_indices < volume_shape[2])
    )

    valid_count = np.sum(valid_indices)
    logger.info(f"有效索引数量: {valid_count}/{len(cells)} ({valid_count / len(cells) * 100:.1f}%)")

    # 准备存储区域ID
    region_ids = np.zeros(len(cells), dtype=int)

    # 使用批处理获取区域ID
    batch_size = 100000
    valid_indices_pos = np.where(valid_indices)[0]

    for i in range(0, len(valid_indices_pos), batch_size):
        batch_indices = valid_indices_pos[i:i + batch_size]
        batch_x = x_indices[batch_indices]
        batch_y = y_indices[batch_indices]
        batch_z = z_indices[batch_indices]

        for j, (x, y, z, orig_idx) in enumerate(zip(batch_x, batch_y, batch_z, batch_indices)):
            try:
                region_ids[orig_idx] = annotation_volume[x, y, z]
            except IndexError:
                # 这不应该发生，因为我们已经检查了索引范围
                pass

        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(valid_indices_pos):
            logger.info(f"已处理 {min(i + batch_size, len(valid_indices_pos))}/{len(valid_indices_pos)} 个有效索引")

    # 将区域ID添加到DataFrame
    cells['region_id'] = region_ids

    # 统计结果
    nonzero_regions = (region_ids > 0).sum()
    unique_regions = np.unique(region_ids[region_ids > 0]).size

    logger.info(f"成功映射了{nonzero_regions}/{len(cells)}个细胞到{unique_regions}个不同区域")

    return cells


def update_adata_region_ids(adata: ad.AnnData, cells_df: pd.DataFrame) -> None:
    """
    更新AnnData对象中的region_id
    """
    if 'cell_label' in cells_df.columns:
        # 使用cell_label匹配
        cell_to_region = cells_df.set_index('cell_label')['region_id'].to_dict()
        region_ids = [cell_to_region.get(cell, 0) for cell in adata.obs.index]
    else:
        # 假设顺序相同
        if len(cells_df) == len(adata.obs):
            region_ids = cells_df['region_id'].values
        else:
            logger.error("细胞数量不匹配，无法更新region_id")
            return

    adata.obs['region_id'] = region_ids
    logger.info(f"更新了{len(region_ids)}个细胞的region_id")


def get_merfish_compatible_gene_modules(adata: ad.AnnData) -> Dict[str, List[str]]:
    """
    创建与MERFISH数据兼容的基因模块
    """
    logger.info("构建MERFISH兼容的基因模块...")

    # 获取数据中的所有基因
    all_genes = set(adata.var_names)

    # 定义基因模块（使用多种可能的基因名称格式）
    raw_modules = {
        "Glutamatergic": ["Slc17a7", "Slc17a6", "Slc17a8", "Satb2", "Cux2", "Rorb", "Fezf2"],
        "GABAergic": ["Gad1", "Gad2", "Slc32a1", "Pvalb", "Sst", "Vip", "Lamp5"],
        "Astrocyte": ["Aldh1l1", "Gfap", "Aqp4", "S100b", "Slc1a3", "Slc1a2"],
        "Oligodendrocyte": ["Olig1", "Olig2", "Mbp", "Plp1", "Mog", "Sox10"],
        "Microglia": ["Cx3cr1", "P2ry12", "Tmem119", "Csf1r", "Hexb"],
        "Endothelial": ["Pecam1", "Cldn5", "Flt1", "Slco1c1", "Ly6c1"]
    }

    # 查找匹配的基因
    gene_modules = {}
    for module_name, gene_list in raw_modules.items():
        found_genes = []
        for gene in gene_list:
            # 尝试不同的格式
            for variant in [gene, gene.upper(), gene.lower(), gene.capitalize()]:
                if variant in all_genes:
                    found_genes.append(variant)
                    break

        if len(found_genes) >= 3:
            gene_modules[module_name] = found_genes
            logger.info(f"模块 {module_name}: 找到 {len(found_genes)} 个基因")

    return gene_modules


def compute_gene_module_scores_fixed(adata: ad.AnnData,
                                     gene_modules: Dict[str, List[str]]) -> pd.DataFrame:
    """
    修正后的基因模块分数计算
    """
    logger.info("计算基因模块分数...")

    module_scores = {}

    for module_name, genes in gene_modules.items():
        # 获取表达数据
        expr_data = adata[:, genes].X

        # 处理稀疏矩阵
        if hasattr(expr_data, 'toarray'):
            expr_data = expr_data.toarray()

        # 计算平均表达
        score = np.mean(expr_data, axis=1)
        module_scores[module_name] = score

    # 创建DataFrame
    scores_df = pd.DataFrame(module_scores, index=adata.obs.index)

    # 按区域聚合
    if 'region_id' in adata.obs.columns:
        scores_df['region_id'] = adata.obs['region_id']
        # 只聚合有效的区域
        valid_scores = scores_df[scores_df['region_id'] > 0]
        region_scores = valid_scores.groupby('region_id').mean()
        return region_scores

    return scores_df


def integrate_merfish_to_regions(region_data: pd.DataFrame,
                                 subclass_proportions: Optional[pd.DataFrame],
                                 module_scores: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    将MERFISH衍生的特征整合到区域数据中
    """
    logger.info("整合MERFISH特征到区域数据...")

    # 确保region_id是字符串类型以便匹配
    if 'region_id' in region_data.columns:
        region_data = region_data.copy()
        region_data['region_id'] = region_data['region_id'].astype(str)

    # 整合细胞类型比例
    if subclass_proportions is not None and not subclass_proportions.empty:
        subclass_proportions.index = subclass_proportions.index.astype(str)

        for col in subclass_proportions.columns:
            col_name = f'subclass_{col}'
            region_data[col_name] = region_data['region_id'].map(
                subclass_proportions[col].to_dict()
            )

    # 整合基因模块分数
    if module_scores is not None and not module_scores.empty:
        module_scores.index = module_scores.index.astype(str)

        for col in module_scores.columns:
            region_data[col] = region_data['region_id'].map(
                module_scores[col].to_dict()
            )

    # 填充缺失值
    numeric_cols = region_data.select_dtypes(include=[np.number]).columns
    region_data[numeric_cols] = region_data[numeric_cols].fillna(0)

    return region_data


def compute_cell_type_proportions(cells_df: pd.DataFrame,
                                 cell_type_col: str = 'subclass') -> pd.DataFrame:
    """
    计算每个脑区中各细胞类型的比例

    参数:
        cells_df: 包含细胞信息的DataFrame
        cell_type_col: 细胞类型列名

    返回:
        区域×细胞类型的比例矩阵
    """
    logger.info(f"计算区域中的{cell_type_col}比例...")

    # 确保必需的列存在
    required_cols = ['region_id', cell_type_col]
    if not all(col in cells_df.columns for col in required_cols):
        logger.error(f"cells_df必须包含以下列: {required_cols}")
        return pd.DataFrame()

    # 计算每个区域中各细胞类型的计数
    type_counts = cells_df.groupby(['region_id', cell_type_col]).size().reset_index(name='count')

    # 计算每个区域的总细胞数
    region_totals = cells_df.groupby('region_id').size().reset_index(name='total')

    # 合并并计算比例
    proportions = pd.merge(type_counts, region_totals, on='region_id')
    proportions['proportion'] = proportions['count'] / proportions['total']

    # 透视为区域×细胞类型的矩阵
    prop_matrix = proportions.pivot(
        index='region_id',
        columns=cell_type_col,
        values='proportion'
    ).fillna(0)

    logger.info(f"生成了{prop_matrix.shape[0]}个区域和{prop_matrix.shape[1]}个细胞类型的比例矩阵")

    return prop_matrix


def compute_gene_module_scores(adata: ad.AnnData,
                              gene_modules: Dict[str, List[str]],
                              min_genes: int = 5) -> pd.DataFrame:
    """
    计算基因模块的表达分数

    参数:
        adata: 包含基因表达的AnnData对象
        gene_modules: 模块名称到基因列表的字典
        min_genes: 计算模块分数所需的最小基因数量

    返回:
        细胞×模块的分数矩阵
    """
    logger.info("计算基因模块分数...")

    # 获取所有基因
    all_genes = set(adata.var_names)

    # 尝试不同的大小写格式匹配基因
    all_genes_upper = {g.upper() for g in all_genes}
    all_genes_lower = {g.lower() for g in all_genes}

    # 创建基因名大小写映射
    gene_case_map = {}
    for g in all_genes:
        gene_case_map[g.upper()] = g
        gene_case_map[g.lower()] = g

    # 为每个模块计算分数
    module_scores = {}
    for module_name, genes in gene_modules.items():
        # 找出实际存在于数据中的基因，处理大小写
        module_genes = []
        for gene in genes:
            if gene in all_genes:
                module_genes.append(gene)
            elif gene.upper() in all_genes_upper:
                module_genes.append(gene_case_map[gene.upper()])
            elif gene.lower() in all_genes_lower:
                module_genes.append(gene_case_map[gene.lower()])

        if len(module_genes) >= min_genes:
            # 计算模块分数 (平均表达)
            score = adata[:, module_genes].X.mean(axis=1)
            module_scores[module_name] = score
            logger.debug(f"模块 {module_name}: 使用 {len(module_genes)}/{len(genes)} 个基因")
        else:
            logger.warning(f"模块 {module_name} 只有 {len(module_genes)} 个基因匹配，低于阈值 {min_genes}")

    # 转换为DataFrame
    scores_df = pd.DataFrame(module_scores, index=adata.obs.index)

    # 按区域聚合
    if 'region_id' in adata.obs.columns:
        region_scores = scores_df.join(adata.obs['region_id']).groupby('region_id').mean()
        logger.info(f"计算了 {len(region_scores)} 个区域的 {len(module_scores)} 个模块分数")
        return region_scores
    else:
        logger.warning("AnnData对象中没有region_id列，返回细胞级别的模块分数")
        return scores_df


def prepare_analysis_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        完整修复版本的数据准备函数
        """
    logger.info("准备数据用于分析（完整修复版）...")

    processed_data = data.copy()

    # 获取数据
    merfish_cells = processed_data.get('merfish_cells', pd.DataFrame())
    merfish_expression = processed_data.get('merfish_expression', None)

    # 1. 加载基因ID映射并转换基因名称
    if merfish_expression is not None:
        gene_mapping = load_gene_panel_mapping("../data/gene_panel_1122.csv")

        if gene_mapping:
            merfish_expression = convert_ensembl_to_symbols(merfish_expression, gene_mapping)
            processed_data['merfish_expression'] = merfish_expression

        # 获取转换后的基因符号
        gene_symbols = set(merfish_expression.var_names)
    else:
        gene_symbols = set()

    # 2. 映射细胞到区域（使用修正的函数）
    if 'annotation' in processed_data and not merfish_cells.empty:
        annotation = processed_data['annotation']

        if all(col in merfish_cells.columns for col in ['x_ccf', 'y_ccf', 'z_ccf']):
            merfish_cells = map_cells_to_regions_fixed(
                merfish_cells,
                annotation['volume'],
                annotation['header']
            )
            processed_data['merfish_cells'] = merfish_cells

            # 更新表达数据中的region_id
            if merfish_expression is not None:
                update_adata_region_ids_safe(merfish_expression, merfish_cells)

    # 3. 创建基因模块
    if gene_symbols:
        gene_modules = get_merfish_gene_modules_with_symbols(gene_symbols)
        processed_data['gene_modules'] = gene_modules

        # 4. 计算基因模块分数
        if gene_modules and merfish_expression is not None:
            try:
                module_scores = compute_gene_module_scores_vectorized(
                    merfish_expression,
                    gene_modules
                )
                if module_scores is not None and not module_scores.empty:
                    processed_data['module_scores'] = module_scores
            except Exception as e:
                logger.error(f"计算模块分数失败: {e}")

    # 5. 计算细胞类型比例
    if 'region_id' in merfish_cells.columns and 'subclass' in merfish_cells.columns:
        valid_cells = merfish_cells[merfish_cells['region_id'] > 0]

        if len(valid_cells) > 0:
            try:
                subclass_proportions = compute_cell_type_proportions_safe(
                    valid_cells,
                    'subclass'
                )
                if not subclass_proportions.empty:
                    processed_data['subclass_proportions'] = subclass_proportions
            except Exception as e:
                logger.error(f"计算细胞类型比例失败: {e}")

    logger.info("数据准备完成")
    return processed_data


def update_adata_region_ids_safe(adata: ad.AnnData, cells_df: pd.DataFrame) -> None:
    """
    安全地更新AnnData中的region_id
    """
    try:
        # 准备细胞标签到区域ID的映射
        region_map = {}

        # 尝试找到匹配的细胞
        if 'cell_label' in cells_df.columns:
            # 创建细胞标签到区域ID的映射
            region_map = cells_df.set_index('cell_label')['region_id'].to_dict()
            logger.info(f"创建了{len(region_map)}个细胞标签到区域ID的映射")

            # 检查AnnData观察名称是否匹配细胞标签
            if any(cell in region_map for cell in adata.obs_names[:10]):
                # 使用索引名称直接映射
                adata.obs['region_id'] = [region_map.get(cell, 0) for cell in adata.obs_names]
                mapped_count = (adata.obs['region_id'] > 0).sum()
                logger.info(f"使用观察名称映射了{mapped_count}个细胞的region_id")
                return

        # 如果AnnData中有cell_label列，尝试使用它
        if 'cell_label' in adata.obs.columns:
            adata.obs['region_id'] = adata.obs['cell_label'].map(
                lambda x: region_map.get(x, 0)
            ).fillna(0).astype(int)

            mapped_count = (adata.obs['region_id'] > 0).sum()
            logger.info(f"使用obs.cell_label列映射了{mapped_count}个细胞的region_id")
            return

        # 最后的回退方法：如果两个数据框长度相同，假设它们以相同顺序排列
        if len(adata.obs) == len(cells_df):
            adata.obs['region_id'] = cells_df['region_id'].values
            mapped_count = (adata.obs['region_id'] > 0).sum()
            logger.info(f"使用索引位置映射了{mapped_count}个细胞的region_id")
            return

        logger.warning(f"无法找到匹配细胞的方法，没有更新region_id")

    except Exception as e:
        logger.error(f"更新region_id失败: {e}")
        # 确保region_id列存在，即使更新失败
        if 'region_id' not in adata.obs.columns:
            adata.obs['region_id'] = 0


def compute_gene_module_scores_vectorized(adata: ad.AnnData,
                                          gene_modules: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
    """
    向量化的基因模块分数计算
    """
    try:
        logger.info("计算基因模块分数...")

        module_scores = {}

        for module_name, genes in gene_modules.items():
            # 获取表达矩阵
            expr_data = adata[:, genes].X

            # 处理稀疏矩阵
            if hasattr(expr_data, 'toarray'):
                expr_data = expr_data.toarray()

            # 计算平均表达
            scores = np.mean(expr_data, axis=1)
            module_scores[module_name] = scores

            logger.debug(f"模块 {module_name}: 计算了 {len(scores)} 个细胞的分数")

        # 创建DataFrame
        scores_df = pd.DataFrame(module_scores, index=adata.obs.index)

        # 按区域聚合
        if 'region_id' in adata.obs.columns:
            scores_df['region_id'] = adata.obs['region_id'].values

            # 只聚合有效区域
            valid_scores = scores_df[scores_df['region_id'] > 0]

            if len(valid_scores) > 0:
                region_scores = valid_scores.groupby('region_id').mean()
                logger.info(f"计算了 {len(region_scores)} 个区域的模块分数")
                return region_scores

        return scores_df

    except Exception as e:
        logger.error(f"计算模块分数失败: {e}")
        return None


def compute_cell_type_proportions_safe(cells_df: pd.DataFrame,
                                       cell_type_col: str = 'subclass') -> pd.DataFrame:
    """
    安全地计算细胞类型比例
    """
    try:
        # 计算比例
        proportions = cells_df.groupby(['region_id', cell_type_col]).size()
        totals = cells_df.groupby('region_id').size()

        # 转换为比例矩阵
        prop_matrix = proportions.unstack(fill_value=0)
        prop_matrix = prop_matrix.div(totals, axis=0)

        logger.info(f"计算了 {prop_matrix.shape[0]} 个区域和 {prop_matrix.shape[1]} 个细胞类型的比例")

        return prop_matrix

    except Exception as e:
        logger.error(f"计算细胞类型比例失败: {e}")
        return pd.DataFrame()


# 默认基因模块
DEFAULT_GENE_MODULES = {
    "Excitatory_Neurons": [
        "SLC17A7", "SATB2", "CUX2", "RORB", "FEZF2", "FOXP2", "NTSR1",
        "GRIN2B", "CAMK2A", "NEUROD2", "NEUROD6", "TBR1", "EMX1"
    ],
    "Inhibitory_Neurons": [
        "GAD1", "GAD2", "PVALB", "SST", "VIP", "NPY", "CCK", "CALB1",
        "CALB2", "LHX6", "DLX1", "DLX2", "NKX2-1"
    ],
    "Astrocytes": [
        "GFAP", "ALDH1L1", "AQP4", "SLC1A3", "SLC1A2", "FGFR3",
        "S100B", "GJA1", "SOX9", "NFIA", "NFIB"
    ],
    "Oligodendrocytes": [
        "OLIG1", "OLIG2", "MBP", "PLP1", "MOG", "CLDN11", "MAG",
        "SOX10", "CNP", "TRF", "PDGFRA", "BCAS1"
    ],
    "Microglia": [
        "CX3CR1", "P2RY12", "CSF1R", "AIF1", "TMEM119", "ITGAM",
        "HEXB", "TREM2", "CD68", "PTPRC", "IRF8", "TYROBP"
    ],
    "Endothelial": [
                "PECAM1", "CLDN5", "FLT1", "CDH5", "VWF", "TEK", "KDR",
        "SLC2A1", "PLVAP", "FN1"
    ]
}


if __name__ == "__main__":
    # 测试数据加载
    from setup import DATA_DIR

    try:
        data = load_data(DATA_DIR)
        processed_data = prepare_analysis_data(data)
        print(f"成功加载和处理数据")

        region_data = processed_data['region_data']
        print(f"区域数据形状: {region_data.shape}")
        print(f"区域数据列: {region_data.columns.tolist()[:10]}...")

        if 'gene_modules' in processed_data:
            modules = processed_data['gene_modules']
            print(f"加载了 {len(modules)} 个基因模块")
            print(f"模块示例: {list(modules.keys())[:5]}")

    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请检查数据路径和文件格式")