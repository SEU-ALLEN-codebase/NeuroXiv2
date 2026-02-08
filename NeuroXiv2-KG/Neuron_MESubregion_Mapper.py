import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger
import nrrd


class NeuronMESubregionMapperV2:
    """正确版本：基于NRRD体素值直接对应ME子区域的理解"""

    def __init__(self,
                 nrrd_file: Path,
                 pkl_file: Path,
                 soma_file: Path,
                 info_file: Path,
                 json_tree_file: Optional[Path] = None):
        """
        初始化映射器

        参数:
            nrrd_file: ME注释的nrrd文件（体素值=ME子区域标识）
            pkl_file: NRRD体素值到父区域ID的映射
            soma_file: 包含神经元soma坐标的CSV文件
            info_file: 神经元信息文件（包含celltype）
            json_tree_file: 区域层级树JSON文件（提供ME子区域名称）
        """
        self.nrrd_file = nrrd_file
        self.pkl_file = pkl_file
        self.soma_file = soma_file
        self.info_file = info_file
        self.json_tree_file = json_tree_file

        # 数据容器
        self.annotation_volume = None
        self.annotation_header = None
        self.voxel_to_parent_map = None      # NRRD体素值 -> 父区域ID
        self.voxel_to_me_info_map = {}       # NRRD体素值 -> ME子区域信息
        self.soma_df = None
        self.info_df = None

        # 统计信息
        self.stats = {
            'total_neurons': 0,
            'mapped_to_me_subregion': 0,
            'mapped_to_non_me_region': 0,
            'unmapped': 0
        }

    def load_nrrd_annotation(self) -> bool:
        """加载NRRD注释文件"""
        logger.info(f"加载NRRD注释文件: {self.nrrd_file}")

        try:
            self.annotation_volume, self.annotation_header = nrrd.read(str(self.nrrd_file))
            logger.info(f"NRRD体积形状: {self.annotation_volume.shape}")
            logger.info(f"NRRD数据类型: {self.annotation_volume.dtype}")

            # 检查注释值的范围
            unique_values = np.unique(self.annotation_volume)
            logger.info(f"NRRD中包含 {len(unique_values)} 个唯一体素值")
            logger.info(f"体素值范围: {unique_values.min()} - {unique_values.max()}")

            return True

        except Exception as e:
            logger.error(f"加载NRRD文件失败: {e}")
            return False

    def load_pkl_mapping(self) -> bool:
        """
        加载PKL映射文件

        PKL格式: {nrrd_voxel_value: parent_region_id}
        作用: 告诉我们每个NRRD体素值(可能是ME子区域)属于哪个父区域
        """
        logger.info(f"加载PKL映射文件: {self.pkl_file}")

        try:
            with open(self.pkl_file, 'rb') as f:
                self.voxel_to_parent_map = pickle.load(f)

            if not isinstance(self.voxel_to_parent_map, dict):
                logger.error(f"PKL文件不是字典格式")
                return False

            logger.info(f"加载了 {len(self.voxel_to_parent_map)} 个体素值->父区域映射")

            # 显示样例
            sample_items = list(self.voxel_to_parent_map.items())[:5]
            logger.info(f"映射样例 (NRRD体素值 -> 父区域ID):")
            for voxel_val, parent_id in sample_items:
                logger.info(f"  {voxel_val} -> {parent_id}")

            return True

        except Exception as e:
            logger.error(f"加载PKL文件失败: {e}")
            return False

    def load_me_regions_from_json(self) -> bool:
        """
        从JSON文件加载ME子区域信息

        关键：提取JSON中'943_953'格式的ID，其中953就是NRRD体素值
        """
        if not self.json_tree_file or not self.json_tree_file.exists():
            logger.warning(f"JSON文件不存在或未提供: {self.json_tree_file}")
            return False

        logger.info(f"加载ME子区域信息: {self.json_tree_file}")

        try:
            import json
            with open(self.json_tree_file, 'r') as f:
                tree_data = json.load(f)

            # 递归提取所有ME子区域
            self._extract_me_regions_from_tree(tree_data)

            logger.info(f"从JSON提取了 {len(self.voxel_to_me_info_map)} 个ME子区域映射")

            # 显示一些ME子区域样例
            if self.voxel_to_me_info_map:
                logger.info("ME子区域样例（前10个）:")
                for i, (voxel, info) in enumerate(list(self.voxel_to_me_info_map.items())[:10]):
                    logger.info(f"  体素{voxel:4d}: {info['acronym']:<20} (父区域{info['parent_id']})")

            # 验证PKL和JSON的一致性
            self._verify_pkl_json_consistency()

            return True

        except Exception as e:
            logger.error(f"加载JSON文件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _extract_me_regions_from_tree(self, nodes, parent_info=None):
        """
        递归提取ME子区域信息

        关键：从'943_953'格式中提取953作为NRRD体素值
        """
        if isinstance(nodes, dict):
            nodes = [nodes]

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_id = node.get('id')
            acronym = node.get('acronym', '')
            name = node.get('name', '')

            # 只处理ME子区域（带-ME后缀）
            if acronym and '-ME' in acronym:
                if isinstance(node_id, str) and '_' in node_id:
                    # 格式: '943_953'
                    parts = node_id.split('_')
                    try:
                        parent_id = int(parts[0])  # 943
                        voxel_value = int(parts[1])  # 953 - 这就是NRRD体素值！

                        self.voxel_to_me_info_map[voxel_value] = {
                            'voxel_value': voxel_value,
                            'parent_id': parent_id,
                            'acronym': acronym,
                            'name': name,
                            'json_id': node_id
                        }
                    except (ValueError, IndexError) as e:
                        logger.warning(f"无法解析ME子区域ID: {node_id}")

            # 递归处理子节点
            children = node.get('children', [])
            if children:
                self._extract_me_regions_from_tree(children, node)

    def _verify_pkl_json_consistency(self):
        """验证PKL和JSON的一致性"""
        logger.info("验证PKL和JSON的一致性...")

        me_voxels = set(self.voxel_to_me_info_map.keys())
        pkl_voxels = set(self.voxel_to_parent_map.keys())

        # ME体素值应该都在PKL中
        me_in_pkl = me_voxels & pkl_voxels
        logger.info(f"ME子区域体素值在PKL中: {len(me_in_pkl)}/{len(me_voxels)} ({len(me_in_pkl)/len(me_voxels)*100:.1f}%)")

        if len(me_in_pkl) < len(me_voxels):
            missing = me_voxels - pkl_voxels
            logger.warning(f"有 {len(missing)} 个ME子区域体素值不在PKL中")
            logger.warning(f"缺失样例: {list(missing)[:5]}")

        # 验证父区域ID是否一致
        mismatches = 0
        for voxel in me_in_pkl:
            expected_parent = self.voxel_to_me_info_map[voxel]['parent_id']
            actual_parent = self.voxel_to_parent_map[voxel]
            if expected_parent != actual_parent:
                mismatches += 1
                if mismatches <= 3:  # 只显示前3个不匹配的
                    logger.warning(f"父区域ID不匹配: 体素{voxel}, "
                                 f"JSON中={expected_parent}, PKL中={actual_parent}")

        if mismatches == 0:
            logger.info("✓ PKL和JSON的父区域ID完全一致")
        else:
            logger.warning(f"⚠ 有 {mismatches} 个ME子区域的父区域ID不匹配")

    def load_soma_coordinates(self) -> bool:
        """加载soma坐标数据"""
        logger.info(f"加载soma坐标文件: {self.soma_file}")

        try:
            self.soma_df = pd.read_csv(self.soma_file)
            logger.info(f"加载了 {len(self.soma_df)} 条soma记录")

            # 检查必要的列
            required_cols = ['ID', 'x', 'y', 'z']
            missing_cols = [col for col in required_cols if col not in self.soma_df.columns]

            if missing_cols:
                logger.error(f"Soma文件缺少必要的列: {missing_cols}")
                return False

            logger.info(f"X坐标范围: {self.soma_df['x'].min():.1f} - {self.soma_df['x'].max():.1f}")
            logger.info(f"Y坐标范围: {self.soma_df['y'].min():.1f} - {self.soma_df['y'].max():.1f}")
            logger.info(f"Z坐标范围: {self.soma_df['z'].min():.1f} - {self.soma_df['z'].max():.1f}")

            return True

        except Exception as e:
            logger.error(f"加载soma文件失败: {e}")
            return False

    def load_info_file(self) -> bool:
        """加载info.csv文件"""
        logger.info(f"加载info文件: {self.info_file}")

        try:
            self.info_df = pd.read_csv(self.info_file)
            logger.info(f"加载了 {len(self.info_df)} 条info记录")

            if 'celltype' in self.info_df.columns:
                non_null_celltype = self.info_df['celltype'].notna().sum()
                logger.info(f"包含 {non_null_celltype}/{len(self.info_df)} 个有celltype的记录")

            return True

        except Exception as e:
            logger.error(f"加载info文件失败: {e}")
            return False

    def coordinate_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """将物理坐标转换为体素索引（假设25μm分辨率）"""
        resolution = 25.0
        voxel_x = int(x / resolution)
        voxel_y = int(y / resolution)
        voxel_z = int(z / resolution)
        return voxel_x, voxel_y, voxel_z

    def get_me_subregion_at_coordinate(self, x: float, y: float, z: float) -> Optional[Dict]:
        """
        获取指定坐标处的ME子区域信息

        返回:
            如果是ME子区域: {
                'voxel_value': int,
                'parent_id': int,
                'acronym': str,
                'name': str,
                'is_me_subregion': True
            }
            如果是其他区域: {
                'voxel_value': int,
                'parent_id': int (from PKL),
                'acronym': None,
                'is_me_subregion': False
            }
            如果无效: None
        """
        # 转换为体素索引
        voxel_x, voxel_y, voxel_z = self.coordinate_to_voxel(x, y, z)

        # 检查边界
        if not (0 <= voxel_x < self.annotation_volume.shape[0] and
                0 <= voxel_y < self.annotation_volume.shape[1] and
                0 <= voxel_z < self.annotation_volume.shape[2]):
            return None

        # 获取NRRD体素值
        voxel_value = int(self.annotation_volume[voxel_x, voxel_y, voxel_z])

        if voxel_value == 0:
            return None

        # 检查是否是ME子区域
        if voxel_value in self.voxel_to_me_info_map:
            # 是ME子区域！
            me_info = self.voxel_to_me_info_map[voxel_value].copy()
            me_info['is_me_subregion'] = True
            return me_info
        else:
            # 不是ME子区域，但可能是其他区域
            parent_id = self.voxel_to_parent_map.get(voxel_value)
            return {
                'voxel_value': voxel_value,
                'parent_id': parent_id,
                'acronym': None,
                'name': None,
                'is_me_subregion': False
            }

    def map_neurons_to_regions(self) -> pd.DataFrame:
        """
        将所有神经元映射到区域（包括ME子区域）

        返回DataFrame包含:
        - ID: 神经元ID
        - subregion: 从celltype获取的Subregion
        - me_subregion_voxel: ME子区域的NRRD体素值
        - me_subregion_acronym: ME子区域的缩写名称
        - me_subregion_name: ME子区域的完整名称
        - parent_region_id: 父区域ID
        - is_me_subregion: 是否为ME子区域
        """
        logger.info("开始映射神经元到区域...")

        # 合并info和soma数据
        merged_df = pd.merge(
            self.info_df[['ID', 'celltype']],
            self.soma_df[['ID', 'x', 'y', 'z']],
            on='ID',
            how='inner'
        )

        logger.info(f"合并后有 {len(merged_df)} 个神经元有完整信息")

        results = []
        self.stats['total_neurons'] = len(merged_df)

        for idx, row in merged_df.iterrows():
            neuron_id = row['ID']
            celltype = row.get('celltype', '')
            x, y, z = row['x'], row['y'], row['z']

            # 查找ME子区域
            region_info = self.get_me_subregion_at_coordinate(x, y, z)

            result = {
                'ID': neuron_id,
                'subregion': celltype if pd.notna(celltype) else None,
                'soma_x': x,
                'soma_y': y,
                'soma_z': z,
            }

            if region_info:
                is_me = region_info.get('is_me_subregion', False)

                if is_me:
                    # 映射到ME子区域
                    result['me_subregion_voxel'] = region_info['voxel_value']
                    result['me_subregion_acronym'] = region_info['acronym']
                    result['me_subregion_name'] = region_info['name']
                    result['parent_region_id'] = region_info['parent_id']
                    result['is_me_subregion'] = True
                    self.stats['mapped_to_me_subregion'] += 1
                else:
                    # 映射到其他区域（非ME）
                    result['me_subregion_voxel'] = None
                    result['me_subregion_acronym'] = None
                    result['me_subregion_name'] = None
                    result['parent_region_id'] = region_info.get('parent_id')
                    result['is_me_subregion'] = False
                    self.stats['mapped_to_non_me_region'] += 1
            else:
                # 未映射
                result['me_subregion_voxel'] = None
                result['me_subregion_acronym'] = None
                result['me_subregion_name'] = None
                result['parent_region_id'] = None
                result['is_me_subregion'] = False
                self.stats['unmapped'] += 1

            results.append(result)

            # 定期报告进度
            if (idx + 1) % 1000 == 0:
                logger.info(f"已处理 {idx + 1}/{len(merged_df)} 个神经元")

        # 打印统计信息
        logger.info("="*60)
        logger.info("映射统计:")
        logger.info(f"  总神经元数: {self.stats['total_neurons']}")
        logger.info(f"  映射到ME_Subregion: {self.stats['mapped_to_me_subregion']} "
                   f"({self.stats['mapped_to_me_subregion']/self.stats['total_neurons']*100:.1f}%)")
        logger.info(f"  映射到非ME区域: {self.stats['mapped_to_non_me_region']} "
                   f"({self.stats['mapped_to_non_me_region']/self.stats['total_neurons']*100:.1f}%)")
        logger.info(f"  未映射: {self.stats['unmapped']} "
                   f"({self.stats['unmapped']/self.stats['total_neurons']*100:.1f}%)")
        logger.info("="*60)

        # 显示映射样例
        result_df = pd.DataFrame(results)
        me_mapped = result_df[result_df['is_me_subregion'] == True]

        if not me_mapped.empty:
            logger.info(f"\nME_Subregion映射样例（前10个）:")
            for idx, row in me_mapped.head(10).iterrows():
                logger.info(f"  神经元 {row['ID']}: "
                          f"{row['me_subregion_acronym']} "
                          f"(体素{row['me_subregion_voxel']}, 父区域{row['parent_region_id']})")

            # 统计每个ME子区域映射的神经元数
            me_counts = me_mapped['me_subregion_acronym'].value_counts()
            logger.info(f"\nME子区域分布（前10个）:")
            for acronym, count in me_counts.head(10).items():
                logger.info(f"  {acronym}: {count} 个神经元")
        else:
            logger.warning("⚠ 没有神经元被映射到ME_Subregion!")

        return result_df

    def update_info_file(self, mapping_df: pd.DataFrame, output_file: Optional[Path] = None):
        """更新info.csv文件，添加ME_Subregion信息"""
        logger.info("更新info文件...")

        columns_to_add = [
            'ID',
            'me_subregion_voxel',
            'me_subregion_acronym',
            'me_subregion_name',
            'parent_region_id',
            'is_me_subregion'
        ]

        updated_info = self.info_df.merge(
            mapping_df[columns_to_add],
            on='ID',
            how='left'
        )

        if output_file is None:
            output_file = self.info_file.parent / "info_with_me_subregion.csv"

        updated_info.to_csv(output_file, index=False)
        logger.info(f"已保存更新后的info文件到: {output_file}")

        me_count = updated_info['is_me_subregion'].sum()
        logger.info(f"Info文件中 {me_count}/{len(updated_info)} 个神经元有ME_Subregion信息")

        return output_file

    def run_full_pipeline(self, output_info_file: Optional[Path] = None) -> bool:
        """运行完整的映射流程"""
        logger.info("="*60)
        logger.info("开始神经元ME_Subregion映射流程（正确版本）")
        logger.info("="*60)

        # 1. 加载NRRD注释
        if not self.load_nrrd_annotation():
            return False

        # 2. 加载PKL映射
        if not self.load_pkl_mapping():
            return False

        # 3. 加载ME子区域信息
        if not self.load_me_regions_from_json():
            logger.error("无法加载JSON文件，无法继续")
            return False

        # 4. 加载soma坐标
        if not self.load_soma_coordinates():
            return False

        # 5. 加载info文件
        if not self.load_info_file():
            return False

        # 6. 执行映射
        mapping_df = self.map_neurons_to_regions()

        # 7. 更新info文件
        output_path = self.update_info_file(mapping_df, output_info_file)

        logger.info("="*60)
        logger.info("映射流程完成")
        logger.info(f"输出文件: {output_path}")
        logger.info("="*60)

        return True


def main():
    """主函数"""
    import sys

    # 设置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("neuron_me_mapping_correct.log", rotation="100 MB", level="DEBUG")

    # 文件路径 - 根据实际情况修改
    data_dir = Path("../data")

    nrrd_file = data_dir / "parc_r671_full.nrrd"
    pkl_file = data_dir / "parc_r671_full.nrrd.pkl"
    soma_file = data_dir / "soma.csv"
    info_file = data_dir / "info.csv"
    json_file = data_dir / "surf_tree_ccf-me.json"
    output_file = data_dir / "info_with_me_subregion.csv"

    # 创建映射器
    mapper = NeuronMESubregionMapper(
        nrrd_file=nrrd_file,
        pkl_file=pkl_file,
        soma_file=soma_file,
        info_file=info_file,
        json_tree_file=json_file
    )

    # 运行映射流程
    success = mapper.run_full_pipeline(output_info_file=output_file)

    if success:
        logger.info("✓ 映射成功完成!")
    else:
        logger.error("✗ 映射失败!")
        return 1

    return 0
