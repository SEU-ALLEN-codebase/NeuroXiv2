import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from loguru import logger

BATCH_SIZE = 1000

class SubregionLoader:
    """Subregion和ME_Subregion数据加载器"""

    def __init__(self, ccf_me_json_path: Path):
        """
        初始化Subregion加载器

        参数:
            ccf_me_json_path: CCF-ME atlas JSON文件路径
        """
        self.ccf_me_json_path = ccf_me_json_path
        self.subregions = []  # Subregion节点列表
        self.me_subregions = []  # ME_Subregion节点列表
        self.me_to_subregion = {}  # ME_Subregion -> Subregion映射
        self.subregion_to_region = {}  # Subregion -> Region映射

    def load_subregion_data(self) -> bool:
        """加载Subregion和ME_Subregion数据"""
        logger.info(f"从 {self.ccf_me_json_path} 加载Subregion数据...")

        try:
            with open(self.ccf_me_json_path, 'r') as f:
                data = json.load(f)

            # 遍历树结构提取数据
            for root in data:
                self._traverse_tree(root)

            logger.info(f"成功加载 {len(self.subregions)} 个Subregion节点")
            logger.info(f"成功加载 {len(self.me_subregions)} 个ME_Subregion节点")

            return True

        except Exception as e:
            logger.error(f"加载Subregion数据失败: {e}")
            return False

    def _traverse_tree(self, node, parent_acronym=None, grandparent_acronym=None):
        """
        递归遍历树结构，提取Subregion和ME_Subregion信息

        参数:
            node: 当前节点
            parent_acronym: 父节点缩写
            grandparent_acronym: 祖父节点缩写
        """
        acronym = node.get('acronym', '')
        node_id = node.get('id')
        name = node.get('name', '')
        rgb_triplet = node.get('rgb_triplet', [])

        # 检查是否是ME节点
        if '-ME' in acronym:
            self.me_subregions.append({
                'me_subregion_id': str(node_id),  # 使用字符串ID，因为有些是"943_953"格式
                'acronym': acronym,
                'name': name,
                'parent_subregion': parent_acronym,  # Subregion（layer节点）
                'rgb_triplet': rgb_triplet
            })
            self.me_to_subregion[acronym] = parent_acronym

        # 检查是否是layer节点（Subregion）
        elif parent_acronym and not '-ME' in parent_acronym:
            is_layer = False

            # 判断是否是layer节点：节点名中包含"layer"
            if 'layer' in name.lower():
                is_layer = True

            # 或者检查缩写是否符合layer模式
            if not is_layer:
                for pattern in ['1', '2/3', '4', '5', '6a', '6b']:
                    if acronym.endswith(pattern) or f'_{pattern}' in acronym:
                        is_layer = True
                        break

            # 检查是否有ME子节点
            has_me_children = any('-ME' in child.get('acronym', '') for child in node.get('children', []))

            # 如果是layer节点或有ME子节点，则记录为Subregion
            if is_layer or has_me_children:
                self.subregions.append({
                    'subregion_id': int(node_id),
                    'acronym': acronym,
                    'name': name,
                    'parent_region': parent_acronym,  # Region节点
                    'rgb_triplet': rgb_triplet,
                    'has_me_children': has_me_children
                })
                self.subregion_to_region[acronym] = parent_acronym

        # 递归处理子节点
        for child in node.get('children', []):
            self._traverse_tree(child, acronym, parent_acronym)

    def get_subregions_by_region(self, region_acronym: str) -> List[Dict]:
        """获取指定Region下的所有Subregion"""
        return [sr for sr in self.subregions if sr['parent_region'] == region_acronym]

    def get_me_subregions_by_subregion(self, subregion_acronym: str) -> List[Dict]:
        """获取指定Subregion下的所有ME_Subregion"""
        return [me for me in self.me_subregions if me['parent_subregion'] == subregion_acronym]


# ==================== 在KnowledgeGraphBuilderNeo4j类中添加新方法 ====================

# 以下方法应该添加到KnowledgeGraphBuilderNeo4j类中

def generate_and_insert_subregion_nodes(self, subregion_loader: SubregionLoader):
    """
    生成并插入Subregion节点

    参数:
        subregion_loader: Subregion数据加载器
    """
    logger.info("生成并插入Subregion节点...")

    batch_nodes = []

    for subregion in tqdm(subregion_loader.subregions, desc="处理Subregion节点"):
        node_dict = {
            'subregion_id': subregion['subregion_id'],
            'acronym': subregion['acronym'],
            'name': subregion['name'],
            'parent_region': subregion['parent_region'],
            'has_me_children': subregion.get('has_me_children', False),
            'rgb_triplet': subregion.get('rgb_triplet', [])
        }

        batch_nodes.append(node_dict)

        # 批量插入
        if len(batch_nodes) >= BATCH_SIZE:
            self.neo4j.insert_nodes_batch('Subregion', batch_nodes)
            self.stats['subregions_inserted'] = self.stats.get('subregions_inserted', 0) + len(batch_nodes)
            batch_nodes = []

    # 插入剩余的节点
    if batch_nodes:
        self.neo4j.insert_nodes_batch('Subregion', batch_nodes)
        self.stats['subregions_inserted'] = self.stats.get('subregions_inserted', 0) + len(batch_nodes)

    logger.info(f"成功插入 {self.stats.get('subregions_inserted', 0)} 个Subregion节点")


def generate_and_insert_me_subregion_nodes(self, subregion_loader: SubregionLoader):
    """
    生成并插入ME_Subregion节点

    参数:
        subregion_loader: Subregion数据加载器
    """
    logger.info("生成并插入ME_Subregion节点...")

    batch_nodes = []

    for me_subregion in tqdm(subregion_loader.me_subregions, desc="处理ME_Subregion节点"):
        node_dict = {
            'me_subregion_id': me_subregion['me_subregion_id'],
            'acronym': me_subregion['acronym'],
            'name': me_subregion['name'],
            'parent_subregion': me_subregion['parent_subregion'],
            'rgb_triplet': me_subregion.get('rgb_triplet', [])
        }

        batch_nodes.append(node_dict)

        # 批量插入
        if len(batch_nodes) >= BATCH_SIZE:
            self.neo4j.insert_nodes_batch('ME_Subregion', batch_nodes)
            self.stats['me_subregions_inserted'] = self.stats.get('me_subregions_inserted', 0) + len(batch_nodes)
            batch_nodes = []

    # 插入剩余的节点
    if batch_nodes:
        self.neo4j.insert_nodes_batch('ME_Subregion', batch_nodes)
        self.stats['me_subregions_inserted'] = self.stats.get('me_subregions_inserted', 0) + len(batch_nodes)

    logger.info(f"成功插入 {self.stats.get('me_subregions_inserted', 0)} 个ME_Subregion节点")


def generate_and_insert_subregion_relationships(self, subregion_loader: SubregionLoader):
    """
    生成并插入Subregion相关的BELONGS_TO关系
    包括:
    1. ME_Subregion -> Subregion (BELONGS_TO)
    2. Subregion -> Region (BELONGS_TO)

    参数:
        subregion_loader: Subregion数据加载器
    """
    logger.info("生成并插入Subregion相关的BELONGS_TO关系...")

    with self.neo4j.driver.session(database=self.neo4j.database) as session:

        # 1. 插入 ME_Subregion -> Subregion 关系
        logger.info("插入 ME_Subregion -> Subregion 关系...")
        me_to_subregion_count = 0

        for me_acronym, subregion_acronym in tqdm(
                subregion_loader.me_to_subregion.items(),
                desc="ME_Subregion->Subregion"
        ):
            query = """
            MATCH (me:ME_Subregion {acronym: $me_acronym})
            MATCH (sr:Subregion {acronym: $subregion_acronym})
            CREATE (me)-[r:BELONGS_TO]->(sr)
            """
            try:
                session.run(query, me_acronym=me_acronym, subregion_acronym=subregion_acronym)
                me_to_subregion_count += 1
            except Exception as e:
                logger.error(f"插入关系失败 {me_acronym} -> {subregion_acronym}: {e}")

        logger.info(f"插入了 {me_to_subregion_count} 个 ME_Subregion->Subregion 关系")

        # 2. 插入 Subregion -> Region 关系
        logger.info("插入 Subregion -> Region 关系...")
        subregion_to_region_count = 0

        for subregion_acronym, region_acronym in tqdm(
                subregion_loader.subregion_to_region.items(),
                desc="Subregion->Region"
        ):
            query = """
            MATCH (sr:Subregion {acronym: $subregion_acronym})
            MATCH (r:Region {acronym: $region_acronym})
            CREATE (sr)-[rel:BELONGS_TO]->(r)
            """
            try:
                session.run(query, subregion_acronym=subregion_acronym, region_acronym=region_acronym)
                subregion_to_region_count += 1
            except Exception as e:
                logger.error(f"插入关系失败 {subregion_acronym} -> {region_acronym}: {e}")

        logger.info(f"插入了 {subregion_to_region_count} 个 Subregion->Region 关系")

        # 更新统计
        self.stats['relationships_inserted'] = self.stats.get('relationships_inserted', 0) + \
                                               me_to_subregion_count + subregion_to_region_count


def print_statistics_report_enhanced_with_subregion(self):
    """打印包含Subregion统计的增强报告"""
    report = []
    report.append("=" * 60)
    report.append("NeuroXiv 2.0 知识图谱Neo4j导入统计报告（含Subregion）")
    report.append("=" * 60)
    report.append("节点统计:")
    report.append(f"  - Region节点: {self.stats.get('regions_inserted', 0)}")
    report.append(f"  - Subregion节点: {self.stats.get('subregions_inserted', 0)}")
    report.append(f"  - ME_Subregion节点: {self.stats.get('me_subregions_inserted', 0)}")
    report.append(f"  - Neuron节点: {self.stats.get('neurons_inserted', 0)}")
    report.append(f"  - Class节点: {self.stats.get('classes_inserted', 0)}")
    report.append(f"  - Subclass节点: {self.stats.get('subclasses_inserted', 0)}")
    report.append(f"  - Supertype节点: {self.stats.get('supertypes_inserted', 0)}")
    report.append(f"  - Cluster节点: {self.stats.get('clusters_inserted', 0)}")

    total_nodes = sum([
        self.stats.get('regions_inserted', 0),
        self.stats.get('subregions_inserted', 0),
        self.stats.get('me_subregions_inserted', 0),
        self.stats.get('neurons_inserted', 0),
        self.stats.get('classes_inserted', 0),
        self.stats.get('subclasses_inserted', 0),
        self.stats.get('supertypes_inserted', 0),
        self.stats.get('clusters_inserted', 0)
    ])
    report.append(f"  - 总节点数: {total_nodes}")

    report.append(f"\n关系统计:")
    report.append(f"  - 总关系数: {self.stats.get('relationships_inserted', 0)}")
    report.append(f"\n生成时间: {pd.Timestamp.now()}")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    # 也保存到日志
    for line in report:
        logger.info(line)


# ==================== 使用示例 ====================

"""
在main函数中的使用方式：

def main(..., ccf_me_json: str = None):
    ...

    # 在Phase 4之前添加
    # Phase 3.75: 加载Subregion数据
    logger.info("Phase 3.75: 加载Subregion和ME_Subregion数据")

    ccf_me_path = Path(ccf_me_json) if ccf_me_json else data_path / "surf_tree_ccf-me.json"
    subregion_loader = SubregionLoader(ccf_me_path)

    if not subregion_loader.load_subregion_data():
        logger.warning("无法加载Subregion数据，跳过Subregion节点插入")
        subregion_loader = None

    # 在Phase 4中插入节点
    # 在插入Region节点之后添加：

    # 生成并插入Subregion节点
    if subregion_loader:
        builder.generate_and_insert_subregion_nodes(subregion_loader)
        builder.generate_and_insert_me_subregion_nodes(subregion_loader)

    # 在所有关系插入之后添加：

    # 生成并插入Subregion关系
    if subregion_loader:
        builder.generate_and_insert_subregion_relationships(subregion_loader)

    # 最后使用增强的统计报告
    builder.print_statistics_report_enhanced_with_subregion()
"""

if __name__ == "__main__":
    # 测试SubregionLoader
    import pandas as pd
    from loguru import logger

    ccf_me_json = Path("/mnt/user-data/uploads/surf_tree_ccf-me.json")

    loader = SubregionLoader(ccf_me_json)
    if loader.load_subregion_data():
        print(f"\n加载成功！")
        print(f"Subregion数量: {len(loader.subregions)}")
        print(f"ME_Subregion数量: {len(loader.me_subregions)}")

        # 显示MOp的例子
        print("\n" + "=" * 60)
        print("MOp的Subregion示例:")
        mop_subregions = loader.get_subregions_by_region('MOp')
        for sr in mop_subregions[:5]:
            print(f"  - {sr['acronym']}: {sr['name']}")
            mes = loader.get_me_subregions_by_subregion(sr['acronym'])
            if mes:
                print(f"    └─> {len(mes)} 个ME子区域")