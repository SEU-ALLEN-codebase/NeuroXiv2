import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm


class NeuronSubregionRelationshipInserter:
    """神经元-Subregion关系插入器"""

    def __init__(self, neo4j_connector, data_path: Path):
        """
        初始化

        参数:
            neo4j_connector: Neo4j连接器实例
            data_path: 数据目录路径
        """
        self.neo4j = neo4j_connector
        self.data_path = data_path
        self.neuron_subregion_df = None

        # 统计
        self.stats = {
            'locate_at_subregion': 0,
            'locate_at_me_subregion': 0,
            'neurons_with_subregion': 0,
            'neurons_with_me_subregion': 0
        }

    def load_neuron_subregion_mapping(self) -> bool:
        """
        加载神经元-Subregion映射数据（修复版）

        返回:
            bool: 是否加载成功
        """
        logger.info("加载神经元-Subregion映射数据...")

        # 查找CSV文件
        csv_file = self.data_path / "info_with_me_subregion_v2.csv"

        if not csv_file.exists():
            logger.error(f"找不到映射文件: {csv_file}")
            return False

        try:
            # 读取CSV文件（无表头）
            self.neuron_subregion_df = pd.read_csv(
                csv_file,
                header=None,
                encoding='latin1',
                low_memory=False
            )

            # 定义列名（根据你的文件结构）
            col_names = [
                'ID',  # 0 - 神经元ID
                'data_source',  # 1
                'celltype',  # 2
                'brain_atlas',  # 3
                'recon_method',  # 4
                'has_recon_axon',  # 5
                'has_recon_den',  # 6
                'has_ab',  # 7
                'layer',  # 8
                'has_apical',  # 9
                'has_local',  # 10
                'hemisphere',  # 11
                'subregion',  # 12 - Subregion的acronym
                'me_subregion_voxel',  # 13 - ME子区域的voxel值
                'me_subregion_acronym',  # 14 - ME子区域的acronym
                'me_subregion_name',  # 15 - ME子区域的完整名称
                'parent_region_id',  # 16 - 父区域ID
                'is_me_subregion'  # 17 - 是否为ME子区域
            ]

            self.neuron_subregion_df.columns = col_names

            logger.info(f"成功加载 {len(self.neuron_subregion_df)} 条神经元映射记录")

            # 统计有映射的神经元（修复版）
            neurons_with_subregion = self.neuron_subregion_df['subregion'].notna().sum()


            neurons_with_me = (self.neuron_subregion_df['is_me_subregion'].astype(str).str.lower() == "true").sum()

            total_neurons = len(self.neuron_subregion_df)

            logger.info(
                f"有Subregion映射的神经元: {neurons_with_subregion} ({neurons_with_subregion / total_neurons * 100:.2f}%)")
            logger.info(f"有ME_Subregion映射的神经元: {neurons_with_me} ({neurons_with_me / total_neurons * 100:.2f}%)")

            # 显示样例
            logger.info("\n样例数据（前5条有ME映射的记录）:")
            me_samples = self.neuron_subregion_df[self.neuron_subregion_df['is_me_subregion'] == True].head()
            for idx, row in me_samples.iterrows():
                logger.info(f"  神经元: {row['ID']}")
                logger.info(f"    Celltype: {row['celltype']}")
                logger.info(f"    Subregion: {row['subregion']}")
                logger.info(f"    ME_Subregion: {row['me_subregion_acronym']}")
                logger.info(f"    ME体素值: {row['me_subregion_voxel']}")
                logger.info("")

            return True

        except Exception as e:
            logger.error(f"加载映射文件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def insert_locate_at_subregion_relationships(self, batch_size: int = 1000):
        """
        插入 LOCATE_AT_SUBREGION 关系

        Neuron -> Subregion
        基于celltype中的Subregion信息（第12列）

        参数:
            batch_size: 批量插入大小
        """
        logger.info("插入 LOCATE_AT_SUBREGION 关系...")

        if self.neuron_subregion_df is None:
            logger.error("未加载映射数据")
            return

        # 筛选有Subregion信息的神经元
        valid_neurons = self.neuron_subregion_df[
            self.neuron_subregion_df['subregion'].notna()
        ].copy()

        logger.info(f"找到 {len(valid_neurons)} 个有Subregion信息的神经元")

        batch_relationships = []
        success_count = 0
        failed_count = 0

        for idx, row in tqdm(valid_neurons.iterrows(),
                             total=len(valid_neurons),
                             desc="处理LOCATE_AT_SUBREGION关系"):
            neuron_id = str(row['ID'])
            subregion_acronym = str(row['subregion']).strip()

            if not subregion_acronym or subregion_acronym == 'nan':
                continue

            rel = {
                'neuron_id': neuron_id,
                'subregion_acronym': subregion_acronym
            }

            batch_relationships.append(rel)

            # 批量插入
            if len(batch_relationships) >= batch_size:
                success, failed = self._execute_subregion_batch(batch_relationships)
                success_count += success
                failed_count += failed
                batch_relationships = []

        # 插入剩余的关系
        if batch_relationships:
            success, failed = self._execute_subregion_batch(batch_relationships)
            success_count += success
            failed_count += failed

        self.stats['locate_at_subregion'] = success_count
        self.stats['neurons_with_subregion'] = len(valid_neurons)

        logger.info(f"成功插入 {success_count} 个 LOCATE_AT_SUBREGION 关系")
        if failed_count > 0:
            logger.warning(f"失败 {failed_count} 个关系（可能是Neuron或Subregion节点不存在）")

    def _execute_subregion_batch(self, batch):
        """执行Subregion关系批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (n:Neuron {neuron_id: rel.neuron_id})
        MATCH (sr:Subregion {acronym: rel.subregion_acronym})
        MERGE (n)-[r:LOCATE_AT_SUBREGION]->(sr)
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                created_count = record['created_count'] if record else 0
                failed_count = len(batch) - created_count
                return created_count, failed_count
        except Exception as e:
            logger.error(f"批量插入LOCATE_AT_SUBREGION关系失败: {e}")
            return 0, len(batch)

    def insert_locate_at_me_subregion_relationships(self, batch_size: int = 1000):
        """
        插入 LOCATE_AT_ME_SUBREGION 关系

        Neuron -> ME_Subregion
        基于映射得到的ME_Subregion信息（第14列）

        参数:
            batch_size: 批量插入大小
        """
        logger.info("插入 LOCATE_AT_ME_SUBREGION 关系...")

        if self.neuron_subregion_df is None:
            logger.error("未加载映射数据")
            return

        # 筛选有ME_Subregion映射的神经元
        valid_neurons = self.neuron_subregion_df[
            self.neuron_subregion_df['is_me_subregion'].astype(str).str.lower() == "true"].copy()

        logger.info(f"找到 {len(valid_neurons)} 个有ME_Subregion映射的神经元")

        batch_relationships = []
        success_count = 0
        failed_count = 0

        for idx, row in tqdm(valid_neurons.iterrows(),
                             total=len(valid_neurons),
                             desc="处理LOCATE_AT_ME_SUBREGION关系"):
            neuron_id = str(row['ID'])
            me_subregion_acronym = str(row['me_subregion_acronym']).strip()

            if not me_subregion_acronym or me_subregion_acronym == 'nan':
                continue

            rel = {
                'neuron_id': neuron_id,
                'me_subregion_acronym': me_subregion_acronym
            }

            batch_relationships.append(rel)

            # 批量插入
            if len(batch_relationships) >= batch_size:
                success, failed = self._execute_me_subregion_batch(batch_relationships)
                success_count += success
                failed_count += failed
                batch_relationships = []

        # 插入剩余的关系
        if batch_relationships:
            success, failed = self._execute_me_subregion_batch(batch_relationships)
            success_count += success
            failed_count += failed

        self.stats['locate_at_me_subregion'] = success_count
        self.stats['neurons_with_me_subregion'] = len(valid_neurons)

        logger.info(f"成功插入 {success_count} 个 LOCATE_AT_ME_SUBREGION 关系")
        if failed_count > 0:
            logger.warning(f"失败 {failed_count} 个关系（可能是Neuron或ME_Subregion节点不存在）")

    def _execute_me_subregion_batch(self, batch):
        """执行ME_Subregion关系批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (n:Neuron {neuron_id: rel.neuron_id})
        MATCH (me:ME_Subregion {acronym: rel.me_subregion_acronym})
        MERGE (n)-[r:LOCATE_AT_ME_SUBREGION]->(me)
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                created_count = record['created_count'] if record else 0
                failed_count = len(batch) - created_count
                return created_count, failed_count
        except Exception as e:
            logger.error(f"批量插入LOCATE_AT_ME_SUBREGION关系失败: {e}")
            return 0, len(batch)

    def insert_all_relationships(self, batch_size: int = 1000):
        """
        插入所有关系

        参数:
            batch_size: 批量插入大小
        """
        logger.info("=" * 60)
        logger.info("开始插入神经元-Subregion关系")
        logger.info("=" * 60)

        # 1. 插入 LOCATE_AT_SUBREGION 关系
        self.insert_locate_at_subregion_relationships(batch_size)

        # 2. 插入 LOCATE_AT_ME_SUBREGION 关系
        self.insert_locate_at_me_subregion_relationships(batch_size)

        # 打印统计
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("神经元-Subregion关系插入统计")
        logger.info("=" * 60)
        logger.info(f"LOCATE_AT_SUBREGION 关系: {self.stats['locate_at_subregion']}")
        logger.info(f"  - 涉及神经元数: {self.stats['neurons_with_subregion']}")
        logger.info(f"\nLOCATE_AT_ME_SUBREGION 关系: {self.stats['locate_at_me_subregion']}")
        logger.info(f"  - 涉及神经元数: {self.stats['neurons_with_me_subregion']}")
        logger.info(f"\n总插入关系数: {self.stats['locate_at_subregion'] + self.stats['locate_at_me_subregion']}")
        logger.info("=" * 60)


# ==================== 验证查询 ====================

def verify_relationships(neo4j_connector, database='neo4j'):
    """
    验证插入的关系

    参数:
        neo4j_connector: Neo4j连接器
        database: 数据库名称
    """
    logger.info("=" * 60)
    logger.info("验证插入的关系")
    logger.info("=" * 60)

    with neo4j_connector.driver.session(database=database) as session:
        # 1. 验证 LOCATE_AT_SUBREGION
        query1 = """
        MATCH (n:Neuron)-[r:LOCATE_AT_SUBREGION]->(sr:Subregion)
        RETURN count(r) as count
        """
        result1 = session.run(query1)
        count1 = result1.single()['count']
        logger.info(f"✓ LOCATE_AT_SUBREGION 关系数: {count1}")

        # 2. 验证 LOCATE_AT_ME_SUBREGION
        query2 = """
        MATCH (n:Neuron)-[r:LOCATE_AT_ME_SUBREGION]->(me:ME_Subregion)
        RETURN count(r) as count
        """
        result2 = session.run(query2)
        count2 = result2.single()['count']
        logger.info(f"✓ LOCATE_AT_ME_SUBREGION 关系数: {count2}")

        # 3. 显示样例
        logger.info("\n样例关系（LOCATE_AT_SUBREGION）:")
        query3 = """
        MATCH (n:Neuron)-[r:LOCATE_AT_SUBREGION]->(sr:Subregion)
        RETURN n.neuron_id as neuron_id, n.celltype as celltype, sr.acronym as subregion
        LIMIT 5
        """
        result3 = session.run(query3)
        for record in result3:
            logger.info(f"  {record['neuron_id']} ({record['celltype']}) -> {record['subregion']}")

        logger.info("\n样例关系（LOCATE_AT_ME_SUBREGION）:")
        query4 = """
        MATCH (n:Neuron)-[r:LOCATE_AT_ME_SUBREGION]->(me:ME_Subregion)
        RETURN n.neuron_id as neuron_id, n.celltype as celltype, me.acronym as me_subregion
        LIMIT 5
        """
        result4 = session.run(query4)
        for record in result4:
            logger.info(f"  {record['neuron_id']} ({record['celltype']}) -> {record['me_subregion']}")

    logger.info("=" * 60)