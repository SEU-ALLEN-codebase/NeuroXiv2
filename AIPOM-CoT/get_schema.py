import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from neo4j import GraphDatabase
from loguru import logger
import pandas as pd
from tqdm import tqdm


class Neo4jSchemaExtractor:
    """Neo4j知识图谱Schema提取器"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j"):
        """
        初始化连接

        参数:
            uri: Neo4j URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

        # 存储schema信息
        self.schema = {
            'nodes': {},
            'relationships': {},
            'statistics': {}
        }

    def connect(self) -> bool:
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 测试连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info(f"✓ 成功连接到Neo4j: {self.uri}/{self.database}")
            return True
        except Exception as e:
            logger.error(f"✗ 连接失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("✓ 连接已关闭")

    def extract_node_labels(self) -> List[str]:
        """提取所有节点标签"""
        logger.info("提取节点标签...")

        with self.driver.session(database=self.database) as session:
            result = session.run("CALL db.labels()")
            labels = [record['label'] for record in result]

        logger.info(f"  找到 {len(labels)} 种节点类型")
        return sorted(labels)

    def extract_node_properties(self, label: str) -> Dict:
        """
        提取指定节点类型的所有属性

        返回:
            {
                'properties': {prop_name: {type, sample_value, non_null_count}},
                'count': 节点总数,
                'sample_nodes': 样例节点列表
            }
        """
        logger.info(f"  分析节点类型: {label}")

        with self.driver.session(database=self.database) as session:
            # 1. 获取节点总数
            count_query = f"MATCH (n:{label}) RETURN count(n) as count"
            count_result = session.run(count_query)
            total_count = count_result.single()['count']

            # 2. 获取样例节点（最多100个）
            sample_query = f"MATCH (n:{label}) RETURN n LIMIT 100"
            sample_result = session.run(sample_query)
            sample_nodes = [dict(record['n']) for record in sample_result]

            # 3. 分析属性
            properties = {}
            if sample_nodes:
                # 收集所有属性名
                all_props = set()
                for node in sample_nodes:
                    all_props.update(node.keys())

                # 分析每个属性
                for prop in all_props:
                    # 统计非空值数量
                    non_null_query = f"""
                    MATCH (n:{label})
                    WHERE n.{prop} IS NOT NULL
                    RETURN count(n) as non_null_count
                    """
                    non_null_result = session.run(non_null_query)
                    non_null_count = non_null_result.single()['non_null_count']

                    # 获取样例值和类型
                    sample_values = [node.get(prop) for node in sample_nodes if prop in node]
                    sample_value = sample_values[0] if sample_values else None

                    # 推断类型
                    prop_type = type(sample_value).__name__ if sample_value is not None else 'None'

                    properties[prop] = {
                        'type': prop_type,
                        'sample_value': sample_value,
                        'non_null_count': non_null_count,
                        'coverage': f"{non_null_count}/{total_count} ({non_null_count / total_count * 100:.1f}%)"
                    }

        return {
            'properties': properties,
            'count': total_count,
            'sample_nodes': sample_nodes[:3]  # 只保留3个样例
        }

    def extract_relationship_types(self) -> List[str]:
        """提取所有关系类型"""
        logger.info("提取关系类型...")

        with self.driver.session(database=self.database) as session:
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record['relationshipType'] for record in result]

        logger.info(f"  找到 {len(rel_types)} 种关系类型")
        return sorted(rel_types)

    def extract_relationship_details(self, rel_type: str) -> Dict:
        """
        提取指定关系类型的详细信息

        返回:
            {
                'count': 关系总数,
                'properties': {prop_name: {type, sample_value}},
                'patterns': [(source_label, target_label, count)]
                'sample_relationships': 样例关系列表
            }
        """
        logger.info(f"  分析关系类型: {rel_type}")

        with self.driver.session(database=self.database) as session:
            # 1. 获取关系总数
            count_query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            count_result = session.run(count_query)
            total_count = count_result.single()['count']

            # 2. 获取样例关系
            sample_query = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            RETURN labels(a) as source_labels, 
                   labels(b) as target_labels,
                   r,
                   a.name as source_name,
                   b.name as target_name
            LIMIT 100
            """
            sample_result = session.run(sample_query)
            sample_rels = []
            for record in sample_result:
                sample_rels.append({
                    'source_labels': record['source_labels'],
                    'target_labels': record['target_labels'],
                    'properties': dict(record['r']),
                    'source_name': record['source_name'],
                    'target_name': record['target_name']
                })

            # 3. 分析属性
            properties = {}
            if sample_rels:
                all_props = set()
                for rel in sample_rels:
                    all_props.update(rel['properties'].keys())

                for prop in all_props:
                    sample_values = [
                        rel['properties'].get(prop)
                        for rel in sample_rels
                        if prop in rel['properties']
                    ]
                    sample_value = sample_values[0] if sample_values else None
                    prop_type = type(sample_value).__name__ if sample_value is not None else 'None'

                    # 统计非空值
                    non_null_query = f"""
                    MATCH ()-[r:{rel_type}]->()
                    WHERE r.{prop} IS NOT NULL
                    RETURN count(r) as non_null_count
                    """
                    non_null_result = session.run(non_null_query)
                    non_null_count = non_null_result.single()['non_null_count']

                    properties[prop] = {
                        'type': prop_type,
                        'sample_value': sample_value,
                        'non_null_count': non_null_count,
                        'coverage': f"{non_null_count}/{total_count} ({non_null_count / total_count * 100:.1f}%)"
                    }

            # 4. 提取关系模式（source -> target）
            pattern_query = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WITH labels(a)[0] as source, labels(b)[0] as target, count(r) as cnt
            RETURN source, target, cnt
            ORDER BY cnt DESC
            """
            pattern_result = session.run(pattern_query)
            patterns = [
                (record['source'], record['target'], record['cnt'])
                for record in pattern_result
            ]

        return {
            'count': total_count,
            'properties': properties,
            'patterns': patterns,
            'sample_relationships': sample_rels[:3]
        }

    def extract_full_schema(self):
        """提取完整schema"""
        logger.info("=" * 80)
        logger.info("开始提取完整Schema")
        logger.info("=" * 80)

        # 1. 提取节点信息
        logger.info("\n[1/3] 提取节点信息...")
        node_labels = self.extract_node_labels()

        for label in tqdm(node_labels, desc="分析节点"):
            self.schema['nodes'][label] = self.extract_node_properties(label)

        # 2. 提取关系信息
        logger.info("\n[2/3] 提取关系信息...")
        rel_types = self.extract_relationship_types()

        for rel_type in tqdm(rel_types, desc="分析关系"):
            self.schema['relationships'][rel_type] = self.extract_relationship_details(rel_type)

        # 3. 统计信息
        logger.info("\n[3/3] 计算统计信息...")
        self.calculate_statistics()

        logger.info("\n✓ Schema提取完成")

    def calculate_statistics(self):
        """计算统计信息"""
        with self.driver.session(database=self.database) as session:
            # 总节点数
            node_count_query = "MATCH (n) RETURN count(n) as count"
            total_nodes = session.run(node_count_query).single()['count']

            # 总关系数
            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            total_rels = session.run(rel_count_query).single()['count']

            self.schema['statistics'] = {
                'total_nodes': total_nodes,
                'total_relationships': total_rels,
                'node_types': len(self.schema['nodes']),
                'relationship_types': len(self.schema['relationships']),
                'node_breakdown': {
                    label: info['count']
                    for label, info in self.schema['nodes'].items()
                },
                'relationship_breakdown': {
                    rel_type: info['count']
                    for rel_type, info in self.schema['relationships'].items()
                }
            }

    def print_schema_report(self):
        """打印格式化的Schema报告"""
        print("\n" + "=" * 80)
        print("NeuroXiv 2.0 知识图谱 Schema 报告")
        print("=" * 80)

        # 统计概览
        stats = self.schema['statistics']
        print(f"\n📊 统计概览:")
        print(f"  总节点数: {stats['total_nodes']:,}")
        print(f"  总关系数: {stats['total_relationships']:,}")
        print(f"  节点类型: {stats['node_types']}")
        print(f"  关系类型: {stats['relationship_types']}")

        # 节点详情
        print("\n" + "=" * 80)
        print("🔵 节点类型详情")
        print("=" * 80)

        for label, info in sorted(self.schema['nodes'].items()):
            print(f"\n【{label}】")
            print(f"  数量: {info['count']:,}")
            print(f"  属性: {len(info['properties'])}")

            if info['properties']:
                print("\n  属性列表:")
                for prop_name, prop_info in sorted(info['properties'].items()):
                    print(f"    • {prop_name}")
                    print(f"      - 类型: {prop_info['type']}")
                    print(f"      - 覆盖率: {prop_info['coverage']}")
                    if prop_info['sample_value'] is not None:
                        sample_str = str(prop_info['sample_value'])
                        if len(sample_str) > 50:
                            sample_str = sample_str[:50] + "..."
                        print(f"      - 样例: {sample_str}")

        # 关系详情
        print("\n" + "=" * 80)
        print("🔗 关系类型详情")
        print("=" * 80)

        for rel_type, info in sorted(self.schema['relationships'].items()):
            print(f"\n【{rel_type}】")
            print(f"  数量: {info['count']:,}")

            if info['patterns']:
                print("\n  关系模式:")
                for source, target, count in info['patterns']:
                    percentage = (count / info['count'] * 100) if info['count'] > 0 else 0
                    print(f"    • ({source})-[{rel_type}]->({target}): {count:,} ({percentage:.1f}%)")

            if info['properties']:
                print("\n  关系属性:")
                for prop_name, prop_info in sorted(info['properties'].items()):
                    print(f"    • {prop_name}")
                    print(f"      - 类型: {prop_info['type']}")
                    print(f"      - 覆盖率: {prop_info['coverage']}")
                    if prop_info['sample_value'] is not None:
                        sample_str = str(prop_info['sample_value'])
                        if len(sample_str) > 50:
                            sample_str = sample_str[:50] + "..."
                        print(f"      - 样例: {sample_str}")

        print("\n" + "=" * 80)

    def export_to_json(self, output_file: str = "schema.json"):
        """导出schema到JSON文件"""
        logger.info(f"导出Schema到 {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✓ Schema已导出到 {output_file}")

    def export_to_markdown(self, output_file: str = "neuroxiv_schema.md"):
        """导出schema到Markdown文件"""
        logger.info(f"导出Schema到 {output_file}...")

        lines = []
        lines.append("# NeuroXiv 2.0 知识图谱 Schema\n")

        # 统计概览
        stats = self.schema['statistics']
        lines.append("## 统计概览\n")
        lines.append(f"- **总节点数**: {stats['total_nodes']:,}")
        lines.append(f"- **总关系数**: {stats['total_relationships']:,}")
        lines.append(f"- **节点类型数**: {stats['node_types']}")
        lines.append(f"- **关系类型数**: {stats['relationship_types']}\n")

        # 节点类型
        lines.append("## 节点类型\n")
        for label, info in sorted(self.schema['nodes'].items()):
            lines.append(f"### {label}\n")
            lines.append(f"**数量**: {info['count']:,}\n")

            if info['properties']:
                lines.append("**属性**:\n")
                lines.append("| 属性名 | 类型 | 覆盖率 | 样例值 |")
                lines.append("|--------|------|--------|--------|")

                for prop_name, prop_info in sorted(info['properties'].items()):
                    sample = str(prop_info['sample_value']) if prop_info['sample_value'] is not None else "N/A"
                    if len(sample) > 30:
                        sample = sample[:30] + "..."
                    lines.append(f"| `{prop_name}` | {prop_info['type']} | {prop_info['coverage']} | {sample} |")
                lines.append("")

        # 关系类型
        lines.append("## 关系类型\n")
        for rel_type, info in sorted(self.schema['relationships'].items()):
            lines.append(f"### {rel_type}\n")
            lines.append(f"**数量**: {info['count']:,}\n")

            if info['patterns']:
                lines.append("**关系模式**:\n")
                for source, target, count in info['patterns']:
                    percentage = (count / info['count'] * 100) if info['count'] > 0 else 0
                    lines.append(f"- `({source})-[{rel_type}]->({target})`: {count:,} ({percentage:.1f}%)")
                lines.append("")

            if info['properties']:
                lines.append("**关系属性**:\n")
                lines.append("| 属性名 | 类型 | 覆盖率 | 样例值 |")
                lines.append("|--------|------|--------|--------|")

                for prop_name, prop_info in sorted(info['properties'].items()):
                    sample = str(prop_info['sample_value']) if prop_info['sample_value'] is not None else "N/A"
                    if len(sample) > 30:
                        sample = sample[:30] + "..."
                    lines.append(f"| `{prop_name}` | {prop_info['type']} | {prop_info['coverage']} | {sample} |")
                lines.append("")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"✓ Schema已导出到 {output_file}")

    def generate_cypher_visualization(self, output_file: str = "schema_visualization.cypher"):
        """生成Cypher查询用于可视化schema"""
        logger.info(f"生成可视化查询到 {output_file}...")

        lines = []
        lines.append("// NeuroXiv 2.0 Schema 可视化查询")
        lines.append("// 在Neo4j Browser中运行以下查询\n")

        # 1. 显示所有节点类型样例
        lines.append("// 1. 显示每种节点类型的样例")
        for label in self.schema['nodes'].keys():
            lines.append(f"MATCH (n:{label}) RETURN n LIMIT 5;")
        lines.append("")

        # 2. 显示关系模式
        lines.append("// 2. 显示所有关系模式")
        for rel_type, info in self.schema['relationships'].items():
            if info['patterns']:
                source, target, _ = info['patterns'][0]
                lines.append(f"MATCH (a:{source})-[r:{rel_type}]->(b:{target}) RETURN a, r, b LIMIT 10;")
        lines.append("")

        # 3. 统计查询
        lines.append("// 3. 统计查询")
        lines.append("CALL db.labels() YIELD label")
        lines.append("CALL apoc.cypher.run('MATCH (:`'+label+'`) RETURN count(*) as count', {}) YIELD value")
        lines.append("RETURN label, value.count ORDER BY value.count DESC;")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"✓ 可视化查询已生成到 {output_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱 Schema 提取器')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j',
                        help='Neo4j用户名')
    parser.add_argument('--password', type=str, default='neuroxiv',required=True,
                        help='Neo4j密码')
    parser.add_argument('--database', type=str, default='neo4j',
                        help='数据库名称')
    parser.add_argument('--output-dir', type=str, default='./schema_output',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 初始化提取器
    extractor = Neo4jSchemaExtractor(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database
    )

    try:
        # 连接数据库
        if not extractor.connect():
            return

        # 提取schema
        extractor.extract_full_schema()

        # 打印报告
        extractor.print_schema_report()

        # 导出文件
        extractor.export_to_json(output_dir / "schema.json")
        extractor.export_to_markdown(output_dir / "neuroxiv_schema.md")
        extractor.generate_cypher_visualization(output_dir / "schema_visualization.cypher")

        logger.info("\n" + "=" * 80)
        logger.info("✓ 所有文件已生成在目录: " + str(output_dir))
        logger.info("=" * 80)

    finally:
        extractor.close()


if __name__ == "__main__":
    main()