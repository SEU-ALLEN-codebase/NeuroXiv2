import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq

from aipom_cot_true_agent_v2 import RealSchemaCache
from intelligent_entity_recognition import EntityMatch, EntityCluster

logger = logging.getLogger(__name__)


# ==================== Schema Path Structures ====================

@dataclass
class SchemaPath:
    """Schema中的一条路径"""
    start_label: str
    end_label: str
    hops: List[Tuple[str, str, str]]  # [(source, rel_type, target), ...]
    score: float
    total_relationships: int


@dataclass
class QueryPlan:
    """基于schema path的查询计划"""
    step_number: int
    purpose: str
    action: str
    schema_path: SchemaPath
    cypher_template: str
    parameters: Dict
    depends_on: List[int]
    modality: Optional[str] = None


# ==================== Schema Graph Builder ====================

class SchemaGraph:
    """
    Schema的图表示

    节点: Node labels
    边: Relationship types with properties
    """

    def __init__(self, schema: RealSchemaCache):
        self.schema = schema

        # Build adjacency list
        self.graph = defaultdict(list)  # {source: [(rel_type, target, count), ...]}
        self.reverse_graph = defaultdict(list)  # For bidirectional search

        self._build_graph()

    def _build_graph(self):
        """从schema构建图"""
        for rel_type, rel_info in self.schema.rel_types.items():
            patterns = rel_info['patterns']
            total_count = rel_info['count']

            for source, target, count in patterns:
                # Forward edge
                self.graph[source].append((rel_type, target, count))

                # Reverse edge
                self.reverse_graph[target].append((rel_type, source, count))

        logger.info(f"  ✓ Schema graph: {len(self.graph)} nodes, "
                    f"{sum(len(v) for v in self.graph.values())} edges")

    def get_neighbors(self, label: str) -> List[Tuple[str, str, int]]:
        """获取某个label的所有邻居"""
        return self.graph.get(label, [])

    def get_reverse_neighbors(self, label: str) -> List[Tuple[str, str, int]]:
        """获取反向邻居 (谁指向这个label)"""
        return self.reverse_graph.get(label, [])


# ==================== Path Finding Algorithms ====================

class SchemaPathFinder:
    """
    在Schema中寻找最优路径（修复版）

    🔧 关键修复：
    1. 统一元组格式：内部用4元组，输出用3元组
    2. 修复_score_path方法
    3. 添加空路径保护
    """

    def __init__(self, schema_graph: SchemaGraph):
        self.graph = schema_graph

    def find_paths(self,
                   start_label: str,
                   end_label: str,
                   max_hops: int = 3,
                   max_paths: int = 5) -> List[SchemaPath]:
        """
        找到从start到end的所有可行路径（修复版）

        🔧 修复：
        - 内部使用4元组 (source, rel_type, target, count)
        - 输出转换为3元组 (source, rel_type, target)
        """

        if start_label == end_label:
            return []

        logger.debug(f"  Finding paths: {start_label} -> {end_label}")

        paths = []
        queue = deque([([start_label], [])])  # (node_path, edge_path)
        visited = set()

        while queue:
            node_path, edge_path = queue.popleft()
            current = node_path[-1]

            # Check depth
            if len(edge_path) >= max_hops:
                continue

            # ✅ Found target
            if current == end_label and len(edge_path) > 0:
                # ✅ 正确处理4元组 -> 3元组
                score = self._score_path(edge_path)
                total_rels = sum(count for _, _, _, count in edge_path)  # ✅ 4元组

                # ✅ 转换为3元组
                hops = [
                    (source, rel_type, target)
                    for source, rel_type, target, _ in edge_path
                ]

                paths.append(SchemaPath(
                    start_label=start_label,
                    end_label=end_label,
                    hops=hops,  # ✅ 3元组列表
                    score=score,
                    total_relationships=total_rels
                ))
                continue

            # Avoid revisiting same state
            state = (current, tuple(edge_path))
            if state in visited:
                continue
            visited.add(state)

            # Expand neighbors
            for rel_type, next_node, count in self.graph.get_neighbors(current):
                new_node_path = node_path + [next_node]
                # ✅ 保持4元组格式用于内部处理
                new_edge_path = edge_path + [(current, rel_type, next_node, count)]
                queue.append((new_node_path, new_edge_path))

        # Sort by score
        paths.sort(key=lambda p: p.score, reverse=True)

        logger.debug(f"    Found {len(paths)} paths")

        return paths[:max_paths]

    def _score_path(self, edge_path: List[Tuple]) -> float:
        """
        评估路径质量（修复版）

        🔧 修复：正确解包4元组 (source, rel_type, target, count)
        """
        if not edge_path:
            return 0.0

        score = 1.0

        # Factor 1: Path length penalty
        length_penalty = 1.0 / (1 + 0.5 * (len(edge_path) - 1))
        score *= length_penalty

        # Factor 2: Relationship frequency bonus
        # ✅ 正确解包4元组
        total_count = sum(count for _, _, _, count in edge_path)
        freq_bonus = min(1.5, 1.0 + (total_count / 10000))
        score *= freq_bonus

        # Factor 3: Preferred relationships
        preferred_rels = {
            'HAS_CLUSTER': 1.3,
            'PROJECT_TO': 1.3,
            'LOCATE_AT': 1.2,
            'HAS_SUBCLASS': 1.2,
            'BELONGS_TO': 1.1
        }

        # ✅ 正确解包4元组
        for _, rel_type, _, _ in edge_path:
            if rel_type in preferred_rels:
                score *= preferred_rels[rel_type]

        return score

    def find_shortest_path(self, start_label: str, end_label: str) -> Optional[SchemaPath]:
        """找到最短路径 (BFS)"""
        paths = self.find_paths(start_label, end_label, max_hops=3, max_paths=1)
        return paths[0] if paths else None


# ==================== Dynamic Query Planner ====================

class DynamicSchemaPathPlanner:
    """
    基于schema path动态生成查询计划

    工作流程:
    1. 识别实体类型
    2. 在schema中寻找连接路径
    3. 根据路径生成Cypher查询
    4. 组装完整推理计划
    """

    def __init__(self, schema: RealSchemaCache):
        self.schema = schema
        self.schema_graph = SchemaGraph(schema)
        self.path_finder = SchemaPathFinder(self.schema_graph)

    def generate_plan(self,
                      entity_clusters: List[EntityCluster],
                      question: str) -> List[QueryPlan]:
        """
        生成动态查询计划

        Args:
            entity_clusters: 识别的实体聚类
            question: 原始问题

        Returns:
            查询计划列表
        """
        if not entity_clusters:
            return self._generate_exploratory_plan()

        primary_cluster = entity_clusters[0]

        if primary_cluster.cluster_type == 'gene_marker':
            return self._plan_gene_marker_analysis(primary_cluster, entity_clusters, question)

        elif primary_cluster.cluster_type == 'region':
            return self._plan_region_analysis(primary_cluster, entity_clusters, question)

        elif primary_cluster.cluster_type == 'cell_type':
            return self._plan_celltype_analysis(primary_cluster, entity_clusters, question)

        else:
            return self._generate_exploratory_plan()

    def _plan_gene_marker_analysis(self, gene_cluster, all_clusters, question):
        """基因marker分析的动态规划（修复版）"""
        plans = []
        gene_name = gene_cluster.primary_entity.entity_id

        # Step 1: Find clusters
        plans.append(QueryPlan(
            step_number=1,
            purpose=f"Find cell clusters expressing {gene_name}",
            action="execute_cypher",
            schema_path=SchemaPath(
                start_label="Cluster",
                end_label="Cluster",
                hops=[],  # ✅ 空列表（不是None）
                score=1.0,
                total_relationships=0
            ),
            cypher_template="""
            MATCH (c:Cluster)
            WHERE c.markers CONTAINS $gene
            RETURN c.name AS cluster_name,
                   c.markers AS markers,
                   c.number_of_neurons AS neuron_count,
                   c.broad_region_distribution AS region_dist
            ORDER BY c.number_of_neurons DESC
            LIMIT 20
            """,
            parameters={'gene': gene_name},
            depends_on=[],
            modality='molecular'
        ))

        # Step 2: Find enriched regions
        region_cluster_path = self.path_finder.find_shortest_path('Region', 'Cluster')

        # ✅ 修复：如果找不到路径，使用默认路径
        if not region_cluster_path:
            logger.warning("   Could not find Region->Cluster path, using default")
            region_cluster_path = SchemaPath(
                start_label="Region",
                end_label="Cluster",
                hops=[("Region", "HAS_CLUSTER", "Cluster")],  # ✅ 3元组
                score=1.0,
                total_relationships=10000
            )

        plans.append(QueryPlan(
            step_number=2,
            purpose=f"Identify regions enriched for {gene_name}+ clusters",
            action="execute_cypher",
            schema_path=region_cluster_path,  # ✅ 保证不是None
            cypher_template="""
            MATCH (r:Region)-[h:HAS_CLUSTER]->(c:Cluster)
            WHERE c.markers CONTAINS $gene
            RETURN r.acronym AS region,
                   r.name AS region_name,
                   count(c) AS cluster_count,
                   sum(c.number_of_neurons) AS total_neurons,
                   collect(c.name)[0..5] AS sample_clusters
            ORDER BY cluster_count DESC
            LIMIT 15
            """,
            parameters={'gene': gene_name},
            depends_on=[1],
            modality='molecular'
        ))

        # Step 3: Morphology (如果问题提到)
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['morpholog', 'feature', 'structure', 'axon', 'dendrite']):
            plans.append(QueryPlan(
                step_number=3,
                purpose=f"Analyze morphological features of {gene_name}+ enriched regions",
                action="execute_cypher",
                schema_path=SchemaPath(
                    start_label="Region",
                    end_label="Neuron",
                    hops=[("Neuron", "LOCATE_AT", "Region")],  # ✅ 3元组
                    score=0.9,
                    total_relationships=5000
                ),
                cypher_template="""
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                WHERE r.acronym IN $enriched_regions
                RETURN r.acronym AS region,
                       count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon_length,
                       avg(n.dendritic_length) AS avg_dendrite_length,
                       avg(n.axonal_branches) AS avg_axon_branches,
                       avg(n.dendritic_branches) AS avg_dendrite_branches
                ORDER BY neuron_count DESC
                LIMIT 20
                """,
                parameters={'enriched_regions': []},
                depends_on=[2],
                modality='morphological'
            ))

        # Step 4: Projections (如果问题提到)
        if any(kw in question_lower for kw in ['project', 'target', 'connect', 'output', 'pathway']):
            plans.append(QueryPlan(
                step_number=len(plans) + 1,
                purpose=f"Identify projection targets of {gene_name}+ regions",
                action="execute_cypher",
                schema_path=SchemaPath(
                    start_label="Region",
                    end_label="Region",
                    hops=[("Region", "PROJECT_TO", "Region")],  # ✅ 3元组
                    score=1.0,
                    total_relationships=20000
                ),
                cypher_template="""
                MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                WHERE r.acronym IN $enriched_regions
                RETURN r.acronym AS source,
                       t.acronym AS target,
                       t.name AS target_name,
                       p.weight AS projection_weight,
                       p.neuron_count AS neuron_count
                ORDER BY p.weight DESC
                LIMIT 30
                """,
                parameters={'enriched_regions': []},
                depends_on=[2],
                modality='projection'
            ))

        return plans

    def _plan_region_analysis(self,
                              region_cluster: EntityCluster,
                              all_clusters: List[EntityCluster],
                              question: str) -> List[QueryPlan]:
        """区域分析的动态规划"""
        plans = []
        region_acronym = region_cluster.primary_entity.context.get('acronym',
                                                                   region_cluster.primary_entity.text)

        # Step 1: Basic region info
        plans.append(QueryPlan(
            step_number=1,
            purpose=f"Get basic information about {region_acronym}",
            action="execute_cypher",
            schema_path=SchemaPath(
                start_label="Region",
                end_label="Region",
                hops=[],
                score=1.0,
                total_relationships=0
            ),
            cypher_template="""
            MATCH (r:Region {acronym: $acronym})
            RETURN r.name AS full_name,
                   r.acronym AS acronym,
                   r.region_id AS region_id,
                   properties(r) AS all_properties
            """,
            parameters={'acronym': region_acronym},
            depends_on=[],
            modality=None
        ))

        # Step 2: Cell composition
        plans.append(QueryPlan(
            step_number=2,
            purpose=f"Get cell type composition of {region_acronym}",
            action="execute_cypher",
            schema_path=SchemaPath(
                start_label="Region",
                end_label="Cluster",
                hops=[("Region", "HAS_CLUSTER", "Cluster")],
                score=1.0,
                total_relationships=10000
            ),
            cypher_template="""
            MATCH (r:Region {acronym: $acronym})-[:HAS_CLUSTER]->(c:Cluster)
            RETURN c.name AS cluster,
                   c.markers AS markers,
                   c.number_of_neurons AS neurons
            ORDER BY c.number_of_neurons DESC
            LIMIT 20
            """,
            parameters={'acronym': region_acronym},
            depends_on=[1],
            modality='molecular'
        ))

        # Step 3: Morphology
        plans.append(QueryPlan(
            step_number=3,
            purpose=f"Analyze morphological features of {region_acronym}",
            action="execute_cypher",
            schema_path=SchemaPath(
                start_label="Region",
                end_label="Neuron",
                hops=[("Neuron", "LOCATE_AT", "Region")],
                score=0.9,
                total_relationships=5000
            ),
            cypher_template="""
            MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $acronym})
            RETURN count(n) AS neuron_count,
                   avg(n.axonal_length) AS avg_axon_length,
                   avg(n.dendritic_length) AS avg_dendrite_length,
                   avg(n.soma_surface) AS avg_soma_surface
            """,
            parameters={'acronym': region_acronym},
            depends_on=[1],
            modality='morphological'
        ))

        # Step 4: Projections
        plans.append(QueryPlan(
            step_number=4,
            purpose=f"Identify projection targets of {region_acronym}",
            action="execute_cypher",
            schema_path=SchemaPath(
                start_label="Region",
                end_label="Region",
                hops=[("Region", "PROJECT_TO", "Region")],
                score=1.0,
                total_relationships=20000
            ),
            cypher_template="""
            MATCH (r:Region {acronym: $acronym})-[p:PROJECT_TO]->(t:Region)
            RETURN t.acronym AS target,
                   t.name AS target_name,
                   p.weight AS weight,
                   p.neuron_count AS neuron_count
            ORDER BY p.weight DESC
            LIMIT 20
            """,
            parameters={'acronym': region_acronym},
            depends_on=[1],
            modality='projection'
        ))

        return plans

    def _plan_celltype_analysis(self,
                                celltype_cluster: EntityCluster,
                                all_clusters: List[EntityCluster],
                                question: str) -> List[QueryPlan]:
        """细胞类型分析"""
        plans = []
        cell_type = celltype_cluster.primary_entity.text

        # Step 1: Find cell type nodes
        plans.append(QueryPlan(
            step_number=1,
            purpose=f"Find information about {cell_type}",
            action="execute_cypher",
            schema_path=SchemaPath(
                start_label="Subclass",
                end_label="Subclass",
                hops=[],
                score=1.0,
                total_relationships=0
            ),
            cypher_template="""
            MATCH (s:Subclass)
            WHERE s.name CONTAINS $cell_type
            RETURN s.name AS name,
                   properties(s) AS properties
            LIMIT 10
            """,
            parameters={'cell_type': cell_type},
            depends_on=[],
            modality='molecular'
        ))

        return plans

    def _generate_exploratory_plan(self) -> List[QueryPlan]:
            """探索性查询计划 (无法识别实体时)"""
            return [
                QueryPlan(
                    step_number=1,
                    purpose="Exploratory database overview",
                    action="execute_cypher",
                    schema_path=SchemaPath(
                        start_label="*",
                        end_label="*",
                        hops=[],
                        score=0.5,
                        total_relationships=0
                    ),
                    cypher_template="""
                    MATCH (n)
                    WITH labels(n)[0] AS node_type, count(n) AS count
                    RETURN node_type, count
                    ORDER BY count DESC
                    LIMIT 10
                    """,
                    parameters={},
                    depends_on=[],
                    modality=None
                )
            ]

    # ==================== Test ====================

if __name__ == "__main__":
        from aipom_cot_true_agent_v2 import RealSchemaCache

        schema = RealSchemaCache("./schema_output/schema.json")

        # Test schema graph
        graph = SchemaGraph(schema)

        print("\n=== Schema Graph Test ===")
        print(f"Region neighbors: {graph.get_neighbors('Region')[:3]}")

        # Test path finding
        finder = SchemaPathFinder(graph)

        paths = finder.find_paths('Region', 'Cluster', max_hops=2)

        print(f"\n=== Path Finding Test ===")
        print(f"Found {len(paths)} paths from Region to Cluster")

        for i, path in enumerate(paths[:3], 1):
            print(f"\nPath {i} (score: {path.score:.3f}):")
            for source, rel, target, count in path.hops:
                print(f"  {source} -[{rel}]-> {target} ({count:,} rels)")

        # Test dynamic planner
        planner = DynamicSchemaPathPlanner(schema)

        # Mock entity cluster
        from intelligent_entity_recognition import EntityMatch, EntityCluster

        gene_entity = EntityMatch(
            text="Car3",
            entity_id="Car3",
            entity_type="GeneMarker",
            match_type="exact",
            confidence=0.95,
            context={}
        )

        cluster = EntityCluster(
            primary_entity=gene_entity,
            related_entities=[],
            cluster_type='gene_marker',
            relevance_score=0.9
        )

        plans = planner.generate_plan([cluster], "Tell me about Car3+ neurons")

        print(f"\n=== Dynamic Planning Test ===")
        print(f"Generated {len(plans)} query plans")

        for plan in plans:
            print(f"\nStep {plan.step_number}: {plan.purpose}")
            print(f"  Modality: {plan.modality}")
            print(f"  Path score: {plan.schema_path.score:.3f}")