import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from schema_cache import SchemaCache


@dataclass
class Entity:
    """识别出的实体"""
    text: str
    entity_type: str  # 'gene', 'region', 'cell_type', 'feature'
    label: str  # Neo4j label: 'Subclass', 'Region', etc.
    confidence: float


@dataclass
class SchemaPath:
    """Schema中的路径"""
    start_label: str
    end_label: str
    path: List[Tuple[str, str]]  # [(rel_type, node_label), ...]
    score: float  # 路径相关性分数


class EntityRecognizer:
    """从问题中识别实体"""

    # 实体模式库
    PATTERNS = {
        'gene': {
            'keywords': ['gene', 'marker', 'expression', 'protein'],
            'suffixes': ['+', 'positive', 'expressing'],
            'examples': ['Car3', 'Pvalb', 'Sst', 'Vip', 'Gad1']
        },
        'region': {
            'keywords': ['region', 'area', 'cortex', 'nucleus'],
            'acronyms': ['CLA', 'MOs', 'MOp', 'ACAd', 'SSp'],
            'patterns': [r'\b[A-Z]{2,5}\b']  # 2-5大写字母
        },
        'cell_type': {
            'keywords': ['neuron', 'cell', 'interneuron'],
            'types': ['IT', 'ET', 'CT', 'PT', 'NP', 'excitatory', 'inhibitory']
        },
        'feature': {
            'morphology': ['axon', 'dendrite', 'branch', 'length'],
            'projection': ['project', 'target', 'connect', 'innervat'],
            'molecular': ['composition', 'profile', 'distribution']
        }
    }

    def __init__(self, schema: SchemaCache):
        self.schema = schema

        # 从schema构建实体词典
        self.known_genes = self._load_known_genes()
        self.known_regions = self._load_known_regions()

    def _load_known_genes(self) -> Set[str]:
        """从schema加载已知基因列表"""
        # 这里可以从Subclass nodes的name属性提取
        # 简化版本：使用常见基因列表
        return {
            'Car3', 'Pvalb', 'Sst', 'Vip', 'Gad1', 'Gad2',
            'Slc17a7', 'Rorb', 'Fezf2', 'Foxp2', 'Cux2'
        }

    def _load_known_regions(self) -> Set[str]:
        """从schema加载已知区域"""
        # 从schema的Region label属性中提取
        return {
            'CLA', 'MOs', 'MOp', 'ACAd', 'SSp', 'ENTl',
            'TH', 'HY', 'CP', 'STR', 'AI', 'PIR'
        }

    def recognize(self, question: str) -> List[Entity]:
        """识别问题中的所有实体"""
        entities = []

        # 1. 识别基因
        gene_entities = self._recognize_genes(question)
        entities.extend(gene_entities)

        # 2. 识别脑区
        region_entities = self._recognize_regions(question)
        entities.extend(region_entities)

        # 3. 识别细胞类型
        celltype_entities = self._recognize_celltypes(question)
        entities.extend(celltype_entities)

        # 4. 识别特征类型
        feature_entities = self._recognize_features(question)
        entities.extend(feature_entities)

        return entities

    def _recognize_genes(self, text: str) -> List[Entity]:
        """识别基因"""
        entities = []

        # 检查已知基因
        for gene in self.known_genes:
            if gene in text or gene.lower() in text.lower():
                # 检查是否有"+"后缀
                confidence = 0.9 if (gene + '+') in text or (gene + ' positive') in text else 0.7

                entities.append(Entity(
                    text=gene,
                    entity_type='gene',
                    label='Subclass',  # 基因通过Subclass关联
                    confidence=confidence
                ))

        # 检查基因关键词模式
        if re.search(r'\b(\w+)\+\s*neuron', text, re.IGNORECASE):
            match = re.search(r'\b(\w+)\+', text)
            if match:
                gene = match.group(1)
                entities.append(Entity(
                    text=gene,
                    entity_type='gene',
                    label='Subclass',
                    confidence=0.8
                ))

        return entities

    def _recognize_regions(self, text: str) -> List[Entity]:
        """识别脑区"""
        entities = []

        for region in self.known_regions:
            if re.search(rf'\b{region}\b', text, re.IGNORECASE):
                entities.append(Entity(
                    text=region,
                    entity_type='region',
                    label='Region',
                    confidence=0.95
                ))

        return entities

    def _recognize_celltypes(self, text: str) -> List[Entity]:
        """识别细胞类型"""
        entities = []
        text_lower = text.lower()

        for cell_type in self.PATTERNS['cell_type']['types']:
            if cell_type.lower() in text_lower:
                entities.append(Entity(
                    text=cell_type,
                    entity_type='cell_type',
                    label='Subclass',
                    confidence=0.8
                ))

        return entities

    def _recognize_features(self, text: str) -> List[Entity]:
        """识别特征类型"""
        entities = []
        text_lower = text.lower()

        # 形态学特征
        for keyword in self.PATTERNS['feature']['morphology']:
            if keyword in text_lower:
                entities.append(Entity(
                    text=keyword,
                    entity_type='morphology',
                    label='Region',  # 形态学属性在Region上
                    confidence=0.7
                ))

        # 投射特征
        for keyword in self.PATTERNS['feature']['projection']:
            if keyword in text_lower:
                entities.append(Entity(
                    text=keyword,
                    entity_type='projection',
                    label='PROJECT_TO',
                    confidence=0.7
                ))

        # 分子特征
        for keyword in self.PATTERNS['feature']['molecular']:
            if keyword in text_lower:
                entities.append(Entity(
                    text=keyword,
                    entity_type='molecular',
                    label='HAS_SUBCLASS',
                    confidence=0.7
                ))

        return entities


class SchemaPathFinder:
    """在Schema中寻找推理路径"""

    def __init__(self, schema: SchemaCache):
        self.schema = schema

        # 构建schema图
        self.graph = self._build_schema_graph()

    def _build_schema_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """构建schema的图表示"""
        graph = defaultdict(list)

        # 从rel_types构建
        for rel_type, spec in self.schema.rel_types.items():
            start_labels = spec['start']
            end_labels = spec['end']

            for start in start_labels:
                for end in end_labels:
                    graph[start].append((rel_type, end))

        return graph

    def find_paths(self,
                   start_label: str,
                   end_label: str,
                   max_depth: int = 3) -> List[SchemaPath]:
        """找到从start到end的所有路径"""

        paths = []

        # BFS搜索
        queue = [([start_label], [])]  # (node_path, edge_path)
        visited = set()

        while queue:
            node_path, edge_path = queue.pop(0)
            current = node_path[-1]

            if len(node_path) > max_depth:
                continue

            if current == end_label and len(edge_path) > 0:
                # 找到一条路径
                score = self._score_path(edge_path)
                paths.append(SchemaPath(
                    start_label=start_label,
                    end_label=end_label,
                    path=edge_path,
                    score=score
                ))
                continue

            state = (current, tuple(edge_path))
            if state in visited:
                continue
            visited.add(state)

            # 扩展邻居
            for rel_type, next_node in self.graph.get(current, []):
                new_node_path = node_path + [next_node]
                new_edge_path = edge_path + [(rel_type, next_node)]
                queue.append((new_node_path, new_edge_path))

        # 按分数排序
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _score_path(self, path: List[Tuple[str, str]]) -> float:
        """评估路径的相关性分数"""
        score = 1.0

        # 惩罚长路径
        score *= 0.8 ** (len(path) - 1)

        # 奖励常用关系
        common_rels = {'HAS_SUBCLASS', 'PROJECT_TO', 'LOCATE_AT'}
        for rel, _ in path:
            if rel in common_rels:
                score *= 1.2

        return score


class SchemaGuidedCoTGenerator:
    """
    核心：基于Schema自动生成Chain-of-Thought
    """

    def __init__(self, schema: SchemaCache):
        self.schema = schema
        self.entity_recognizer = EntityRecognizer(schema)
        self.path_finder = SchemaPathFinder(schema)

    def generate_cot(self, question: str) -> Dict[str, any]:
        """
        从问题生成完整的推理链

        Example:
            Input: "Tell me something about Car3+ neurons"

            Output: {
                'entities': [Entity('Car3', 'gene', ...)],
                'reasoning_chain': [
                    {
                        'step': 1,
                        'purpose': 'Identify Car3 as a gene marker',
                        'action': 'entity_recognition',
                        'details': 'Car3 is a subclass marker gene'
                    },
                    {
                        'step': 2,
                        'purpose': 'Find regions enriched for Car3',
                        'action': 'follow_relationship',
                        'path': ['Region', 'HAS_SUBCLASS', 'Subclass'],
                        'query_template': 'region_by_gene'
                    },
                    {
                        'step': 3,
                        'purpose': 'Analyze morphological features of Car3+ regions',
                        'action': 'extract_properties',
                        'properties': ['axonal_length', 'dendritic_length', ...]
                    },
                    ...
                ]
            }
        """

        # 1. 实体识别
        entities = self.entity_recognizer.recognize(question)

        if not entities:
            # 无法识别实体，返回通用探索
            return self._generate_exploratory_cot(question)

        # 2. 确定主要实体
        primary_entity = self._select_primary_entity(entities)

        # 3. 基于实体类型生成推理链
        if primary_entity.entity_type == 'gene':
            return self._generate_gene_centric_cot(primary_entity, entities, question)
        elif primary_entity.entity_type == 'region':
            return self._generate_region_centric_cot(primary_entity, entities, question)
        else:
            return self._generate_generic_cot(primary_entity, entities, question)

    def _select_primary_entity(self, entities: List[Entity]) -> Entity:
        """选择主要实体"""
        # 优先级：gene > region > cell_type > feature
        priority = {'gene': 4, 'region': 3, 'cell_type': 2}

        entities.sort(key=lambda e: (
            priority.get(e.entity_type, 1),
            e.confidence
        ), reverse=True)

        return entities[0]

    def _generate_gene_centric_cot(self,
                                   gene_entity: Entity,
                                   all_entities: List[Entity],
                                   question: str) -> Dict:
        """
        基因为中心的推理链（Figure 3的核心逻辑）
        """
        gene_name = gene_entity.text

        reasoning_chain = []

        # Step 1: 识别基因
        reasoning_chain.append({
            'step': 1,
            'purpose': f'Identify {gene_name} as a gene marker',
            'action': 'entity_recognition',
            'rationale': f'{gene_name} is recognized as a subclass marker gene in the schema',
            'modality': 'molecular',
            'query_template': None
        })

        # Step 2: 找富集区域
        reasoning_chain.append({
            'step': 2,
            'purpose': f'Find brain regions enriched for {gene_name} expression',
            'action': 'follow_relationship',
            'rationale': 'Schema path: Region -[HAS_SUBCLASS]-> Subclass',
            'modality': 'molecular',
            'schema_path': [
                ('Region', 'HAS_SUBCLASS', 'Subclass')
            ],
            'query_template': 'region_by_gene',
            'query': f"""
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.name CONTAINS '{gene_name}'
RETURN r.acronym AS region, 
       r.region_id AS region_id,
       avg(h.pct_cells) AS enrichment,
       count(*) AS cell_count
ORDER BY enrichment DESC
LIMIT 10
            """.strip()
        })

        # Step 3: 分析这些区域的特征
        # 检查问题中是否提到形态学
        has_morphology = any(e.entity_type == 'morphology' for e in all_entities)

        if has_morphology or 'morphology' in question.lower() or 'feature' in question.lower():
            reasoning_chain.append({
                'step': 3,
                'purpose': f'Analyze morphological features of {gene_name}+ enriched regions',
                'action': 'extract_properties',
                'rationale': 'Schema: Region node has morphological properties',
                'modality': 'morphological',
                'properties': [
                    'axonal_length', 'dendritic_length',
                    'axonal_branches', 'dendritic_branches',
                    'axonal_maximum_branch_order', 'dendritic_maximum_branch_order'
                ],
                'depends_on': [2],
                'query': """
MATCH (r:Region)
WHERE r.acronym IN $enriched_regions
RETURN r.acronym AS region,
       r.axonal_length AS axonal_length,
       r.dendritic_length AS dendritic_length,
       r.axonal_branches AS axonal_branches,
       r.dendritic_branches AS dendritic_branches
LIMIT 20
                """.strip()
            })

        # Step 4: 投射模式
        has_projection = any(e.entity_type == 'projection' for e in all_entities)

        if has_projection or 'project' in question.lower() or 'target' in question.lower():
            step_num = len(reasoning_chain) + 1
            reasoning_chain.append({
                'step': step_num,
                'purpose': f'Identify projection targets of {gene_name}+ regions',
                'action': 'follow_relationship',
                'rationale': 'Schema path: Region -[PROJECT_TO]-> Region',
                'modality': 'projection',
                'schema_path': [
                    ('Region', 'PROJECT_TO', 'Region')
                ],
                'depends_on': [2],
                'query': """
MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
WHERE r.acronym IN $enriched_regions
RETURN r.acronym AS source,
       t.acronym AS target,
       p.weight AS weight,
       p.neuron_count AS neuron_count
ORDER BY weight DESC
LIMIT 30
                """.strip()
            })

            # Step 5: 目标区域的分子特征
            step_num = len(reasoning_chain) + 1
            reasoning_chain.append({
                'step': step_num,
                'purpose': f'Analyze molecular profiles of projection targets',
                'action': 'follow_relationship',
                'rationale': 'Schema path: Region -[HAS_SUBCLASS]-> Subclass for target regions',
                'modality': 'molecular',
                'schema_path': [
                    ('Region', 'HAS_SUBCLASS', 'Subclass')
                ],
                'depends_on': [step_num - 1],
                'query': """
MATCH (t:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE t.acronym IN $target_regions
RETURN t.acronym AS target,
       s.name AS subclass,
       h.pct_cells AS percentage,
       h.rank AS rank
ORDER BY t.acronym, h.rank
LIMIT 50
                """.strip()
            })

        return {
            'primary_entity': {
                'text': gene_name,
                'type': 'gene',
                'confidence': gene_entity.confidence
            },
            'entities': [
                {'text': e.text, 'type': e.entity_type, 'confidence': e.confidence}
                for e in all_entities
            ],
            'reasoning_chain': reasoning_chain,
            'complexity': 'multihop' if len(reasoning_chain) >= 4 else 'pattern',
            'expected_modalities': self._extract_modalities(reasoning_chain)
        }

    def _generate_region_centric_cot(self,
                                     region_entity: Entity,
                                     all_entities: List[Entity],
                                     question: str) -> Dict:
        """区域为中心的推理链"""
        region_name = region_entity.text

        reasoning_chain = []

        # Step 1: 区域识别
        reasoning_chain.append({
            'step': 1,
            'purpose': f'Identify {region_name} as a brain region',
            'action': 'entity_recognition',
            'rationale': f'{region_name} is recognized as a Region in the schema'
        })

        # Step 2: 分子特征
        reasoning_chain.append({
            'step': 2,
            'purpose': f'Get cell type composition of {region_name}',
            'action': 'follow_relationship',
            'modality': 'molecular',
            'schema_path': [('Region', 'HAS_SUBCLASS', 'Subclass')],
            'query': f"""
MATCH (r:Region {{acronym: '{region_name}'}})-[h:HAS_SUBCLASS]->(s:Subclass)
RETURN s.name AS subclass,
       h.pct_cells AS percentage,
       h.rank AS rank
ORDER BY h.rank
LIMIT 20
            """.strip()
        })

        # Step 3: 形态学
        reasoning_chain.append({
            'step': 3,
            'purpose': f'Analyze morphological features of {region_name}',
            'action': 'extract_properties',
            'modality': 'morphological',
            'query': f"""
MATCH (r:Region {{acronym: '{region_name}'}})
RETURN r.axonal_length, r.dendritic_length,
       r.axonal_branches, r.dendritic_branches
            """.strip()
        })

        # Step 4: 投射
        reasoning_chain.append({
            'step': 4,
            'purpose': f'Identify projection targets of {region_name}',
            'action': 'follow_relationship',
            'modality': 'projection',
            'schema_path': [('Region', 'PROJECT_TO', 'Region')],
            'query': f"""
MATCH (r:Region {{acronym: '{region_name}'}})-[p:PROJECT_TO]->(t:Region)
RETURN t.acronym AS target,
       p.weight AS weight,
       p.neuron_count AS neuron_count
ORDER BY weight DESC
LIMIT 20
            """.strip()
        })

        return {
            'primary_entity': {
                'text': region_name,
                'type': 'region',
                'confidence': region_entity.confidence
            },
            'entities': [
                {'text': e.text, 'type': e.entity_type, 'confidence': e.confidence}
                for e in all_entities
            ],
            'reasoning_chain': reasoning_chain,
            'complexity': 'pattern',
            'expected_modalities': self._extract_modalities(reasoning_chain)
        }

    def _generate_generic_cot(self,
                              primary_entity: Entity,
                              all_entities: List[Entity],
                              question: str) -> Dict:
        """通用推理链"""
        return {
            'primary_entity': {
                'text': primary_entity.text,
                'type': primary_entity.entity_type,
                'confidence': primary_entity.confidence
            },
            'entities': [
                {'text': e.text, 'type': e.entity_type}
                for e in all_entities
            ],
            'reasoning_chain': [
                {
                    'step': 1,
                    'purpose': 'Exploratory search',
                    'action': 'general_query',
                    'query': 'MATCH (n) RETURN n LIMIT 50'
                }
            ],
            'complexity': 'simple'
        }

    def _generate_exploratory_cot(self, question: str) -> Dict:
        """无法识别实体时的探索性推理"""
        return {
            'primary_entity': None,
            'entities': [],
            'reasoning_chain': [
                {
                    'step': 1,
                    'purpose': 'Exploratory data retrieval',
                    'action': 'general_query',
                    'rationale': 'No specific entities recognized, performing broad search'
                }
            ],
            'complexity': 'simple'
        }

    def _extract_modalities(self, reasoning_chain: List[Dict]) -> List[str]:
        """提取涉及的模态"""
        modalities = set()
        for step in reasoning_chain:
            if 'modality' in step:
                modalities.add(step['modality'])
        return list(modalities)