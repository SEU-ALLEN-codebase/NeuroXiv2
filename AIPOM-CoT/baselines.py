import time
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== Abstract Base Class ====================

class BaselineAgent(ABC):
    """Baseline抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def answer(self, question: str, timeout: int = 120, **kwargs) -> Dict[str, Any]:
        """回答问题"""
        pass


# ==================== Baseline 1: Direct GPT-4o ====================

class DirectGPT4oBaseline(BaselineAgent):
    """
    Direct GPT-4o Baseline

    特点：
    - 使用GPT-4o模型
    - 无KG访问
    - 纯粹依赖预训练知识
    - 单次推理

    优势：
    - 强大的语言理解和推理能力
    - 速度快
    - 对概念性问题表现好

    劣势：
    - 无法访问最新/专有数据
    - 可能产生幻觉
    - 无系统分析能力
    """

    def __init__(self, openai_client):
        super().__init__("Direct GPT-4o")
        self.client = openai_client
        self.model = "gpt-4o"

    def answer(self, question: str, timeout: int = 120, **kwargs) -> Dict[str, Any]:
        """使用GPT-4o直接回答"""
        start_time = time.time()

        system_prompt = """You are an expert neuroscientist with deep knowledge of:
- Brain anatomy and neuroanatomy (Allen Mouse Brain Atlas)
- Cell types and molecular markers (Pvalb, Sst, VIP, Car3, etc.)
- Neuronal morphology and electrophysiology
- Brain connectivity and neural circuits
- Mouse brain regions and their functions

Provide scientifically accurate, detailed answers based on your knowledge.
Include specific quantitative data when possible (neuron counts, connectivity strengths, etc.).
If you're uncertain about specific details, acknowledge it rather than speculate."""

        user_prompt = f"""Question about neuroscience:

{question}

Please provide a comprehensive, scientifically rigorous answer that includes:
1. Direct answer to the question
2. Relevant molecular markers or cell types (if applicable)
3. Brain regions involved (if applicable)
4. Quantitative data when available
5. Key scientific context

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                timeout=timeout
            )

            answer = response.choices[0].message.content
            execution_time = time.time() - start_time

            entities_recognized = self._extract_entities_heuristic(answer)

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': [{
                    'purpose': 'Direct GPT-4o inference',
                    'modality': None,
                }],
                'schema_paths_used': [],
                'execution_time': execution_time,
                'total_steps': 1,
                'confidence_score': 0.75,
                'success': True,
                'method': 'Direct GPT-4o',
            }

        except Exception as e:
            logger.error(f"Direct GPT-4o failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_entities_heuristic(self, answer: str) -> List[Dict]:
        """启发式提取实体"""
        entities = []

        regions = re.findall(r'\b[A-Z]{2,5}\b', answer)
        for r in set(regions):
            if r not in ['DNA', 'RNA', 'ATP', 'GABA', 'LLM', 'GPT', 'USA', 'PHD']:
                entities.append({
                    'text': r,
                    'type': 'Region',
                    'confidence': 0.7,
                })

        genes = re.findall(r'\b[A-Z][a-z]{2,8}\+?\b', answer)
        for g in set(genes):
            if g not in ['The', 'This', 'That', 'There', 'These', 'Their', 'When', 'Where', 'Which']:
                entities.append({
                    'text': g.rstrip('+'),
                    'type': 'Gene',
                    'confidence': 0.6,
                })

        return entities[:15]

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'Direct GPT-4o',
            'error': error,
        }


# ==================== Baseline 2: Template-KG (Enhanced) ====================

class TemplateKGBaseline(BaselineAgent):
    """
    Template-based Knowledge Graph Query Baseline (Enhanced)

    🔧 v3.0改进：
    - 扩展到27个模板（覆盖多跳、比较、统计、闭环）
    - 智能模板选择
    - 支持复杂查询序列
    """

    def __init__(self, neo4j_exec, openai_client):
        super().__init__("Template-KG")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = "gpt-4o"
        self.templates = self._build_templates()

    def _build_templates(self) -> Dict:
        """🔧 扩展模板库到27个"""
        return {
            # ========== 基础查询 (5个) ==========

            'gene_to_clusters': """
                MATCH (c:Cluster)
                WHERE c.markers CONTAINS $gene
                RETURN c.name AS cluster, 
                       c.number_of_neurons AS neurons,
                       c.broad_region_distribution AS regions,
                       c.markers AS markers
                ORDER BY c.number_of_neurons DESC
                LIMIT 20
            """,

            'region_to_clusters': """
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE r.acronym = $region
                RETURN r.name AS region_name, 
                       c.name AS cluster,
                       c.markers AS markers, 
                       c.number_of_neurons AS neurons
                ORDER BY c.number_of_neurons DESC
                LIMIT 30
            """,

            'region_projections': """
                MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                WHERE r.acronym = $region
                RETURN r.name AS source, 
                       t.acronym AS target, 
                       t.name AS target_name,
                       p.weight AS weight,
                       p.neuron_count AS neuron_count
                ORDER BY p.weight DESC
                LIMIT 20
            """,

            'region_morphology': """
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                WHERE r.acronym = $region
                RETURN r.name AS region,
                       count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon_length,
                       avg(n.dendritic_length) AS avg_dendrite_length,
                       avg(n.axonal_branches) AS avg_axon_branches,
                       avg(n.dendritic_branches) AS avg_dendrite_branches
            """,

            'gene_to_regions': """
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE c.markers CONTAINS $gene
                WITH r, count(c) AS cluster_count, sum(c.number_of_neurons) AS total_neurons
                RETURN r.acronym AS region,
                       r.name AS region_name,
                       cluster_count,
                       total_neurons
                ORDER BY total_neurons DESC
                LIMIT 15
            """,

            # ========== 多跳查询 (5个) ==========

            'gene_to_projection_targets': """
                MATCH (c:Cluster)-[:LOCATE_AT]->(r:Region)
                WHERE c.markers CONTAINS $gene
                WITH r, sum(c.number_of_neurons) AS source_neurons
                MATCH (r)-[p:PROJECT_TO]->(t:Region)
                RETURN r.acronym AS source_region,
                       r.name AS source_name,
                       t.acronym AS target_region,
                       t.name AS target_name,
                       p.weight AS projection_weight,
                       source_neurons
                ORDER BY p.weight DESC
                LIMIT 15
            """,

            'region_to_target_composition': """
                MATCH (r:Region {acronym: $region})-[:PROJECT_TO]->(t:Region)
                WITH t, r
                ORDER BY t.acronym
                MATCH (t)-[:HAS_CLUSTER]->(c:Cluster)
                WITH t, collect(c.name)[..5] AS clusters, collect(c.markers)[..5] AS markers
                RETURN t.acronym AS target,
                       t.name AS target_name,
                       clusters,
                       markers
                LIMIT 10
            """,

            'cluster_projection_pattern': """
                MATCH (c:Cluster)-[:LOCATE_AT]->(r:Region)
                WHERE c.name CONTAINS $cluster_pattern OR c.markers CONTAINS $cluster_pattern
                WITH c, r LIMIT 5
                MATCH (r)-[p:PROJECT_TO]->(t:Region)
                RETURN c.name AS cluster,
                       r.acronym AS source,
                       t.acronym AS target,
                       p.weight AS weight
                ORDER BY p.weight DESC
                LIMIT 15
            """,

            'subclass_spatial_distribution': """
                MATCH (s:Subclass)-[:HAS_CLUSTER]->(c:Cluster)-[:LOCATE_AT]->(r:Region)
                WHERE s.name CONTAINS $subclass OR c.markers CONTAINS $subclass
                WITH r, count(c) AS cluster_count, sum(c.number_of_neurons) AS total_neurons
                RETURN r.acronym AS region,
                       r.name AS region_name,
                       cluster_count,
                       total_neurons
                ORDER BY total_neurons DESC
                LIMIT 15
            """,

            'multi_hop_circuit': """
                MATCH path = (r1:Region {acronym: $region})-[:PROJECT_TO*1..2]->(r2:Region)
                WITH r2, length(path) AS hop_count
                RETURN DISTINCT r2.acronym AS target,
                       r2.name AS target_name,
                       hop_count
                ORDER BY hop_count, r2.acronym
                LIMIT 20
            """,

            # ========== 比较分析 (5个) ==========

            'compare_regions_cell_diversity': """
                MATCH (r1:Region {acronym: $region1})-[:HAS_CLUSTER]->(c1:Cluster)
                WITH count(DISTINCT c1) AS diversity1, sum(c1.number_of_neurons) AS neurons1
                MATCH (r2:Region {acronym: $region2})-[:HAS_CLUSTER]->(c2:Cluster)
                WITH diversity1, neurons1, count(DISTINCT c2) AS diversity2, sum(c2.number_of_neurons) AS neurons2
                RETURN diversity1, neurons1, diversity2, neurons2
            """,

            'compare_regions_projections': """
                MATCH (r1:Region {acronym: $region1})-[p1:PROJECT_TO]->(t:Region)
                WITH count(p1) AS proj_count1, avg(p1.weight) AS avg_weight1, collect(t.acronym) AS targets1
                MATCH (r2:Region {acronym: $region2})-[p2:PROJECT_TO]->(t2:Region)
                WITH proj_count1, avg_weight1, targets1, 
                     count(p2) AS proj_count2, avg(p2.weight) AS avg_weight2, collect(t2.acronym) AS targets2
                RETURN proj_count1, avg_weight1, targets1[..10] AS sample_targets1,
                       proj_count2, avg_weight2, targets2[..10] AS sample_targets2
            """,

            'compare_genes_expression_breadth': """
                MATCH (c1:Cluster)
                WHERE c1.markers CONTAINS $gene1
                WITH count(c1) AS clusters1, sum(c1.number_of_neurons) AS neurons1
                MATCH (c2:Cluster)
                WHERE c2.markers CONTAINS $gene2
                WITH clusters1, neurons1, count(c2) AS clusters2, sum(c2.number_of_neurons) AS neurons2
                RETURN clusters1, neurons1, clusters2, neurons2
            """,

            'morphology_comparison': """
                MATCH (n1:Neuron)-[:LOCATE_AT]->(r1:Region {acronym: $region1})
                WITH avg(n1.axonal_length) AS axon1, 
                     avg(n1.dendritic_length) AS dend1,
                     avg(n1.axonal_branches) AS axon_br1,
                     avg(n1.dendritic_branches) AS dend_br1,
                     count(n1) AS count1
                MATCH (n2:Neuron)-[:LOCATE_AT]->(r2:Region {acronym: $region2})
                RETURN axon1, dend1, axon_br1, dend_br1, count1,
                       avg(n2.axonal_length) AS axon2,
                       avg(n2.dendritic_length) AS dend2,
                       avg(n2.axonal_branches) AS axon_br2,
                       avg(n2.dendritic_branches) AS dend_br2,
                       count(n2) AS count2
            """,

            'connectivity_overlap_analysis': """
                MATCH (r1:Region {acronym: $region1})-[:PROJECT_TO]->(t:Region)
                WITH collect(t.acronym) AS targets1
                MATCH (r2:Region {acronym: $region2})-[:PROJECT_TO]->(t2:Region)
                WITH targets1, collect(t2.acronym) AS targets2
                RETURN targets1[..10] AS sample_targets1,
                       targets2[..10] AS sample_targets2,
                       [x IN targets1 WHERE x IN targets2] AS shared_targets
            """,

            # ========== 统计分析 (5个) ==========

            'region_diversity_metrics': """
                MATCH (r:Region {acronym: $region})-[:HAS_CLUSTER]->(c:Cluster)
                WITH count(c) AS num_clusters,
                     sum(c.number_of_neurons) AS total_neurons,
                     collect(c.number_of_neurons) AS neuron_distribution,
                     collect(c.markers) AS all_markers
                RETURN num_clusters,
                       total_neurons,
                       neuron_distribution[..10] AS sample_distribution,
                       size(all_markers) AS marker_diversity
            """,

            'gene_enrichment_analysis': """
                MATCH (c:Cluster)
                WHERE c.markers CONTAINS $gene
                WITH sum(c.number_of_neurons) AS expressing_neurons, 
                     count(c) AS expressing_clusters
                MATCH (all:Cluster)
                WITH expressing_neurons, expressing_clusters,
                     sum(all.number_of_neurons) AS total_neurons,
                     count(all) AS total_clusters
                RETURN expressing_neurons,
                       total_neurons,
                       toFloat(expressing_neurons) / total_neurons AS enrichment_ratio,
                       expressing_clusters,
                       total_clusters
            """,

            'projection_strength_statistics': """
                MATCH (r:Region {acronym: $region})-[p:PROJECT_TO]->(t:Region)
                WITH collect(p.weight) AS weights, collect(t.acronym) AS targets
                RETURN weights,
                       targets,
                       size(weights) AS num_projections,
                       reduce(s = 0.0, w IN weights | s + w) / size(weights) AS mean_weight
            """,

            'cell_type_hierarchy': """
                MATCH (s:Subclass)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE s.name CONTAINS $pattern OR c.markers CONTAINS $pattern
                WITH s, count(c) AS num_clusters, 
                     sum(c.number_of_neurons) AS total_neurons,
                     collect(c.name)[..8] AS cluster_samples
                RETURN s.name AS subclass,
                       num_clusters,
                       total_neurons,
                       cluster_samples
                ORDER BY total_neurons DESC
                LIMIT 10
            """,

            'regional_molecular_signature': """
                MATCH (r:Region {acronym: $region})-[:HAS_CLUSTER]->(c:Cluster)
                WITH collect(DISTINCT c.markers) AS all_marker_sets
                UNWIND all_marker_sets AS marker_set
                WITH split(marker_set, ',') AS markers
                UNWIND markers AS marker
                WITH trim(marker) AS clean_marker
                WHERE clean_marker <> ''
                RETURN clean_marker AS marker, count(*) AS frequency
                ORDER BY frequency DESC
                LIMIT 20
            """,

            # ========== 闭环分析 (7个) ==========

            'closed_loop_gene_profiling': """
                MATCH (c:Cluster)-[:LOCATE_AT]->(r:Region)
                WHERE c.markers CONTAINS $gene
                WITH r, sum(c.number_of_neurons) AS source_neurons
                ORDER BY source_neurons DESC
                LIMIT 3
                MATCH (r)-[p:PROJECT_TO]->(t:Region)
                WITH r, t, p, source_neurons
                ORDER BY p.weight DESC
                LIMIT 5
                MATCH (t)-[:HAS_CLUSTER]->(tc:Cluster)
                WITH r, t, p, source_neurons,
                     collect(tc.markers)[..5] AS target_markers,
                     sum(tc.number_of_neurons) AS target_neurons
                RETURN r.acronym AS source_region,
                       r.name AS source_name,
                       source_neurons,
                       t.acronym AS target_region,
                       t.name AS target_name,
                       p.weight AS projection_weight,
                       target_markers,
                       target_neurons
            """,

            'comprehensive_region_profiling': """
                MATCH (r:Region {acronym: $region})
                OPTIONAL MATCH (r)-[:HAS_CLUSTER]->(c:Cluster)
                WITH r, count(c) AS num_clusters, sum(c.number_of_neurons) AS total_neurons
                OPTIONAL MATCH (r)-[p:PROJECT_TO]->(t:Region)
                WITH r, num_clusters, total_neurons, 
                     count(p) AS num_projections, collect(t.acronym)[..10] AS top_targets
                OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
                RETURN r.name AS region_name,
                       num_clusters,
                       total_neurons,
                       num_projections,
                       top_targets,
                       avg(n.axonal_length) AS avg_axon_length,
                       count(n) AS neuron_sample_size
            """,

            'pathway_decomposition': """
                MATCH path = (r1:Region {acronym: $region})-[:PROJECT_TO*1..2]->(r2:Region)
                WITH r2, length(path) AS path_length, path
                ORDER BY path_length, r2.acronym
                LIMIT 15
                MATCH (r2)-[:HAS_CLUSTER]->(c:Cluster)
                WITH r2, path_length, 
                     count(c) AS cell_type_count,
                     collect(c.markers)[..3] AS sample_markers
                RETURN r2.acronym AS destination,
                       r2.name AS destination_name,
                       path_length AS hops,
                       cell_type_count,
                       sample_markers
            """,

            'reciprocal_circuit_analysis': """
                MATCH (r1:Region {acronym: $region})-[p1:PROJECT_TO]->(r2:Region)
                MATCH (r2)-[p2:PROJECT_TO]->(r1)
                RETURN r2.acronym AS reciprocal_region,
                       r2.name AS region_name,
                       p1.weight AS forward_weight,
                       p2.weight AS backward_weight,
                       abs(p1.weight - p2.weight) AS asymmetry
                ORDER BY p1.weight DESC
                LIMIT 10
            """,

            'circuit_motif_search': """
                MATCH (r:Region {acronym: $region})-[:PROJECT_TO]->(t1:Region)
                MATCH (t1)-[:PROJECT_TO]->(t2:Region)
                WHERE t2 <> r
                WITH t1, t2, count(*) AS path_count
                ORDER BY path_count DESC
                LIMIT 10
                RETURN t1.acronym AS intermediate,
                       t1.name AS intermediate_name,
                       t2.acronym AS terminal,
                       t2.name AS terminal_name,
                       path_count
            """,

            'convergent_divergent_analysis': """
                MATCH (r:Region {acronym: $region})-[:PROJECT_TO]->(t:Region)
                WITH count(DISTINCT t) AS divergence
                MATCH (s:Region)-[:PROJECT_TO]->(r:Region {acronym: $region})
                WITH divergence, count(DISTINCT s) AS convergence
                MATCH (r:Region {acronym: $region})-[:HAS_CLUSTER]->(c:Cluster)
                RETURN divergence AS projection_divergence,
                       convergence AS input_convergence,
                       count(c) AS local_cell_type_diversity
            """,

            'multi_modal_consistency_check': """
                MATCH (r:Region {acronym: $region})-[:HAS_CLUSTER]->(c:Cluster)
                WITH r, count(c) AS molecular_diversity
                MATCH (r)-[:PROJECT_TO]->(t:Region)
                WITH r, molecular_diversity, count(t) AS projection_diversity
                OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
                RETURN molecular_diversity,
                       projection_diversity,
                       count(n) AS morphology_sample_size,
                       avg(n.axonal_length) AS avg_axon_length
            """,
        }

    def answer(self, question: str, timeout: int = 120, **kwargs) -> Dict[str, Any]:
        """使用模板回答问题（增强版）"""
        start_time = time.time()

        try:
            # Step 1: 智能分类
            question_type = self._classify_question_enhanced(question)
            logger.info(f"  Template-KG: Type='{question_type}'")

            # Step 2: 提取参数
            params = self._extract_parameters_enhanced(question)
            logger.info(f"  Template-KG: Params={params}")

            if not params:
                return self._fallback_answer(question, time.time() - start_time)

            # Step 3: 选择模板序列
            template_sequence = self._select_template_sequence(question_type, params)
            logger.info(f"  Template-KG: Selected {len(template_sequence)} templates")

            # Step 4: 执行模板序列
            results, executed_steps = self._execute_template_sequence(template_sequence, params)

            # Step 5: 合成答案
            if not results or not any(r.get('success') for r in results):
                return self._fallback_answer(question, time.time() - start_time)

            answer = self._synthesize_answer(question, results)

            execution_time = time.time() - start_time

            # 提取实体
            entities_recognized = []
            for key, value in params.items():
                if value and isinstance(value, str):
                    entities_recognized.append({
                        'text': value,
                        'type': 'Gene' if key.startswith('gene') else 'Region',
                        'confidence': 1.0,
                    })

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': executed_steps,
                'schema_paths_used': [s['template'] for s in executed_steps],
                'execution_time': execution_time,
                'total_steps': len(executed_steps),
                'confidence_score': 0.75,
                'success': True,
                'method': 'Template-KG',
            }

        except Exception as e:
            logger.error(f"Template-KG failed: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(question, str(e), time.time() - start_time)

    def _classify_question_enhanced(self, question: str) -> str:
        """🔧 增强的问题分类"""
        q_lower = question.lower()

        # Closed-loop patterns (highest priority)
        if any(kw in q_lower for kw in ['comprehensive', 'full', 'complete']) and \
           any(kw in q_lower for kw in ['analysis', 'profile', 'characterization']):
            if any(kw in q_lower for kw in ['+', 'positive', 'expressing']):
                return 'closed_loop_gene_profiling'
            else:
                return 'comprehensive_region_profiling'

        # Comparison patterns
        if any(kw in q_lower for kw in ['compare', 'versus', 'vs', 'difference', 'contrast']):
            if 'morphology' in q_lower or 'morpholog' in q_lower:
                return 'morphology_comparison'
            elif 'projection' in q_lower or 'connectivity' in q_lower:
                return 'compare_projections'
            else:
                return 'compare_cell_diversity'

        # Statistical/screening patterns
        if any(kw in q_lower for kw in ['diversity', 'enrichment', 'distribution', 'which', 'find all']):
            return 'statistical_analysis'

        # Multi-hop patterns
        if any(kw in q_lower for kw in ['target', 'circuit', 'pathway', 'downstream']):
            if any(kw in q_lower for kw in ['+', 'positive', 'gene', 'marker']):
                return 'gene_to_targets'
            else:
                return 'projection_analysis'

        # Profiling patterns
        if any(kw in q_lower for kw in ['tell me about', 'about', 'profile', 'characterize', 'describe']):
            if any(kw in q_lower for kw in ['+', 'neuron', 'cell', 'interneuron', 'positive']):
                return 'gene_profiling'
            else:
                return 'region_analysis'

        # Simple lookup
        return 'simple_lookup'

    def _extract_parameters_enhanced(self, question: str) -> Dict:
        """🔧 增强的参数提取"""
        params = {}

        # 提取基因名
        gene_patterns = [
            r'\b([A-Z][a-z]{2,8})\+',  # Pvalb+
            r'\b([A-Z][a-z]{2,8})-positive',  # Sst-positive
            r'\b([A-Z][a-z]{2,8})\s+expressing',  # Car3 expressing
        ]
        for pattern in gene_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                gene = matches[0] if isinstance(matches[0], str) else matches[0][0]
                if gene not in {'What', 'Which', 'Where', 'Tell', 'Give', 'Show', 'Find', 'The', 'This', 'That'}:
                    params['gene'] = gene
                    break

        # 提取脑区
        known_regions = {
            'MOp', 'MOs', 'SSp', 'SSs', 'VISp', 'VISal', 'VISam', 'VISl', 'VISpm',
            'AUDp', 'AUDpo', 'AUDv', 'ACA', 'PL', 'ILA', 'ORB',
            'RSP', 'CLA', 'HPF', 'HIP', 'TH', 'HY', 'CTX'
        }
        regions = re.findall(r'\b([A-Z]{2,5})\b', question)
        for r in regions:
            if r in known_regions:
                if 'region' not in params:
                    params['region'] = r
                elif 'region1' not in params:
                    params['region1'] = params['region']
                    params['region'] = r
                elif 'region2' not in params:
                    params['region2'] = r

        # 对于比较问题，提取两个实体
        if 'compare' in question.lower() or 'versus' in question.lower():
            words = question.split()
            for i, word in enumerate(words):
                if word.lower() in ['and', 'vs', 'versus', 'with']:
                    # 前后可能是实体
                    if i > 0 and i < len(words) - 1:
                        entity1 = words[i-1].strip(',').strip()
                        entity2 = words[i+1].strip(',').strip()

                        if entity1.isupper() and len(entity1) <= 5:
                            params['region1'] = entity1
                        elif entity1[0].isupper() and len(entity1) >= 3:
                            params['gene1'] = entity1.rstrip('+')

                        if entity2.isupper() and len(entity2) <= 5:
                            params['region2'] = entity2
                        elif entity2[0].isupper() and len(entity2) >= 3:
                            params['gene2'] = entity2.rstrip('+')

        # 提取模式/关键词（用于模糊匹配）
        if 'pattern' not in params and 'gene' not in params:
            # 提取可能的subclass/cluster关键词
            concepts = ['interneuron', 'pyramidal', 'excitatory', 'inhibitory', 'basket', 'chandelier']
            for concept in concepts:
                if concept in question.lower():
                    params['pattern'] = concept
                    break

        return params

    def _select_template_sequence(self, question_type: str, params: Dict) -> List[Tuple[str, Dict]]:
        """🔧 选择模板执行序列"""

        sequences = {
            'closed_loop_gene_profiling': [
                ('closed_loop_gene_profiling', {'gene': params.get('gene')}),
                ('gene_to_regions', {'gene': params.get('gene')}),
                ('gene_enrichment_analysis', {'gene': params.get('gene')}),
            ],

            'comprehensive_region_profiling': [
                ('comprehensive_region_profiling', {'region': params.get('region')}),
                ('region_diversity_metrics', {'region': params.get('region')}),
                ('convergent_divergent_analysis', {'region': params.get('region')}),
            ],

            'gene_profiling': [
                ('gene_to_clusters', {'gene': params.get('gene')}),
                ('gene_to_regions', {'gene': params.get('gene')}),
                ('gene_to_projection_targets', {'gene': params.get('gene')}),
            ],

            'region_analysis': [
                ('region_to_clusters', {'region': params.get('region')}),
                ('region_morphology', {'region': params.get('region')}),
                ('region_projections', {'region': params.get('region')}),
            ],

            'gene_to_targets': [
                ('gene_to_regions', {'gene': params.get('gene')}),
                ('gene_to_projection_targets', {'gene': params.get('gene')}),
            ],

            'projection_analysis': [
                ('region_projections', {'region': params.get('region')}),
                ('region_to_target_composition', {'region': params.get('region')}),
            ],

            'compare_cell_diversity': [
                ('compare_regions_cell_diversity', {
                    'region1': params.get('region1') or params.get('region'),
                    'region2': params.get('region2'),
                }),
            ],

            'compare_projections': [
                ('compare_regions_projections', {
                    'region1': params.get('region1') or params.get('region'),
                    'region2': params.get('region2'),
                }),
            ],

            'morphology_comparison': [
                ('morphology_comparison', {
                    'region1': params.get('region1') or params.get('region'),
                    'region2': params.get('region2'),
                }),
            ],

            'statistical_analysis': [
                ('region_diversity_metrics', {'region': params.get('region')}),
                ('gene_enrichment_analysis', {'gene': params.get('gene')}),
            ] if params.get('region') or params.get('gene') else [],

            'simple_lookup': [
                ('gene_to_clusters', {'gene': params.get('gene')}) if params.get('gene') else None,
                ('region_to_clusters', {'region': params.get('region')}) if params.get('region') else None,
            ],
        }

        sequence = sequences.get(question_type, [])

        # 过滤None和验证参数
        valid_sequence = []
        for item in sequence:
            if item is None:
                continue
            template_name, template_params = item
            # 验证所有参数都存在
            if all(v is not None for v in template_params.values()):
                valid_sequence.append((template_name, template_params))

        return valid_sequence if valid_sequence else [('gene_to_clusters', {'gene': params.get('gene', 'Pvalb')})]

    def _execute_template_sequence(self, sequence: List[Tuple[str, Dict]], params: Dict) -> Tuple[List[Dict], List[Dict]]:
        """执行模板序列"""
        results = []
        executed_steps = []

        for template_name, template_params in sequence:
            if template_name not in self.templates:
                logger.warning(f"  Template '{template_name}' not found")
                continue

            try:
                query = self.templates[template_name]
                result = self.db.run(query, template_params)

                results.append(result)
                executed_steps.append({
                    'purpose': f"Execute {template_name}",
                    'template': template_name,
                    'modality': self._infer_modality(template_name),
                    'success': result.get('success', False),
                    'params': template_params,
                })

                logger.info(f"  Template '{template_name}': {'✓' if result.get('success') else '✗'}")

            except Exception as e:
                logger.error(f"  Template '{template_name}' failed: {e}")
                executed_steps.append({
                    'purpose': f"Execute {template_name}",
                    'template': template_name,
                    'modality': None,
                    'success': False,
                    'error': str(e),
                })

        return results, executed_steps

    def _infer_modality(self, template_name: str) -> str:
        """推断模态"""
        if 'projection' in template_name or 'target' in template_name or 'circuit' in template_name:
            return 'projection'
        elif 'morphology' in template_name or 'morpholog' in template_name:
            return 'morphological'
        elif 'cluster' in template_name or 'gene' in template_name or 'marker' in template_name:
            return 'molecular'
        elif 'compare' in template_name or 'statistical' in template_name:
            return 'statistical'
        else:
            return 'mixed'

    def _synthesize_answer(self, question: str, results: List[Dict]) -> str:
        """合成答案（使用GPT-4o）"""
        all_data = []
        for result in results:
            if result.get('success') and result.get('data'):
                all_data.extend(result['data'][:10])

        if not all_data:
            return "No data found in knowledge graph."

        context = "Data from Knowledge Graph:\n"
        for i, row in enumerate(all_data[:25], 1):
            context += f"\n{i}. "
            context += ", ".join(f"{k}: {v}" for k, v in list(row.items())[:6])

        prompt = f"""Based on the following data from a neuroscience knowledge graph, provide a comprehensive scientific answer.

Question: {question}

{context}

Provide a detailed answer using the data above. Include:
1. Direct answer to the question
2. Quantitative data from the graph
3. Scientific interpretation
4. Key observations

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are analyzing neuroscience knowledge graph data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1200,
                timeout=40
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Based on {len(all_data)} data entries from knowledge graph. " + context[:600]

    def _fallback_answer(self, question: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': "Unable to extract parameters or execute templates for this question.",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'Template-KG',
        }

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'Template-KG',
            'error': error,
        }


# ==================== Baseline 3: RAG (Enhanced) ====================

class RAGBaseline(BaselineAgent):
    """
    RAG baseline (Enhanced with LLM-based keyword extraction)

    🔧 v3.0改进：
    - 使用LLM智能提取关键词（替代正则）
    - 改进检索策略
    - 更好的文档格式化
    """

    def __init__(self, neo4j_exec, openai_client):
        super().__init__("RAG")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = "gpt-4o"

    def answer(self, question: str, timeout: int = 120, **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        # 🔧 Step 1: 使用LLM智能提取关键词
        keywords = self._extract_keywords_with_llm(question)
        logger.info(f"  RAG: Extracted keywords: {keywords}")

        # Step 2: 检索文档
        docs = self._retrieve_documents_enhanced(keywords, top_k=15)
        logger.info(f"  RAG: Retrieved {len(docs)} documents")

        # Step 3: 构建context
        if docs:
            context = self._format_documents_enhanced(docs)
        else:
            context = "No relevant documents found in the knowledge graph."

        # Step 4: 生成答案
        try:
            answer = self._generate_answer(question, context, timeout)
            execution_time = time.time() - start_time

            entities_recognized = self._extract_entities_from_docs(docs)

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': [{
                    'purpose': f'Retrieved {len(docs)} documents from KG',
                    'modality': 'retrieval',
                    'keywords': keywords,
                }],
                'schema_paths_used': [],
                'execution_time': execution_time,
                'total_steps': 1,
                'confidence_score': 0.65,
                'success': True,
                'method': 'RAG',
            }

        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_keywords_with_llm(self, question: str) -> Dict[str, List[str]]:
        """🔧 使用LLM智能提取关键词（替代正则）"""

        try:
            prompt = f"""Extract key neuroscience entities from this question for knowledge graph search.

Question: {question}

Return JSON with these fields (use empty arrays if not found):
{{
  "genes": ["list of gene markers like Pvalb, Sst, Car3"],
  "regions": ["list of brain region acronyms like MOp, SSp, VISp"],
  "cell_types": ["specific cell type names"],
  "concepts": ["key neuroscience concepts like morphology, projection, connectivity"]
}}

Only include entities that are explicitly mentioned in the question."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert at extracting neuroscience entities for database queries."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300,
                timeout=15
            )

            entities = json.loads(response.choices[0].message.content)

            logger.info(f"  RAG: LLM extracted: {entities}")

            return entities

        except Exception as e:
            logger.warning(f"  RAG: LLM extraction failed, using fallback: {e}")
            return self._extract_keywords_fallback(question)

    def _extract_keywords_fallback(self, question: str) -> Dict[str, List[str]]:
        """Fallback: 正则提取"""
        entities = {
            'genes': [],
            'regions': [],
            'cell_types': [],
            'concepts': []
        }

        # 提取基因
        genes = re.findall(r'\b([A-Z][a-z]{2,8})\+?', question)
        stopwords = {'What', 'Which', 'Where', 'Tell', 'Give', 'Show', 'Find', 'The', 'This', 'That'}
        entities['genes'] = [g for g in genes if g not in stopwords][:3]

        # 提取脑区
        regions = re.findall(r'\b([A-Z]{2,5})\b', question)
        known_regions = {'MOp', 'MOs', 'SSp', 'VISp', 'AUDp', 'ACA', 'CLA', 'RSP', 'TH'}
        entities['regions'] = [r for r in regions if r in known_regions][:3]

        # 提取概念
        concepts = ['neuron', 'interneuron', 'projection', 'morphology', 'connectivity', 'circuit']
        q_lower = question.lower()
        entities['concepts'] = [c for c in concepts if c in q_lower][:3]

        return entities

    def _retrieve_documents_enhanced(self, entities: Dict[str, List[str]], top_k: int = 15) -> List[Dict]:
        """🔧 增强的文档检索"""
        docs = []

        # 1. 检索基因相关
        for gene in entities.get('genes', [])[:3]:
            query = """
            MATCH (c:Cluster)
            WHERE c.markers CONTAINS $gene
            RETURN 'Cluster' AS type, 
                   c.name AS cluster_name,
                   c.markers AS markers,
                   c.number_of_neurons AS neurons,
                   c.broad_region_distribution AS regions
            ORDER BY c.number_of_neurons DESC
            LIMIT 5
            """
            result = self.db.run(query, {'gene': gene})
            if result.get('success') and result.get('data'):
                for row in result['data']:
                    row['search_term'] = gene
                    row['search_type'] = 'gene'
                docs.extend(result['data'])

        # 2. 检索脑区相关
        for region in entities.get('regions', [])[:3]:
            # 2a. 脑区基本信息
            query_region = """
            MATCH (r:Region {acronym: $region})
            RETURN 'Region' AS type,
                   r.acronym AS acronym,
                   r.name AS name,
                   r.number_of_transcriptomic_neurons AS neuron_count
            """
            result = self.db.run(query_region, {'region': region})
            if result.get('success') and result.get('data'):
                for row in result['data']:
                    row['search_term'] = region
                    row['search_type'] = 'region'
                docs.extend(result['data'])

            # 2b. 脑区的clusters
            query_clusters = """
            MATCH (r:Region {acronym: $region})-[:HAS_CLUSTER]->(c:Cluster)
            RETURN 'RegionCluster' AS type,
                   r.acronym AS region,
                   c.name AS cluster_name,
                   c.markers AS markers,
                   c.number_of_neurons AS neurons
            ORDER BY c.number_of_neurons DESC
            LIMIT 5
            """
            result = self.db.run(query_clusters, {'region': region})
            if result.get('success') and result.get('data'):
                for row in result['data']:
                    row['search_term'] = region
                    row['search_type'] = 'region_clusters'
                docs.extend(result['data'])

            # 2c. 脑区的投射
            query_proj = """
            MATCH (r:Region {acronym: $region})-[p:PROJECT_TO]->(t:Region)
            RETURN 'Projection' AS type,
                   r.acronym AS source,
                   t.acronym AS target,
                   t.name AS target_name,
                   p.weight AS weight
            ORDER BY p.weight DESC
            LIMIT 5
            """
            result = self.db.run(query_proj, {'region': region})
            if result.get('success') and result.get('data'):
                for row in result['data']:
                    row['search_term'] = region
                    row['search_type'] = 'projection'
                docs.extend(result['data'])

        # 3. 概念相关（如果没有找到足够文档）
        if len(docs) < 5:
            for concept in entities.get('concepts', [])[:2]:
                if concept in ['interneuron', 'neuron']:
                    query = """
                    MATCH (c:Cluster)
                    WHERE c.name CONTAINS 'interneuron' OR c.markers CONTAINS 'Gad'
                    RETURN 'Cluster' AS type,
                           c.name AS cluster_name,
                           c.markers AS markers,
                           c.number_of_neurons AS neurons
                    ORDER BY c.number_of_neurons DESC
                    LIMIT 5
                    """
                    result = self.db.run(query)
                    if result.get('success') and result.get('data'):
                        docs.extend(result['data'])

        # 去重并限制数量
        seen = set()
        unique_docs = []
        for doc in docs:
            # 创建唯一key（避免完全重复）
            key_parts = [str(doc.get(k, '')) for k in ['type', 'cluster_name', 'acronym', 'source', 'target']]
            key = '|'.join(key_parts)

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs[:top_k]

    def _format_documents_enhanced(self, docs: List[Dict]) -> str:
        """🔧 增强的文档格式化"""
        if not docs:
            return "No documents found."

        formatted = []

        for i, doc in enumerate(docs, 1):
            doc_type = doc.get('type', 'Unknown')

            if doc_type == 'Region':
                text = f"**Brain Region**: {doc.get('name', 'N/A')} ({doc.get('acronym', 'N/A')})"
                if doc.get('neuron_count'):
                    text += f"\n  - Total neurons: {doc['neuron_count']:,}"

            elif doc_type == 'Cluster':
                text = f"**Cell Cluster**: {doc.get('cluster_name', 'N/A')}"
                if doc.get('markers'):
                    text += f"\n  - Markers: {doc['markers']}"
                if doc.get('neurons'):
                    text += f"\n  - Neuron count: {doc['neurons']:,}"
                if doc.get('regions'):
                    text += f"\n  - Distribution: {doc['regions']}"

            elif doc_type == 'RegionCluster':
                text = f"**Region-Cluster**: {doc.get('region', 'N/A')} contains {doc.get('cluster_name', 'N/A')}"
                if doc.get('markers'):
                    text += f"\n  - Markers: {doc['markers']}"
                if doc.get('neurons'):
                    text += f"\n  - Neurons: {doc['neurons']:,}"

            elif doc_type == 'Projection':
                text = f"**Projection**: {doc.get('source', 'N/A')} → {doc.get('target', 'N/A')}"
                if doc.get('target_name'):
                    text += f" ({doc['target_name']})"
                if doc.get('weight'):
                    text += f"\n  - Weight: {doc['weight']:.4f}"

            else:
                # Generic formatting
                text = f"**{doc_type}**"
                for key, value in list(doc.items())[:5]:
                    if key not in ['type', 'search_term', 'search_type']:
                        text += f"\n  - {key}: {value}"

            formatted.append(f"[Document {i}]\n{text}")

        return "\n\n".join(formatted)

    def _generate_answer(self, question: str, context: str, timeout: int) -> str:
        """生成答案"""

        system_prompt = """You are a neuroscience expert analyzing data from a knowledge graph.

Your task:
1. Answer the question using ONLY the provided documents
2. Be precise and cite specific data points
3. Include quantitative information when available
4. Acknowledge limitations if documents are insufficient
5. Organize your answer clearly"""

        user_prompt = f"""Based on the following documents from a neuroscience knowledge graph, answer the question.

{context}

Question: {question}

Provide a detailed, scientific answer using ONLY information from the documents above.

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            timeout=timeout
        )

        return response.choices[0].message.content

    def _extract_entities_from_docs(self, docs: List[Dict]) -> List[Dict]:
        """从文档提取实体"""
        entities = []

        for doc in docs[:10]:
            doc_type = doc.get('type')

            if doc_type == 'Region' and doc.get('acronym'):
                entities.append({
                    'text': doc['acronym'],
                    'type': 'Region',
                    'confidence': 1.0,
                })

            elif doc_type in ['Cluster', 'RegionCluster']:
                markers = doc.get('markers', '')
                if markers:
                    for marker in markers.split(',')[:3]:
                        entities.append({
                            'text': marker.strip(),
                            'type': 'Gene',
                            'confidence': 0.9,
                        })

            elif doc_type == 'Projection':
                if doc.get('source'):
                    entities.append({'text': doc['source'], 'type': 'Region', 'confidence': 1.0})
                if doc.get('target'):
                    entities.append({'text': doc['target'], 'type': 'Region', 'confidence': 1.0})

        # 去重
        seen = set()
        unique = []
        for e in entities:
            key = (e['text'], e['type'])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique[:15]

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'RAG',
            'error': error,
        }


# ==================== Baseline 4: ReAct (Enhanced) ====================

class ReActBaseline(BaselineAgent):
    """
    ReAct baseline (Enhanced with dynamic iterations)

    🔧 v3.0改进：
    - 根据问题复杂度动态调整max_iterations (3-10)
    - 与AIPOM-CoT公平对比
    - 改进query生成
    """

    def __init__(self, neo4j_exec, openai_client, base_max_iterations=10):
        super().__init__("ReAct")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = "gpt-4o"
        self.base_max_iterations = base_max_iterations

    def answer(self, question: str, timeout: int = 120, question_tier=None, **kwargs) -> Dict[str, Any]:
        """
        🔧 修复：根据问题复杂度动态调整迭代次数

        参数说明：
        - question_tier: 'simple', 'medium', 'deep', 'screening'
        """
        start_time = time.time()

        # 🔧 动态设置max_iterations（与AIPOM-CoT一致）
        max_iterations = self._determine_max_iterations(question, question_tier)

        logger.info(f"  ReAct: Using max_iterations={max_iterations} (tier={question_tier})")

        history = []
        executed_steps = []
        entities_recognized = []

        system_prompt = """You are a neuroscience expert with access to a Neo4j knowledge graph database.

You can execute Cypher queries to retrieve information about:
- Brain regions (Region nodes)
- Cell clusters (Cluster nodes)
- Neurons and morphology (Neuron nodes)
- Projections (PROJECT_TO relationships)

Use the ReAct (Reasoning + Acting) framework:

1. **Thought**: Analyze what information you need next
2. **Action**: Choose either:
   - "query": Execute a Cypher query to get data
   - "answer": Provide final answer when you have enough information
3. **Query**: If action="query", write a Cypher query
4. **Observation**: System provides query results
5. **Repeat** until you can answer

Respond in JSON format:
{
  "thought": "your reasoning about what to do next",
  "action": "query" or "answer",
  "query": "MATCH ... RETURN ... LIMIT 20" (if action is "query", otherwise null),
  "final_answer": "your comprehensive answer" (if action is "answer", otherwise null)
}

Tips:
- Keep queries focused and simple
- Always use LIMIT (max 20 rows)
- Build on previous observations
- Don't repeat queries"""

        try:
            for iteration in range(max_iterations):
                logger.info(f"  ReAct iteration {iteration + 1}/{max_iterations}")

                # 构建context
                if history:
                    context = "\n\n".join(history[-5:])  # 只保留最近5步
                else:
                    context = "This is your first step. Start by reasoning about what information you need."

                prompt = f"""Question: {question}

Previous reasoning and observations:
{context}

What's your next step? Respond in JSON format.

Remember:
- If you have enough information, use action="answer"
- If you need more data, use action="query" with a Cypher query
- Build on what you already know"""

                # LLM推理
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=1000,
                        timeout=max(20, timeout // max_iterations)
                    )

                    result = json.loads(response.choices[0].message.content)

                except json.JSONDecodeError as e:
                    logger.warning(f"    JSON decode error: {e}, retrying...")
                    continue

                thought = result.get('thought', '')
                action = result.get('action', '')

                if not thought:
                    logger.warning(f"    Empty thought, skipping")
                    continue

                history.append(f"**Iteration {iteration + 1}**\nThought: {thought}")
                logger.info(f"    Thought: {thought[:100]}...")

                # Action: answer
                if action == 'answer':
                    final_answer = result.get('final_answer', '')

                    if not final_answer:
                        logger.warning(f"    Empty final answer, continuing...")
                        continue

                    execution_time = time.time() - start_time

                    logger.info(f"  ReAct: Finished in {iteration + 1} iterations")

                    return {
                        'question': question,
                        'answer': final_answer,
                        'entities_recognized': entities_recognized,
                        'executed_steps': executed_steps,
                        'schema_paths_used': [],
                        'execution_time': execution_time,
                        'total_steps': len(executed_steps),
                        'confidence_score': 0.7,
                        'success': True,
                        'method': 'ReAct',
                        'iterations_used': iteration + 1,
                    }

                # Action: query
                elif action == 'query':
                    query = result.get('query', '')

                    if not query:
                        logger.warning(f"    Empty query, skipping")
                        history.append("Action: Attempted query but it was empty")
                        continue

                    history.append(f"Action: Execute Cypher query")
                    logger.info(f"    Query: {query[:100]}...")

                    # 执行查询
                    db_result = self.db.run(query)

                    if db_result.get('success'):
                        data = db_result.get('data', [])[:20]

                        if data:
                            observation = f"Query returned {len(data)} results:\n"
                            # 格式化前3条结果
                            for i, row in enumerate(data[:3], 1):
                                observation += f"  {i}. " + ", ".join(
                                    f"{k}={v}" for k, v in list(row.items())[:4]) + "\n"

                            if len(data) > 3:
                                observation += f"  ... and {len(data) - 3} more results"
                        else:
                            observation = "Query returned 0 results"

                        # 提取实体
                        entities_recognized.extend(self._extract_entities_from_data(data))

                    else:
                        error = db_result.get('error', 'Unknown error')
                        observation = f"Query failed: {error}"
                        data = []

                    history.append(f"Observation: {observation}")
                    logger.info(f"    {observation[:100]}...")

                    executed_steps.append({
                        'purpose': thought[:100],
                        'query': query,
                        'result_count': len(data) if db_result.get('success') else 0,
                        'success': db_result.get('success', False),
                        'modality': self._infer_modality(query),
                    })

                else:
                    logger.warning(f"    Unknown action: {action}")
                    history.append(f"Action: Unknown action '{action}'")

            # 达到最大迭代
            logger.warning(f"  ReAct: Reached max_iterations={max_iterations}")

            final_answer = f"Analysis incomplete after {max_iterations} iterations. "

            if executed_steps:
                successful_queries = sum(1 for s in executed_steps if s.get('success'))
                final_answer += f"Executed {len(executed_steps)} queries ({successful_queries} successful) but need more iterations to complete the analysis."
            else:
                final_answer += "Could not generate valid queries."

            return {
                'question': question,
                'answer': final_answer,
                'entities_recognized': entities_recognized,
                'executed_steps': executed_steps,
                'schema_paths_used': [],
                'execution_time': time.time() - start_time,
                'total_steps': len(executed_steps),
                'confidence_score': 0.4,
                'success': False,
                'method': 'ReAct',
                'iterations_used': max_iterations,
            }

        except Exception as e:
            logger.error(f"ReAct failed: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(question, str(e), time.time() - start_time)

    def _determine_max_iterations(self, question: str, tier: Optional[str] = None) -> int:
        """🔧 根据问题复杂度确定最大迭代次数"""

        # 如果明确给了tier，使用tier
        if tier:
            tier_map = {
                'simple': 3,
                'medium': 6,
                'deep': 10,
                'screening': 8,
            }
            return tier_map.get(tier, self.base_max_iterations)

        # 否则根据问题内容启发式判断
        q_lower = question.lower()

        # Deep indicators
        if any(kw in q_lower for kw in ['comprehensive', 'detailed', 'complete', 'full', 'analyze in depth']):
            return 10

        # Medium-Complex indicators
        if any(kw in q_lower for kw in ['compare', 'analyze', 'characterize', 'profile']):
            return 6

        # Simple indicators
        if any(kw in q_lower for kw in ['what is', 'define', 'how many', 'list']):
            return 3

        # Default
        return self.base_max_iterations

    def _infer_modality(self, query: str) -> str:
        """推断查询的modality"""
        query_lower = query.lower()

        if 'project' in query_lower or 'target' in query_lower:
            return 'projection'
        elif 'morpholog' in query_lower or 'axon' in query_lower or 'dendrit' in query_lower:
            return 'morphological'
        elif 'cluster' in query_lower or 'marker' in query_lower:
            return 'molecular'
        else:
            return None

    def _extract_entities_from_data(self, data: List[Dict]) -> List[Dict]:
        """从数据提取实体"""
        entities = []

        for row in data[:8]:
            for key, value in row.items():
                if not isinstance(value, str):
                    continue

                # 脑区缩写
                if len(value) >= 2 and len(value) <= 5 and value.isupper():
                    entities.append({
                        'text': value,
                        'type': 'Region',
                        'confidence': 0.8,
                    })
                # 基因名
                elif len(value) >= 3 and value[0].isupper() and not value.isupper():
                    entities.append({
                        'text': value,
                        'type': 'Gene',
                        'confidence': 0.6,
                    })

        # 去重
        seen = set()
        unique = []
        for e in entities:
            key = (e['text'], e['type'])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique[:12]

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'ReAct',
            'error': error,
        }


# ==================== Factory Function ====================

def create_baseline(baseline_type: str, **kwargs) -> BaselineAgent:
    """工厂函数创建baseline"""

    if baseline_type == 'direct-gpt4o':
        return DirectGPT4oBaseline(
            openai_client=kwargs['openai_client']
        )

    elif baseline_type == 'template-kg':
        return TemplateKGBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client']
        )

    elif baseline_type == 'rag':
        return RAGBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client']
        )

    elif baseline_type == 'react':
        return ReActBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client'],
            base_max_iterations=kwargs.get('base_max_iterations', 10)
        )

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


# ==================== Test ====================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("✅ Enhanced baselines.py v3.0 (Fair Comparison) loaded successfully!")
    print("=" * 80)
    print("\nAvailable baselines:")
    print("1. Direct GPT-4o   - Strong LLM baseline (no KG)")
    print("2. Template-KG     - 27 templates covering multi-hop, comparison, closed-loop")
    print("3. RAG             - LLM-based keyword extraction + enhanced retrieval")
    print("4. ReAct           - Dynamic iterations (3-10) based on question complexity")
    print("\n🔧 Key improvements for fairness:")
    print("  ✓ ReAct max_iterations: 3-10 (matches AIPOM-CoT)")
    print("  ✓ Template-KG: 27 templates (vs original 5)")
    print("  ✓ RAG: LLM keyword extraction (vs regex)")
    print("  ✓ All baselines fully optimized")
    print("=" * 80)