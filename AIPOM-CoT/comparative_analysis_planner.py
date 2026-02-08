import logging
from typing import List, Dict, Optional
from itertools import combinations

logger = logging.getLogger(__name__)


class ComparativeAnalysisPlanner:
    """系统对比分析规划器"""

    def __init__(self, db, fingerprint_analyzer, stats_tools, config=None):
        self.db = db
        self.fingerprint = fingerprint_analyzer
        self.stats = stats_tools

        # 配置参数
        self.config = config or {
            'max_regions': 50,
            'n_top_pairs': 10,
            'mismatch_threshold': 0.5,
            'fdr_alpha': 0.05,
            'n_permutations': 1000
        }

    def generate_comparative_plan(self,
                                  analysis_state,
                                  question: str) -> List:
        """
        生成对比分析计划

        自动判断模式:
        - "Compare A and B" → Pairwise
        - "Which regions..." → Systematic
        """

        question_lower = question.lower()

        # Pairwise模式
        if self._is_pairwise_question(question_lower):
            return self._pairwise_comparison_plan(analysis_state, question)

        # Systematic模式
        elif self._is_systematic_question(question_lower):
            return self._systematic_screening_plan(analysis_state, question)

        else:
            logger.warning("Could not determine comparison mode")
            return []

    def _is_pairwise_question(self, question_lower: str) -> bool:
        """判断是否是pairwise比较"""
        pairwise_keywords = ['compare', 'versus', 'vs', 'difference between', 'contrast']
        return any(kw in question_lower for kw in pairwise_keywords)

    def _is_systematic_question(self, question_lower: str) -> bool:
        """判断是否是systematic筛选"""
        systematic_keywords = ['which', 'find', 'identify', 'screen', 'all regions', 'highest', 'top']
        return any(kw in question_lower for kw in systematic_keywords)

    def _pairwise_comparison_plan(self, state, question) -> List:
        """Pairwise对比计划"""
        from adaptive_planner import CandidateStep

        entities = state.discovered_entities.get('Region', [])

        if len(entities) < 2:
            logger.warning("Need at least 2 entities for pairwise comparison")
            return []

        entity_a, entity_b = entities[0], entities[1]

        logger.info(f"Pairwise comparison: {entity_a} vs {entity_b}")

        return [
            # Molecular comparison
            CandidateStep(
                step_id='pairwise_molecular',
                step_type='molecular',
                purpose=f'Compare molecular profiles: {entity_a} vs {entity_b}',
                rationale='Cell type composition differences reveal regional specialization',
                priority=9.0,
                schema_path='Region -[HAS_CLUSTER]-> Cluster',
                expected_data='Side-by-side cluster composition',
                cypher_template="""
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE r.acronym IN [$entity_a, $entity_b]
                RETURN r.acronym AS region,
                       c.name AS cluster,
                       c.markers AS markers,
                       c.number_of_neurons AS neurons
                ORDER BY r.acronym, c.number_of_neurons DESC
                LIMIT 100
                """,
                parameters={'entity_a': entity_a, 'entity_b': entity_b},
                depends_on=[]
            ),

            # Morphological comparison
            CandidateStep(
                step_id='pairwise_morphological',
                step_type='morphological',
                purpose=f'Compare morphological features: {entity_a} vs {entity_b}',
                rationale='Morphological differences indicate functional specialization',
                priority=8.5,
                schema_path='Neuron -[LOCATE_AT]-> Region',
                expected_data='Morphological statistics for both regions',
                cypher_template="""
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                WHERE r.acronym IN [$entity_a, $entity_b]
                RETURN r.acronym AS region,
                       count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon,
                       avg(n.dendritic_length) AS avg_dendrite,
                       stdev(n.axonal_length) AS std_axon,
                       stdev(n.dendritic_length) AS std_dendrite
                """,
                parameters={'entity_a': entity_a, 'entity_b': entity_b},
                depends_on=[]
            ),

            # Projection comparison
            CandidateStep(
                step_id='pairwise_projection',
                step_type='projection',
                purpose=f'Compare connectivity patterns: {entity_a} vs {entity_b}',
                rationale='Projection patterns reveal functional roles in brain circuits',
                priority=8.5,
                schema_path='Region -[PROJECT_TO]-> Target',
                expected_data='Projection targets for both regions',
                cypher_template="""
                MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                WHERE r.acronym IN [$entity_a, $entity_b]
                RETURN r.acronym AS source,
                       t.acronym AS target,
                       p.weight AS weight
                ORDER BY r.acronym, p.weight DESC
                LIMIT 100
                """,
                parameters={'entity_a': entity_a, 'entity_b': entity_b},
                depends_on=[]
            ),

            # Statistical comparison
            CandidateStep(
                step_id='pairwise_statistical',
                step_type='statistical',
                purpose=f'Statistical significance test: {entity_a} vs {entity_b}',
                rationale='Quantify significance of observed morphological differences',
                priority=9.0,
                schema_path='Statistical analysis',
                expected_data='P-values, effect sizes, confidence intervals',
                cypher_template='',  # Python计算
                parameters={
                    'test_type': 'permutation',
                    'comparison_type': 'morphology',
                    'entity_a': entity_a,
                    'entity_b': entity_b
                },
                depends_on=['pairwise_morphological']
            )
        ]

    def _systematic_screening_plan(self, state, question) -> List:
        """
        Systematic全脑筛选计划（防重复版）

        🔧 关键修复：
        1. 检查已执行步骤
        2. 避免重复执行
        3. 添加调试日志

        Steps:
        1. Get top regions by neuron count
        2. Compute cross-modal mismatch
        3. FDR correction
        4. Characterize top pairs
        """
        from adaptive_planner import CandidateStep

        candidates = []

        # 🔧 获取已执行步骤的ID
        executed_step_ids = set()
        for step in state.executed_steps:
            # ✅ 兼容dict和对象两种格式
            if isinstance(step, dict):
                step_id = step.get('step_id', '')
                if not step_id:
                    # Fallback: 从purpose生成ID
                    purpose = step.get('purpose', '')
                    step_id = purpose.lower().replace(' ', '_')[:30]
                executed_step_ids.add(step_id)
            elif hasattr(step, 'step_id'):
                executed_step_ids.add(step.step_id)

        logger.info(f"   Already executed: {executed_step_ids}")

        # ===== Step 1: Get regions (只执行一次) =====
        if 'systematic_get_regions' not in executed_step_ids:
            logger.info("   Adding Step 1: Get top regions")
            candidates.append(
                CandidateStep(
                    step_id='systematic_get_regions',
                    step_type='spatial',
                    purpose='Identify top brain regions by neuron count',
                    rationale='Select regions with most neurons for systematic comparison (Figure 4 method)',
                    priority=9.5,
                    schema_path='Region-Neuron relationship',
                    expected_data='Top 30 regions ordered by neuron count',
                    cypher_template="""
                    MATCH (r:Region)
                    OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
                    WITH r, COUNT(DISTINCT n) AS neuron_count
                    WHERE neuron_count > 0
                    RETURN r.acronym AS region,
                           r.name AS region_name,
                           neuron_count
                    ORDER BY neuron_count DESC
                    LIMIT 30
                    """,
                    parameters={},
                    depends_on=[]
                )
            )
        else:
            logger.info("   Skipping Step 1: Already executed")

        # ===== Step 2: Compute mismatch (只执行一次) =====
        if 'systematic_mismatch' not in executed_step_ids:
            # ✅ 确保Step 1已执行
            if 'systematic_get_regions' in executed_step_ids or 'systematic_get_regions' in [c.step_id for c in
                                                                                             candidates]:
                logger.info("   Adding Step 2: Compute mismatch")
                candidates.append(
                    CandidateStep(
                        step_id='systematic_mismatch',
                        step_type='multi-modal',
                        purpose='Compute cross-modal mismatch index for all region pairs',
                        rationale='Mismatch quantifies molecular-morphological-projection discordance',
                        priority=10.0,
                        schema_path='Fingerprint computation',
                        expected_data='Mismatch matrix with p-values and similarity scores',
                        cypher_template='',  # Python计算
                        parameters={
                            'analysis_type': 'cross_modal_mismatch',
                            'modalities': ['molecular', 'morphological', 'projection'],
                            'n_pairs': self.config['max_regions'] * 10
                        },
                        depends_on=['systematic_get_regions']
                    )
                )
            else:
                logger.warning("   Cannot add Step 2: Step 1 not ready")
        else:
            logger.info("   Skipping Step 2: Already executed")

        # ===== Step 3: FDR correction (只执行一次) =====
        if 'systematic_fdr' not in executed_step_ids:
            # ✅ 确保Step 2已执行
            if 'systematic_mismatch' in executed_step_ids:
                logger.info("   Adding Step 3: FDR correction")
                candidates.append(
                    CandidateStep(
                        step_id='systematic_fdr',
                        step_type='statistical',
                        purpose='FDR correction for multiple testing',
                        rationale='Control false discovery rate in large-scale screening',
                        priority=9.5,
                        schema_path='FDR correction',
                        expected_data='Significant pairs after FDR adjustment',
                        cypher_template='',
                        parameters={
                            'test_type': 'fdr',
                            'alpha': self.config['fdr_alpha']
                        },
                        depends_on=['systematic_mismatch']
                    )
                )
            else:
                logger.info("   Skipping Step 3: Step 2 not ready yet")
        else:
            logger.info("   Skipping Step 3: Already executed")

        # ===== Step 4: Case study (只执行一次) =====
        if 'systematic_characterize_top' not in executed_step_ids:
            # ✅ 确保Step 3已执行
            if 'systematic_fdr' in executed_step_ids:
                logger.info("   Adding Step 4: Characterize top pairs")
                candidates.append(
                    CandidateStep(
                        step_id='systematic_characterize_top',
                        step_type='molecular',
                        purpose='Deep characterization of top mismatch pairs',
                        rationale='Understand biological mechanisms driving high mismatch',
                        priority=8.5,
                        schema_path='Multi-modal queries',
                        expected_data='Detailed profiles of high-mismatch region pairs',
                        cypher_template='',
                        parameters={
                            'n_top_pairs': self.config['n_top_pairs']
                        },
                        depends_on=['systematic_fdr']
                    )
                )
            else:
                logger.info("   Skipping Step 4: Step 3 not ready yet")
        else:
            logger.info("   Skipping Step 4: Already executed")

        if not candidates:
            logger.info("   No new steps to add (all systematic steps completed or waiting)")

        return candidates