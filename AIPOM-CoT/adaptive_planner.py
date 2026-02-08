import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== Data Structures ====================

class AnalysisDepth(Enum):
    """分析深度"""
    SHALLOW = "shallow"      # 浅层: 1-2步,只找直接关系
    MEDIUM = "medium"        # 中层: 3-4步,2-hop分析
    DEEP = "deep"           # 深层: 5-6步,多模态闭环


@dataclass
class AnalysisState:
    """当前分析状态"""
    # 已发现的实体 {type: [entities]}
    discovered_entities: Dict[str, List[Any]] = field(default_factory=dict)

    # 已执行的步骤
    executed_steps: List[Dict] = field(default_factory=list)

    # 已覆盖的模态
    modalities_covered: List[str] = field(default_factory=list)

    # 当前焦点
    current_focus: str = 'gene'  # 'gene' | 'region' | 'projection_target'

    # 分析深度
    target_depth: AnalysisDepth = AnalysisDepth.MEDIUM

    # 问题意图
    question_intent: str = ''  # 'simple_query' | 'comprehensive' | 'comparison'


@dataclass
class CandidateStep:
    """候选步骤"""
    step_id: str
    step_type: str           # 'molecular' | 'morphological' | 'projection' | 'spatial'
    purpose: str
    rationale: str
    priority: float          # 基础优先级 0-10
    schema_path: str
    expected_data: str
    cypher_template: str
    parameters: Dict
    depends_on: List[str]

    # LLM评估 (由rank_steps填充)
    llm_score: float = 0.0
    llm_reasoning: str = ""


# ==================== Adaptive Planner ====================

class AdaptivePlanner:
    """
    自适应规划器

    核心思想: 不预设路径,根据当前状态动态决定下一步
    """

    def __init__(self, schema, path_finder, llm_client):
        self.schema = schema
        self.path_finder = path_finder  # SchemaPathFinder实例
        self.llm = llm_client

    # ==================== 主入口 ====================

    def plan_next_steps(self,
                       state: AnalysisState,
                       question: str,
                       max_steps: int = 2) -> List[CandidateStep]:
        """
        规划接下来的N步

        这是主入口方法!

        Args:
            state: 当前分析状态
            question: 原始问题
            max_steps: 最多规划几步

        Returns:
            排序后的候选步骤列表
        """
        logger.info(f"🎯 Adaptive planning (depth: {state.target_depth.value})...")

        # Step 1: 分析当前状态
        state_analysis = self._analyze_state(state)
        logger.info(f"   State: {state_analysis}")

        # Step 2: 生成候选步骤
        candidates = self._generate_candidate_steps(state, state_analysis)
        logger.info(f"   Generated {len(candidates)} candidate steps")

        if not candidates:
            logger.info("   No candidates available")
            return []

        # Step 3: LLM评分排序
        ranked_steps = self._rank_steps_by_value(
            candidates,
            question,
            state_analysis
        )

        # Step 4: 返回top-N
        selected = ranked_steps[:max_steps]

        for i, step in enumerate(selected, 1):
            logger.info(f"   {i}. {step.purpose} (score: {step.llm_score:.2f})")

        return selected

    def should_continue(self, state: AnalysisState, question: str) -> bool:
        """
        决定是否应该继续分析

        终止条件:
        1. 已达到目标深度
        2. 所有关键模态已覆盖
        3. LLM认为已完成
        """

        # 硬性限制
        max_steps_map = {
            AnalysisDepth.SHALLOW: 2,
            AnalysisDepth.MEDIUM: 4,
            AnalysisDepth.DEEP: 8
        }

        max_allowed = max_steps_map[state.target_depth]
        if len(state.executed_steps) >= max_allowed:
            logger.info(f"📌 Reached max steps for {state.target_depth.value} depth")
            return False

        # 分析完整性检查
        analysis = self._analyze_state(state)

        # Deep模式需要闭环
        if state.target_depth == AnalysisDepth.DEEP:
            if not analysis['projection_targets_analyzed']:
                logger.info("   Need to complete loop - analyzing projection targets")
                return True

        # LLM判断
        return self._llm_should_continue(state, question, analysis)

    # ==================== 状态分析 ====================

    def _analyze_state(self, state: AnalysisState) -> Dict:
        """
        分析当前状态,返回分析摘要

        Returns:
            {
                'has_gene': bool,
                'has_subclass': bool,
                'has_cluster': bool,
                'has_regions': bool,
                'has_morphology': bool,
                'has_projections': bool,
                'projection_targets_analyzed': bool,
                'missing_modalities': [str],
                'depth_achieved': int,
                'region_count': int,
                'target_count': int
            }
        """
        entities = state.discovered_entities

        analysis = {
            'has_gene': 'GeneMarker' in entities and len(entities['GeneMarker']) > 0,
            'has_subclass': 'Subclass' in entities and len(entities['Subclass']) > 0,
            'has_cluster': 'Cluster' in entities and len(entities['Cluster']) > 0,
            'has_regions': 'Region' in entities and len(entities['Region']) > 0,
            'has_morphology': 'morphological' in state.modalities_covered,
            'has_projections': 'projection' in state.modalities_covered,
            'projection_targets_analyzed': False,
            'missing_modalities': [],
            'depth_achieved': len(state.executed_steps),
            'region_count': len(entities.get('Region', [])),
            'target_count': len(entities.get('ProjectionTarget', []))
        }

        # 检查缺失的模态
        all_modalities = {'molecular', 'morphological', 'projection'}
        covered = set(state.modalities_covered)
        analysis['missing_modalities'] = list(all_modalities - covered)

        # 检查是否对projection targets做了分子分析
        if analysis['has_projections']:
            # 查看最近步骤
            recent_steps = state.executed_steps[-3:]
            for step in recent_steps:
                purpose_lower = step.get('purpose', '').lower()
                if ('target' in purpose_lower or 'projection' in purpose_lower) and \
                   step.get('modality') == 'molecular':
                    analysis['projection_targets_analyzed'] = True
                    break

        return analysis

    # ==================== 候选步骤生成 ====================

    def _generate_candidate_steps(self,
                                  state: AnalysisState,
                                  analysis: Dict) -> List[CandidateStep]:
        """
        生成所有可能的候选步骤

        这是核心逻辑! 根据schema和当前状态,生成所有可行的下一步
        """
        candidates = []

        # ===== 1. 分子层面候选 =====
        candidates.extend(self._generate_molecular_candidates(state, analysis))

        # ===== 2. 形态层面候选 =====
        candidates.extend(self._generate_morphological_candidates(state, analysis))

        # ===== 3. 投射层面候选 =====
        candidates.extend(self._generate_projection_candidates(state, analysis))

        # ===== 4. 空间/层级候选 =====
        candidates.extend(self._generate_spatial_candidates(state, analysis))

        return candidates

    def _generate_molecular_candidates(self,
                                       state: AnalysisState,
                                       analysis: Dict) -> List[CandidateStep]:
        """
        生成分子层面的候选步骤（增强版 v2.0）

        🔧 关键修复：
        1. 提高闭环步骤优先级
        2. 添加智能fallback提取targets
        3. 增强日志
        """
        candidates = []

        # 🔹 Candidate 1: Gene -> Subclass
        if analysis['has_gene'] and not analysis['has_subclass']:
            gene = state.discovered_entities['GeneMarker'][0]

            candidates.append(CandidateStep(
                step_id='mol_gene_to_subclass',
                step_type='molecular',
                purpose=f'Find Subclass cell types expressing {gene}',
                rationale='Gene markers define cell types at the Subclass taxonomy level',
                priority=9.0,
                schema_path='Subclass (via markers property)',
                expected_data='List of Subclass nodes with gene in markers field',
                cypher_template="""
                MATCH (s:Subclass)
                WHERE s.markers CONTAINS $gene
                RETURN s.name AS subclass_name,
                       s.markers AS markers,
                       s.description AS description
                ORDER BY s.name
                LIMIT 20
                """,
                parameters={'gene': gene},
                depends_on=[]
            ))

        # 🔹 Candidate 2: Gene -> Cluster
        if analysis['has_gene'] and not analysis['has_cluster']:
            gene = state.discovered_entities['GeneMarker'][0]

            candidates.append(CandidateStep(
                step_id='mol_gene_to_cluster',
                step_type='molecular',
                purpose=f'Find cell clusters expressing {gene}',
                rationale='Clusters provide finer-grained cell type resolution than Subclass',
                priority=8.5,
                schema_path='Cluster (via markers property)',
                expected_data='List of Cluster nodes with neuron counts',
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
                parameters={'gene': gene},
                depends_on=[]
            ))

        # 🔹 Candidate 3: Subclass/Cluster -> Region
        if (analysis['has_subclass'] or analysis['has_cluster']) and not analysis['has_regions']:
            gene = state.discovered_entities.get('GeneMarker', ['unknown'])[0]

            candidates.append(CandidateStep(
                step_id='mol_cluster_to_region',
                step_type='molecular',
                purpose=f'Identify brain regions enriched for {gene}+ cell types',
                rationale='Spatial localization reveals regional specialization',
                priority=8.0,
                schema_path='Region -[HAS_CLUSTER]-> Cluster',
                expected_data='Regions ranked by cluster count and neuron density',
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
                parameters={'gene': gene},
                depends_on=['mol_gene_to_cluster']
            ))

        # 🔹 Candidate 4: Region -> Cluster composition
        if analysis['has_regions'] and not any('composition' in s['purpose'].lower()
                                               for s in state.executed_steps):
            regions = state.discovered_entities.get('Region', [])[:5]

            candidates.append(CandidateStep(
                step_id='mol_region_composition',
                step_type='molecular',
                purpose='Characterize detailed cell type composition of discovered regions',
                rationale='Understanding local cell type diversity reveals functional organization',
                priority=7.0,
                schema_path='Region -[HAS_CLUSTER]-> Cluster',
                expected_data='Cluster distribution per region',
                cypher_template="""
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE r.acronym IN $regions
                RETURN r.acronym AS region,
                       c.name AS cluster,
                       c.markers AS markers,
                       c.number_of_neurons AS neurons
                ORDER BY r.acronym, c.number_of_neurons DESC
                LIMIT 50
                """,
                parameters={'regions': regions},
                depends_on=['mol_cluster_to_region']
            ))

        # 🔹 Candidate 5: Projection Target -> Molecular composition (闭环!) 🔥
        # 🔧 关键修复：提高优先级 + 智能fallback
        if analysis['has_projections'] and not analysis['projection_targets_analyzed']:

            # 尝试从discovered_entities获取targets
            targets = state.discovered_entities.get('ProjectionTarget', [])[:5]

            # 🔧 Fallback 1: 如果没有discovered targets，从最近的projection步骤提取
            if not targets:
                logger.info(f"   🔍 Fallback: Extracting targets from recent projection steps")

                for step in reversed(state.executed_steps[-5:]):  # 检查最近5步
                    if 'projection' in step.get('purpose', '').lower():
                        step_num = step.get('step_number', 0)
                        step_data_key = f"step_{step_num}"

                        # 从intermediate_data获取
                        if hasattr(state, 'intermediate_data'):
                            step_data = getattr(state, 'intermediate_data', {}).get(step_data_key, [])
                        else:
                            step_data = []

                        if step_data and isinstance(step_data, list) and len(step_data) > 0:
                            # 尝试多个字段名
                            target_fields = ['target', 'target_region', 'target_acronym',
                                             'tgt', 'downstream_region']

                            for field in target_fields:
                                if field in step_data[0]:
                                    targets = [
                                        row.get(field)
                                        for row in step_data[:5]
                                        if row.get(field)
                                    ]
                                    if targets:
                                        logger.info(f"      ✓ Extracted {len(targets)} targets from '{field}' field")
                                        break

                            if targets:
                                break

            # 🔧 Fallback 2: 如果还是没有，从projection步骤的purpose提取
            if not targets:
                logger.warning(f"   ⚠️ Could not extract targets from data, trying purpose text")

                for step in reversed(state.executed_steps[-5:]):
                    purpose = step.get('purpose', '').lower()
                    if 'projection' in purpose or 'target' in purpose:
                        # 尝试提取大写字母组合（可能是region acronyms）
                        import re
                        potential_targets = re.findall(r'\b[A-Z]{2,5}\b', step.get('purpose', ''))

                        if potential_targets:
                            # 过滤掉常见非region词
                            stopwords = {'MOp', 'MOs', 'SSp', 'VISp'}  # 源region
                            targets = [t for t in potential_targets if t not in stopwords][:5]

                            if targets:
                                logger.info(f"      ✓ Extracted {len(targets)} targets from purpose text: {targets}")
                                break

            # 如果有targets，生成闭环步骤
            if targets:
                candidates.append(CandidateStep(
                    step_id='mol_target_composition',
                    step_type='molecular',
                    purpose=f'Analyze molecular composition of projection target regions (CLOSED LOOP!)',
                    rationale='Complete the circuit analysis loop by characterizing downstream target cell types',
                    priority=9.8,  # 🔧 提高优先级！(原8.5 → 9.8)
                    schema_path='Target_Region -[HAS_CLUSTER]-> Cluster',
                    expected_data='Cell type composition of target regions',
                    cypher_template="""
                    MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                    WHERE r.acronym IN $targets
                    RETURN r.acronym AS target_region,
                           r.name AS target_name,
                           c.name AS cluster,
                           c.markers AS markers,
                           c.number_of_neurons AS neurons
                    ORDER BY r.acronym, c.number_of_neurons DESC
                    LIMIT 50
                    """,
                    parameters={'targets': targets},
                    depends_on=['proj_identify_targets']
                ))

                logger.info(f"   🔥 Generated CLOSED LOOP step for {len(targets)} targets")
                logger.info(f"      Priority: 9.8 (highest!)")
                logger.info(f"      Targets: {targets}")
            else:
                # 如果有projections但真的提取不到targets
                if analysis['has_projections']:
                    logger.warning(f"   ❌ Has projections but could NOT extract any targets")
                    logger.warning(f"      This will prevent closed-loop analysis")
                    logger.warning(f"      Recent steps: {[s['purpose'][:50] for s in state.executed_steps[-3:]]}")

        return candidates

    def _generate_morphological_candidates(self,
                                          state: AnalysisState,
                                          analysis: Dict) -> List[CandidateStep]:
        """生成形态层面的候选步骤"""
        candidates = []

        # 🔹 Candidate: Region -> Morphology
        if analysis['has_regions'] and not analysis['has_morphology']:
            regions = state.discovered_entities.get('Region', [])[:10]

            candidates.append(CandidateStep(
                step_id='morph_region_features',
                step_type='morphological',
                purpose='Analyze morphological features of neurons in discovered regions',
                rationale='Morphological specialization reflects functional roles',
                priority=7.0,
                schema_path='Region <-[LOCATE_AT]- Neuron',
                expected_data='Aggregated morphological statistics (axon, dendrite, soma)',
                cypher_template="""
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                WHERE r.acronym IN $regions
                RETURN r.acronym AS region,
                       count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon_length,
                       avg(n.dendritic_length) AS avg_dendrite_length,
                       avg(n.axonal_branches) AS avg_axon_branches,
                       avg(n.dendritic_branches) AS avg_dendrite_branches,
                       avg(n.soma_surface) AS avg_soma_surface
                ORDER BY neuron_count DESC
                LIMIT 20
                """,
                parameters={'regions': regions},
                depends_on=['mol_cluster_to_region']
            ))

        return candidates

    def _generate_projection_candidates(self,
                                       state: AnalysisState,
                                       analysis: Dict) -> List[CandidateStep]:
        """生成投射层面的候选步骤"""
        candidates = []

        # 🔹 Candidate 1: Region -> Projection targets
        if analysis['has_regions'] and not analysis['has_projections']:
            regions = state.discovered_entities.get('Region', [])[:10]

            candidates.append(CandidateStep(
                step_id='proj_identify_targets',
                step_type='projection',
                purpose='Identify projection targets of discovered regions',
                rationale='Connectivity patterns reveal functional integration and information flow',
                priority=7.5,
                schema_path='Region -[PROJECT_TO]-> Target_Region',
                expected_data='Projection weights and target regions',
                cypher_template="""
                MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                WHERE r.acronym IN $regions
                RETURN r.acronym AS source,
                       t.acronym AS target,
                       t.name AS target_name,
                       p.weight AS projection_weight,
                       p.neuron_count AS neuron_count
                ORDER BY p.weight DESC
                LIMIT 50
                """,
                parameters={'regions': regions},
                depends_on=['mol_cluster_to_region']
            ))

        # 🔹 Candidate 2: Categorize targets by function
        if analysis['has_projections'] and state.target_depth == AnalysisDepth.DEEP:
            targets = state.discovered_entities.get('ProjectionTarget', [])[:10]

            if targets:
                candidates.append(CandidateStep(
                    step_id='proj_categorize_targets',
                    step_type='projection',
                    purpose='Categorize projection targets by functional systems',
                    rationale='Grouping targets reveals whether circuit is sensory, motor, or associative',
                    priority=6.0,
                    schema_path='Target_Region properties',
                    expected_data='Functional categories of targets',
                    cypher_template="""
                    MATCH (t:Region)
                    WHERE t.acronym IN $targets
                    RETURN t.acronym AS target,
                           t.name AS target_name,
                           t.parent_structure AS parent,
                           t.rgb_triplet AS color_code
                    ORDER BY t.name
                    LIMIT 30
                    """,
                    parameters={'targets': targets},
                    depends_on=['proj_identify_targets']
                ))

        return candidates

    def _generate_spatial_candidates(self,
                                    state: AnalysisState,
                                    analysis: Dict) -> List[CandidateStep]:
        """生成空间/层级的候选步骤"""
        candidates = []

        # 🔹 Candidate: Identify subregions of major targets
        if analysis['has_projections'] and state.target_depth == AnalysisDepth.DEEP:
            targets = state.discovered_entities.get('ProjectionTarget', [])[:5]

            if targets and len(targets) >= 2:
                candidates.append(CandidateStep(
                    step_id='spatial_target_subregions',
                    step_type='spatial',
                    purpose='Map hierarchical subregions of major projection targets',
                    rationale='Fine-grained circuit mapping requires subregion specificity',
                    priority=6.5,
                    schema_path='Target_Region -[BELONGS_TO]-> Parent/Child',
                    expected_data='Hierarchical organization of target regions',
                    cypher_template="""
                    MATCH (t:Region)
                    WHERE t.acronym IN $targets
                    OPTIONAL MATCH (t)-[:BELONGS_TO]->(parent:Region)
                    OPTIONAL MATCH (child:Region)-[:BELONGS_TO]->(t)
                    RETURN t.acronym AS target,
                           t.name AS target_name,
                           parent.acronym AS parent_region,
                           collect(DISTINCT child.acronym)[0..10] AS child_regions
                    LIMIT 20
                    """,
                    parameters={'targets': targets},
                    depends_on=['proj_identify_targets']
                ))

        return candidates

    # ==================== LLM评估 ====================

    def _rank_steps_by_value(self,
                            candidates: List[CandidateStep],
                            question: str,
                            state_analysis: Dict) -> List[CandidateStep]:
        """
        使用LLM对候选步骤进行价值评估和排序

        考虑:
        1. 与问题意图的相关性
        2. 数据完整性需求
        3. 科学故事的连贯性
        """

        # 准备候选步骤摘要
        candidates_summary = []
        for i, c in enumerate(candidates):
            candidates_summary.append({
                'index': i,
                'purpose': c.purpose,
                'rationale': c.rationale,
                'type': c.step_type,
                'priority': c.priority
            })

        prompt = f"""You are planning the next analysis steps for neuroscience research.

**Original Question:** {question}

**Current State:**
- Has gene: {state_analysis['has_gene']}
- Has subclass: {state_analysis['has_subclass']}
- Has regions: {state_analysis['has_regions']} ({state_analysis['region_count']} found)
- Has morphology: {state_analysis['has_morphology']}
- Has projections: {state_analysis['has_projections']}
- Projection targets analyzed: {state_analysis['projection_targets_analyzed']}
- Missing modalities: {state_analysis['missing_modalities']}
- Steps completed: {state_analysis['depth_achieved']}

**Candidate Next Steps:**
{json.dumps(candidates_summary, indent=2)}

**Your Task:**
Rank these steps by SCIENTIFIC VALUE for:
1. Directly answering the question
2. Completing a coherent multi-modal story (molecular → spatial → morphological → projection → back to molecular)
3. Filling critical gaps in the analysis

**IMPORTANT:**
- If projections exist but targets NOT analyzed molecularly, prioritize "mol_target_composition" (closes the loop!)
- Balance breadth (covering modalities) vs depth (detailed analysis)
- Consider if step enables valuable downstream analysis

Return JSON:
{{
  "ranked_steps": [
    {{"index": 0, "score": 0.95, "reasoning": "..."}},
    ...
  ]
}}
"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert neuroscientist planning multi-modal analysis strategies."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)

            # 按排名重新排序
            ranked = []
            for item in result['ranked_steps']:
                idx = item['index']
                if idx < len(candidates):
                    candidates[idx].llm_score = item['score']
                    candidates[idx].llm_reasoning = item['reasoning']
                    ranked.append(candidates[idx])

            logger.info(f"   LLM ranked {len(ranked)} steps")
            return ranked

        except Exception as e:
            logger.error(f"LLM ranking failed: {e}")
            # Fallback: 按priority排序
            return sorted(candidates, key=lambda x: x.priority, reverse=True)

    def _llm_should_continue(self,
                            state: AnalysisState,
                            question: str,
                            analysis: Dict) -> bool:
        """
        使用LLM判断是否应该继续分析
        """

        prompt = f"""Should we continue the analysis or is it complete?

**Question:** {question}

**Target Depth:** {state.target_depth.value}

**Current State:**
- Steps executed: {len(state.executed_steps)}
- Gene found: {analysis['has_gene']}
- Regions identified: {analysis['has_regions']} ({analysis['region_count']})
- Morphology analyzed: {analysis['has_morphology']}
- Projections mapped: {analysis['has_projections']}
- Projection targets characterized: {analysis['projection_targets_analyzed']}
- Modalities covered: {', '.join(state.modalities_covered)}

**Recent steps:**
{json.dumps([s['purpose'] for s in state.executed_steps[-3:]], indent=2)}

**Decision criteria:**
- SHALLOW depth: Stop after 2 steps (basic answer)
- MEDIUM depth: Stop after 3-4 steps (standard multi-modal)
- DEEP depth: Stop after 5-6 steps (comprehensive with closed loop)

Should we continue?

Return JSON: {{"continue": true/false, "reason": "..."}}
"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You decide when scientific analysis is complete."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=200
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"   Continue: {result['continue']} - {result['reason']}")

            return result['continue']

        except Exception as e:
            logger.error(f"Continue decision failed: {e}")
            # Fallback: 简单规则
            max_steps = {
                AnalysisDepth.SHALLOW: 2,
                AnalysisDepth.MEDIUM: 4,
                AnalysisDepth.DEEP: 6
            }
            return len(state.executed_steps) < max_steps[state.target_depth]


# ==================== Utility Functions ====================

# 在原文件中找到这两个方法，替换为以下代码

def determine_analysis_depth(question: str) -> AnalysisDepth:
    """
    根据问题确定分析深度（优化版 v2.0）

    🔧 修复：
    - 更精确的shallow/medium/deep判断
    - 避免过度aggressive的深度分类
    - 确保simple问题不会被误判为deep
    """
    question_lower = question.lower()

    # ====== Priority 1: Explicit SHALLOW indicators ======
    shallow_patterns = [
        # 简单查询
        'what is', 'what are', 'define', 'definition of',

        # 名称查询
        'acronym', 'full name', 'abbreviation', 'stands for',
        'name of', 'called',

        # 快速查询
        'briefly', 'quick', 'simple', 'short', 'summarize',
        'in brief', 'concise',

        # 计数查询
        'how many', 'count', 'number of',
    ]

    if any(pattern in question_lower for pattern in shallow_patterns):
        logger.info(f"   Detected SHALLOW depth: explicit indicator found")
        return AnalysisDepth.SHALLOW

    # ====== Priority 2: Explicit DEEP indicators ======
    deep_patterns = [
        # 全面分析
        'comprehensive', 'comprehensive analysis', 'complete analysis',
        'detailed', 'detailed analysis', 'in-depth', 'thorough',

        # 分析动词
        'analyze', 'analysis', 'characterize', 'characterization',
        'investigate', 'examination', 'explore', 'exploration',

        # Profile patterns
        'profile', 'profiling',

        # "Tell me about" patterns
        'tell me about', 'tell me everything about',
        'explain about', 'describe in detail',
    ]

    for pattern in deep_patterns:
        if pattern in question_lower:
            logger.info(f"   Detected DEEP depth: '{pattern}'")
            return AnalysisDepth.DEEP

    # 特殊处理："about X"模式
    # "Tell me about Car3" → DEEP
    # "Information about Car3" → MEDIUM
    if ' about ' in question_lower:
        words_before_about = question_lower.split(' about ')[0].split()

        # 如果"about"前面少于5个词，且不是"information/data/facts"
        if len(words_before_about) <= 4:
            preceding_words = set(words_before_about[-2:]) if len(words_before_about) >= 2 else set(words_before_about)
            shallow_about_words = {'information', 'data', 'facts', 'details', 'some'}

            if not preceding_words & shallow_about_words:
                logger.info(f"   Detected DEEP depth: 'about X' pattern")
                return AnalysisDepth.DEEP

    # ====== Priority 3: MEDIUM indicators ======
    medium_patterns = [
        # 比较
        'compare', 'comparison', 'versus', 'vs ', 'vs.',
        'difference between', 'differences between', 'contrast',

        # 描述
        'describe', 'explain', 'what are the',

        # 列举
        'list', 'identify', 'find',
    ]

    if any(pattern in question_lower for pattern in medium_patterns):
        logger.info(f"   Detected MEDIUM depth")
        return AnalysisDepth.MEDIUM

    # ====== Default: MEDIUM (NOT DEEP!) ======
    # 🔧 关键修复：默认为MEDIUM，不是DEEP
    logger.info(f"   Default to MEDIUM depth")
    return AnalysisDepth.MEDIUM


def should_continue(self, state: AnalysisState, question: str) -> bool:
    """
    判断是否继续规划下一步（优化版 v2.0）

    🔧 修复：
    - 确保至少执行1步
    - 为shallow问题设置合理的最小步数
    - 为deep问题确保足够的步数
    """

    # 🔧 Fix: 确保至少执行1步
    if len(state.executed_steps) == 0:
        logger.info("   Continue: Must execute at least 1 step")
        return True

    # 检查当前步数
    current_steps = len(state.executed_steps)

    # 🔧 根据目标深度设置最小和最大步数
    if state.target_depth == AnalysisDepth.SHALLOW:
        min_steps = 1
        max_steps = 2
    elif state.target_depth == AnalysisDepth.MEDIUM:
        min_steps = 2
        max_steps = 4
    else:  # DEEP
        min_steps = 4
        max_steps = 8

    # 🔧 如果未达到最小步数，继续
    if current_steps < min_steps:
        logger.info(f"   Continue: {current_steps} < min_steps({min_steps}) for {state.target_depth.value}")
        return True

    # 如果超过最大步数，停止
    if current_steps >= max_steps:
        logger.info(f"   Stop: Reached max_steps({max_steps}) for {state.target_depth.value}")
        return False

    # 检查是否有待执行的候选步骤
    if not hasattr(state, 'candidate_steps') or not state.candidate_steps:
        logger.info("   Stop: No more candidate steps")
        return False

    # 检查是否所有必要的模态都已覆盖
    expected_modalities = self._get_expected_modalities(question)
    covered_modalities = set(state.modalities_covered)

    if expected_modalities and not expected_modalities.issubset(covered_modalities):
        missing = expected_modalities - covered_modalities
        logger.info(f"   Continue: Missing modalities: {missing}")
        return True

    # 🔧 对于DEEP问题，检查是否完成了闭环
    if state.target_depth == AnalysisDepth.DEEP:
        # 检查是否有projection步骤
        has_projection = any('projection' in s.get('purpose', '').lower()
                             for s in state.executed_steps)

        # 检查是否有target composition步骤
        has_target_composition = any(
            ('target' in s.get('purpose', '').lower() and
             'composition' in s.get('purpose', '').lower())
            for s in state.executed_steps
        )

        # 如果有projection但没有target composition，继续（尝试闭环）
        if has_projection and not has_target_composition:
            logger.info(f"   Continue: Attempting closed-loop (has projection but no target composition)")
            return True

    # 默认：如果在合理范围内且有候选步骤，继续
    logger.info(f"   Continue: Within reasonable range ({current_steps}/{max_steps})")
    return True


# ==================== Test ====================

if __name__ == "__main__":
    # 简单测试
    print("Testing AdaptivePlanner...")

    state = AnalysisState(
        discovered_entities={'GeneMarker': ['Car3']},
        target_depth=AnalysisDepth.DEEP
    )

    print(f"Depth for 'Tell me about Car3': {determine_analysis_depth('Tell me about Car3').value}")
    print(f"Depth for 'Comprehensive analysis of Car3': {determine_analysis_depth('Comprehensive analysis of Car3').value}")
    print(f"Depth for 'Briefly describe Car3': {determine_analysis_depth('Briefly describe Car3').value}")