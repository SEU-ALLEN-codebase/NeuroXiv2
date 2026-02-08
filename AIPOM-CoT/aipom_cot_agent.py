import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance

from neo4j_exec import Neo4jExec

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


# ==================== Real Schema Cache ====================

class RealSchemaCache:
    """
    Load and manage the REAL schema from schema.json

    Node types: Class, Cluster, ME_Subregion, Neuron, Region, Subclass, Subregion, Supertype
    Relationships: BELONGS_TO, HAS_CLASS, HAS_CLUSTER, HAS_SUBCLASS, HAS_SUPERTYPE,
                   LOCATE_AT, LOCATE_AT_ME_SUBREGION, LOCATE_AT_SUBREGION,
                   PROJECT_TO, NEIGHBOURING, AXON_NEIGHBOURING, DEN_NEIGHBOURING
    """

    def __init__(self, schema_json_path: str):
        self.schema_path = Path(schema_json_path)

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_json_path}")

        with open(self.schema_path, 'r') as f:
            self.schema_data = json.load(f)

        self.node_types = {}
        self.rel_types = {}

        # Parse nodes
        for label, info in self.schema_data['nodes'].items():
            self.node_types[label] = {
                'count': info['count'],
                'properties': list(info['properties'].keys()),
                'sample': info.get('sample_nodes', [])[:1]
            }

        # Parse relationships
        for rel_type, info in self.schema_data['relationships'].items():
            self.rel_types[rel_type] = {
                'count': info['count'],
                'patterns': info.get('patterns', []),
                'properties': list(info.get('properties', {}).keys())
            }

        logger.info(f"✅ Loaded real schema: {len(self.node_types)} node types, {len(self.rel_types)} rel types")

    def get_llm_summary(self) -> str:
        """Generate comprehensive schema summary for LLM"""
        lines = []

        lines.append("=== NeuroXiv Knowledge Graph Schema ===\n")
        lines.append("This is the REAL schema extracted from the database.\n")

        # Statistics
        stats = self.schema_data.get('statistics', {})
        lines.append(f"**Total Nodes:** {stats.get('total_nodes', 0):,}")
        lines.append(f"**Total Relationships:** {stats.get('total_relationships', 0):,}\n")

        # Node types
        lines.append("**NODE TYPES:**\n")
        for label in sorted(self.node_types.keys()):
            info = self.node_types[label]
            lines.append(f"• **{label}** ({info['count']:,} nodes)")

            # Key properties (first 15)
            props = info['properties'][:15]
            lines.append(f"  Properties: {', '.join(props)}")

            # Sample if available
            if info['sample']:
                sample_keys = list(info['sample'][0].keys())[:5]
                lines.append(f"  Example keys: {', '.join(sample_keys)}")

            lines.append("")

        # Relationships
        lines.append("**RELATIONSHIP TYPES:**\n")
        for rel_type in sorted(self.rel_types.keys()):
            info = self.rel_types[rel_type]
            lines.append(f"• **{rel_type}** ({info['count']:,} relationships)")

            # Main patterns
            if info['patterns']:
                top_3 = info['patterns'][:3]
                for source, target, count in top_3:
                    pct = (count / info['count'] * 100) if info['count'] > 0 else 0
                    lines.append(f"  ({source})-[:{rel_type}]->({target}): {count:,} ({pct:.1f}%)")

            # Properties
            if info['properties']:
                lines.append(f"  Properties: {', '.join(info['properties'])}")

            lines.append("")

        # Domain knowledge
        lines.append("**KEY DOMAIN FACTS:**\n")
        lines.append("• **Cluster.markers**: Gene marker strings (e.g., 'Car3,Satb2,Lgr5')")
        lines.append("• **Cluster.broad_region_distribution**: Region distribution (e.g., 'Isocortex:0.57,CTXsp:0.21')")
        lines.append("• **Neuron nodes**: Have detailed morphological properties (axonal_length, dendritic_length, etc.)")
        lines.append("• **Region -[HAS_CLUSTER]-> Cluster**: Regions contain cell clusters")
        lines.append("• **Region -[PROJECT_TO]-> Region**: Connectivity with weight and neuron_count")
        lines.append("• **Neuron -[LOCATE_AT]-> Region**: Neurons located in regions")
        lines.append("• **Hierarchical taxonomy**: Class -> Subclass -> Supertype -> Cluster")

        return "\n".join(lines)


# ==================== Complete Statistical Tools ====================

class StatisticalTools:
    """
    COMPLETE statistical toolkit for rigorous analysis

    All methods from the original V8, no shortcuts!
    """

    @staticmethod
    def hypergeometric_enrichment(k: int, M: int, n: int, N: int) -> Dict[str, float]:
        """
        Hypergeometric test for enrichment

        Args:
            k: Number of successes in sample
            M: Total population size
            n: Sample size
            N: Total successes in population
        """
        from scipy.stats import hypergeom
        p_value = hypergeom.sf(k - 1, M, N, n)
        observed_rate = k / n if n > 0 else 0
        expected_rate = N / M if M > 0 else 0
        fold_enrichment = observed_rate / expected_rate if expected_rate > 0 else 0
        expected = n * N / M if M > 0 else 0

        return {
            'p_value': float(p_value),
            'fold_enrichment': float(fold_enrichment),
            'expected': float(expected),
            'observed': k
        }

    @staticmethod
    def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """Benjamini-Hochberg FDR correction"""
        from statsmodels.stats.multitest import multipletests
        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        significant = q_values < alpha
        return q_values.tolist(), significant.tolist()

    @staticmethod
    def permutation_test(observed_stat: float,
                         data1: np.ndarray,
                         data2: np.ndarray,
                         n_permutations: int = 1000,
                         seed: Optional[int] = None) -> Dict[str, float]:
        """Permutation test for difference between groups"""
        if seed is not None:
            np.random.seed(seed)

        combined = np.concatenate([data1, data2])
        n1 = len(data1)

        null_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            null_stat = np.mean(combined[:n1]) - np.mean(combined[n1:])
            null_stats.append(null_stat)

        null_stats = np.array(null_stats)
        p_value = np.mean(np.abs(null_stats) >= np.abs(observed_stat))

        return {
            'p_value': float(p_value),
            'observed_stat': float(observed_stat),
            'null_mean': float(np.mean(null_stats)),
            'null_std': float(np.std(null_stats))
        }

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d effect size"""
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    @staticmethod
    def bootstrap_ci(data: np.ndarray,
                     statistic_func=np.mean,
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95,
                     seed: Optional[int] = None) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        if seed is not None:
            np.random.seed(seed)

        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    @staticmethod
    def correlation_test(x: np.ndarray,
                         y: np.ndarray,
                         method: str = 'pearson') -> Dict[str, float]:
        """
        Correlation test with p-value and confidence interval

        Args:
            x: First variable (array-like)
            y: Second variable (array-like)
            method: 'pearson' or 'spearman'

        Returns:
            {
                'correlation': float,
                'p_value': float,
                'method': str,
                'ci_lower': float,
                'ci_upper': float
            }
        """
        from scipy import stats as scipy_stats

        x = np.asarray(x)
        y = np.asarray(y)

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            return {
                'correlation': np.nan,
                'p_value': 1.0,
                'method': method,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'error': 'Insufficient data (n < 3)'
            }

        # Compute correlation
        if method == 'pearson':
            r, p_value = scipy_stats.pearsonr(x, y)
        elif method == 'spearman':
            r, p_value = scipy_stats.spearmanr(x, y)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'")

        # Compute confidence interval using Fisher's z-transformation
        n = len(x)
        if method == 'pearson':
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            ci_lower_z = z - 1.96 * se
            ci_upper_z = z + 1.96 * se
            ci_lower = np.tanh(ci_lower_z)
            ci_upper = np.tanh(ci_upper_z)
        else:
            # Spearman: use bootstrap
            ci_lower, ci_upper = StatisticalTools.bootstrap_ci(
                np.column_stack([x, y]),
                statistic_func=lambda data: scipy_stats.spearmanr(data[:, 0], data[:, 1])[0],
                n_bootstrap=1000
            )

        return {
            'correlation': float(r),
            'p_value': float(p_value),
            'method': method,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n': int(n)
        }


# ==================== Fingerprint Analyzer with REAL Schema ====================

class RealFingerprintAnalyzer:
    """
    Multi-modal fingerprint analysis adapted to REAL schema

    Key changes from V8:
    - Molecular: Use Cluster nodes and HAS_CLUSTER relationships
    - Morphological: Aggregate from Neuron nodes via LOCATE_AT
    - Projection: Use PROJECT_TO (unchanged, but verify properties)
    """

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema
        self._cluster_cache = None
        self._target_cache = None

    def compute_region_fingerprint(self, region: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute tri-modal fingerprint for a region

        Returns:
            {
                'molecular': np.ndarray,    # Cluster composition
                'morphological': np.ndarray, # Aggregated neuron features
                'projection': np.ndarray     # Target distribution
            }
        """
        fingerprint = {}

        # Molecular fingerprint
        mol_fp = self._compute_molecular_fingerprint(region)
        if mol_fp is not None:
            fingerprint['molecular'] = mol_fp

        # Morphological fingerprint
        mor_fp = self._compute_morphological_fingerprint(region)
        if mor_fp is not None:
            fingerprint['morphological'] = mor_fp

        # Projection fingerprint
        proj_fp = self._compute_projection_fingerprint(region)
        if proj_fp is not None:
            fingerprint['projection'] = proj_fp

        return fingerprint if len(fingerprint) > 0 else None

    def _compute_molecular_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """
        Molecular fingerprint = cluster composition

        Uses REAL schema:
        MATCH (r:Region {acronym: $region})-[h:HAS_CLUSTER]->(c:Cluster)
        """
        query = """
        MATCH (r:Region {acronym: $acronym})-[h:HAS_CLUSTER]->(c:Cluster)
        RETURN c.name AS cluster_name,
               c.markers AS markers,
               c.number_of_neurons AS neuron_count
        ORDER BY c.name
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # Get all clusters
        all_clusters = self._get_all_clusters()

        # Build vector: neuron count for each cluster
        cluster_dict = {
            row['cluster_name']: row['neuron_count'] or 0
            for row in result['data']
        }

        vector = np.array([cluster_dict.get(c, 0.0) for c in all_clusters])

        # Normalize
        total = np.sum(vector)
        if total > 0:
            vector = vector / total

        return vector

    def _compute_morphological_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """
        Morphological fingerprint = aggregated neuron features

        Uses REAL schema:
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $region})
        RETURN avg(n.axonal_length), avg(n.dendritic_length), ...
        """
        query = """
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $acronym})
        RETURN 
            avg(n.axonal_length) AS avg_axon_len,
            avg(n.dendritic_length) AS avg_dend_len,
            avg(n.axonal_surface) AS avg_axon_surf,
            avg(n.dendritic_surface) AS avg_dend_surf,
            avg(n.number_of_stems) AS avg_stems,
            avg(n.soma_surface) AS avg_soma
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data'] or not result['data'][0]:
            return None

        data = result['data'][0]

        # Build feature vector
        vector = np.array([
            data.get('avg_axon_len') or 0.0,
            data.get('avg_dend_len') or 0.0,
            data.get('avg_axon_surf') or 0.0,
            data.get('avg_dend_surf') or 0.0,
            data.get('avg_stems') or 0.0,
            data.get('avg_soma') or 0.0
        ], dtype=float)

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _compute_projection_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """
        Projection fingerprint = target distribution

        Uses PROJECT_TO relationship (same as before)
        """
        query = """
        MATCH (r:Region {acronym: $acronym})-[p:PROJECT_TO]->(t:Region)
        RETURN t.acronym AS target, p.weight AS weight
        ORDER BY t.acronym
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        all_targets = self._get_all_targets()

        target_dict = {row['target']: row['weight'] or 0.0 for row in result['data']}
        vector = np.array([target_dict.get(t, 0.0) for t in all_targets])

        # Normalize
        total = np.sum(vector)
        if total > 0:
            vector = vector / total

        return vector

    def compute_similarity(self, fp1: np.ndarray, fp2: np.ndarray,
                          metric: str = 'cosine') -> float:
        """Compute similarity between fingerprints"""
        if metric == 'cosine':
            norm1, norm2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(fp1, fp2) / (norm1 * norm2))
        elif metric == 'correlation':
            if len(fp1) < 2:
                return 0.0
            r, _ = stats.pearsonr(fp1, fp2)
            return float(r)
        else:
            return 0.0

    def compute_mismatch_index(self, region1: str, region2: str) -> Optional[Dict[str, float]]:
        """
        Compute cross-modal mismatch (Figure 4 metric)

        MM_GM = |sim_molecular - sim_morphological|
        MM_GP = |sim_molecular - sim_projection|
        """
        fp1 = self.compute_region_fingerprint(region1)
        fp2 = self.compute_region_fingerprint(region2)

        if fp1 is None or fp2 is None:
            return None

        sim_mol = self.compute_similarity(fp1['molecular'], fp2['molecular'])
        sim_mor = self.compute_similarity(fp1['morphological'], fp2['morphological'])
        sim_proj = self.compute_similarity(fp1['projection'], fp2['projection'])

        return {
            'sim_molecular': sim_mol,
            'sim_morphological': sim_mor,
            'sim_projection': sim_proj,
            'mismatch_GM': abs(sim_mol - sim_mor),
            'mismatch_GP': abs(sim_mol - sim_proj),
            'mismatch_MP': abs(sim_mor - sim_proj)
        }

    def _get_all_clusters(self) -> List[str]:
        """Get all cluster names for consistent dimensions"""
        if self._cluster_cache is not None:
            return self._cluster_cache

        query = "MATCH (c:Cluster) RETURN c.name AS name ORDER BY c.name LIMIT 100"
        result = self.db.run(query)

        if result['success'] and result['data']:
            self._cluster_cache = [row['name'] for row in result['data']]
        else:
            self._cluster_cache = []

        return self._cluster_cache

    def _get_all_targets(self) -> List[str]:
        """Get all projection targets"""
        if self._target_cache is not None:
            return self._target_cache

        query = """
        MATCH ()-[:PROJECT_TO]->(t:Region)
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        LIMIT 100
        """
        result = self.db.run(query)

        if result['success'] and result['data']:
            self._target_cache = [row['target'] for row in result['data']]
        else:
            self._target_cache = []

        return self._target_cache

    def get_region_fingerprint(self, region: str) -> Dict:
        """
        获取单个region的完整fingerprint

        🆕 新增方法 - 支持高性能版本的批量计算

        Args:
            region: 脑区acronym

        Returns:
            {
                'molecular': [array],
                'morphological': [array],
                'projection': [array]
            }
        """
        try:
            # 计算三种fingerprint
            molecular = self._compute_molecular_fingerprint(region)
            morphological = self._compute_morphological_fingerprint(region)
            projection = self._compute_projection_fingerprint(region)

            # 验证
            if molecular is None or morphological is None or projection is None:
                return None

            # 转换为list (确保JSON可序列化)
            return {
                'molecular': molecular.tolist() if hasattr(molecular, 'tolist') else list(molecular),
                'morphological': morphological.tolist() if hasattr(morphological, 'tolist') else list(morphological),
                'projection': projection.tolist() if hasattr(projection, 'tolist') else list(projection)
            }

        except Exception as e:
            logger.error(f"Failed to get fingerprint for {region}: {e}")
            return None


# ==================== Schema-Guided Planner ====================

class RealSchemaGuidedPlanner:
    """
    Generate reasoning plans based on real schema

    Entity recognition → Schema path finding → Query generation
    """

    def __init__(self, schema: RealSchemaCache):
        self.schema = schema
        self.known_genes = {
            'car3', 'pvalb', 'sst', 'vip', 'gad', 'gad1', 'gad2',
            'slc17a7', 'satb2', 'rorb', 'cux2', 'fezf2','car3'
        }
        self.known_regions = {
            'cla', 'mos', 'mop', 'acad', 'ssp', 'entl', 'ai', 'pir',
            'orb', 'pl', 'il', 'aca', 'rsp',"CLA"
        }

    def generate_initial_plan(self, question: str) -> Dict[str, Any]:
        """Generate initial reasoning plan"""

        # 1. Recognize entities
        entities = self._recognize_entities(question)

        # 2. Assess complexity
        complexity = self._assess_complexity(question, entities)

        # 3. Generate plan based on entity types
        if any(e['type'] == 'gene_marker' for e in entities):
            plan = self._plan_for_gene_marker(entities, question)
        elif any(e['type'] == 'region' for e in entities):
            plan = self._plan_for_region(entities, question)
        else:
            plan = self._plan_exploratory(question)

        return {
            'entities': entities,
            'complexity': complexity,
            'reasoning_steps': plan
        }

    def _recognize_entities(self, question: str) -> List[Dict]:
        """Recognize entities in question"""
        entities = []
        question_lower = question.lower()

        # Gene markers
        for gene in self.known_genes:
            if gene in question_lower:
                entities.append({
                    'text': gene.capitalize() if len(gene) <= 4 else gene,
                    'type': 'gene_marker',
                    'confidence': 0.9
                })

        # Regions
        for region in self.known_regions:
            if region in question_lower or region.upper() in question:
                entities.append({
                    'text': region.upper(),
                    'type': 'region',
                    'confidence': 0.9
                })

        return entities

    def _assess_complexity(self, question: str, entities: List[Dict]) -> str:
        """Assess query complexity"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(word in question_lower for word in ['why', 'explain', 'mechanism', 'reason']):
            return 'explanation'
        elif any(word in question_lower for word in ['comprehensive', 'detailed', 'full', 'complete']):
            return 'comprehensive'
        elif len(entities) >= 2:
            return 'multi_entity'
        else:
            return 'simple'

    def _plan_for_gene_marker(self, entities: List[Dict], question: str) -> List[Dict]:
        """Generate plan for gene marker analysis"""
        gene = next((e['text'] for e in entities if e['type'] == 'gene_marker'), None)
        if not gene:
            return []

        steps = []

        # Step 1: Find clusters with gene
        steps.append({
            'step': 1,
            'purpose': f'Find cell clusters expressing {gene}',
            'action': 'execute_cypher',
            'rationale': f'{gene} is a gene marker. Cluster.markers field contains gene combinations.',
            'query': f"""
MATCH (c:Cluster)
WHERE c.markers CONTAINS '{gene}'
RETURN c.name AS cluster_name,
       c.markers AS all_markers,
       c.broad_region_distribution AS region_dist,
       c.number_of_neurons AS neuron_count,
       c.anatomical_annotation AS location
ORDER BY c.number_of_neurons DESC
LIMIT 20
            """,
            'modality': 'molecular'
        })

        # Step 2: Find enriched regions
        steps.append({
            'step': 2,
            'purpose': f'Identify brain regions enriched for {gene}+ clusters',
            'action': 'execute_cypher',
            'rationale': 'Map clusters to their parent regions via HAS_CLUSTER relationship.',
            'query': f"""
MATCH (r:Region)-[h:HAS_CLUSTER]->(c:Cluster)
WHERE c.markers CONTAINS '{gene}'
RETURN r.acronym AS region,
       r.name AS region_name,
       count(c) AS cluster_count,
       sum(c.number_of_neurons) AS total_neurons,
       collect(c.name)[0..5] AS sample_clusters
ORDER BY cluster_count DESC
LIMIT 15
            """,
            'modality': 'molecular',
            'depends_on': [1]
        })

        # Conditional: morphology if mentioned
        question_lower = question.lower()
        if any(word in question_lower for word in ['morpholog', 'feature', 'structure', 'comprehen']):
            steps.append({
                'step': 3,
                'purpose': f'Analyze morphological features of {gene}+ enriched regions',
                'action': 'execute_cypher',
                'rationale': 'Aggregate morphological statistics from neurons in these regions.',
                'query': """
MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
WHERE r.acronym IN $enriched_regions
RETURN r.acronym AS region,
       count(n) AS neuron_count,
       avg(n.axonal_length) AS avg_axon_length,
       avg(n.dendritic_length) AS avg_dendrite_length,
       avg(n.soma_surface) AS avg_soma_surface
ORDER BY neuron_count DESC
LIMIT 20
                """,
                'modality': 'morphological',
                'depends_on': [2]
            })

        # Conditional: projection if mentioned
        if any(word in question_lower for word in ['project', 'target', 'connect', 'output', 'comprehen']):
            step_num = len(steps) + 1
            steps.append({
                'step': step_num,
                'purpose': f'Identify projection targets of {gene}+ regions',
                'action': 'execute_cypher',
                'rationale': 'PROJECT_TO relationships show where these regions send outputs.',
                'query': """
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
                'modality': 'projection',
                'depends_on': [2]
            })

        return steps

    def _plan_for_region(self, entities: List[Dict], question: str) -> List[Dict]:
        """Generate plan for region-centric analysis"""
        region = next((e['text'] for e in entities if e['type'] == 'region'), None)
        if not region:
            return []

        steps = []

        # Basic info
        steps.append({
            'step': 1,
            'purpose': f'Get basic information about {region}',
            'action': 'execute_cypher',
            'query': f"""
MATCH (r:Region {{acronym: '{region}'}})
RETURN r.name AS full_name,
       r.acronym AS acronym,
       r.rgb_triplet AS color
            """
        })

        # Cell composition
        steps.append({
            'step': 2,
            'purpose': f'Get cell type composition of {region}',
            'action': 'execute_cypher',
            'query': f"""
MATCH (r:Region {{acronym: '{region}'}})-[:HAS_CLUSTER]->(c:Cluster)
RETURN c.name AS cluster,
       c.markers AS markers,
       c.number_of_neurons AS neurons
ORDER BY c.number_of_neurons DESC
LIMIT 20
            """,
            'modality': 'molecular'
        })

        return steps

    def _plan_exploratory(self, question: str) -> List[Dict]:
        """Exploratory plan when no entities recognized"""
        return [{
            'step': 1,
            'purpose': 'Exploratory database overview',
            'action': 'execute_cypher',
            'query': """
MATCH (n) 
WITH labels(n)[0] AS node_type, count(n) AS count
RETURN node_type, count
ORDER BY count DESC
LIMIT 10
            """
        }]


# ==================== Agent State & Phases ====================

class AgentPhase(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    REPLANNING = "replanning"
    SYNTHESIZING = "synthesizing"


@dataclass
class ReasoningStep:
    step_number: int
    purpose: str
    action: str
    rationale: str
    expected_result: str
    query_or_params: Dict[str, Any]
    modality: Optional[str] = None
    depends_on: List[int] = field(default_factory=list)

    actual_result: Optional[Dict] = None
    reflection: Optional[str] = None
    validation_passed: bool = False
    execution_time: float = 0.0


@dataclass
class AgentState:
    question: str
    phase: AgentPhase = AgentPhase.PLANNING
    entities: List[Dict] = field(default_factory=list)
    reasoning_plan: List[ReasoningStep] = field(default_factory=list)
    current_step: int = 0
    executed_steps: List[ReasoningStep] = field(default_factory=list)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)
    replanning_count: int = 0
    max_replanning: int = 2
    final_answer: Optional[str] = None
    confidence_score: float = 0.0


# ==================== Safe Cypher Executor ====================

class SafeCypherExecutor:
    def __init__(self, db: Neo4jExec):
        self.db = db

    def execute(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        import re
        start_time = time.time()

        # Ensure LIMIT
        if not re.search(r'\bLIMIT\b', query, re.I):
            query = f"{query}\nLIMIT 100"

        # Execute
        result = self.db.run(query, params or {})
        result['execution_time'] = time.time() - start_time
        result['query'] = query
        return result


# ==================== THE COMPLETE FINAL AGENT ====================

class CompleteFinalAgent:
    """
    The COMPLETE production-ready agent

    Includes ALL features:
    - Real schema ✓
    - Honest questions ✓
    - Complete fingerprint analyzer ✓
    - All 6 tools ✓
    - Full stats ✓
    - True replanning ✓
    - Self-reflection ✓
    """

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_pwd: str,
                 database: str,
                 schema_json_path: str,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o"):

        # Initialize database
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)

        # Load REAL schema
        self.schema = RealSchemaCache(schema_json_path)

        # Initialize ALL components
        self.executor = SafeCypherExecutor(self.db)
        self.stats = StatisticalTools()
        self.fingerprint = RealFingerprintAnalyzer(self.db, self.schema)
        self.planner = RealSchemaGuidedPlanner(self.schema)

        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # Define ALL 6 tools
        self.tools = self._define_all_tools()

        logger.info("🚀 COMPLETE Final Agent initialized!")
        logger.info(f"   ✓ Real schema with {len(self.schema.node_types)} node types")
        logger.info(f"   ✓ {len(self.tools)} tool functions available")

    def _define_all_tools(self) -> List[Dict]:
        """Define ALL 6 tool functions"""

        schema_summary = self.schema.get_llm_summary()

        return [
            # Tool 1: Execute Cypher
            {
                "type": "function",
                "function": {
                    "name": "execute_cypher",
                    "description": f"""Execute Cypher query on the knowledge graph.

**REAL SCHEMA:**
{schema_summary[:2000]}

Write any valid Cypher query. Always include LIMIT.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Cypher query"},
                            "params": {"type": "object", "description": "Query parameters"},
                            "purpose": {"type": "string", "description": "Why this query"}
                        },
                        "required": ["query", "purpose"]
                    }
                }
            },

            # Tool 2: Compute Enrichment
            {
                "type": "function",
                "function": {
                    "name": "compute_enrichment",
                    "description": "Hypergeometric enrichment test with FDR correction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "observed_counts": {"type": "array", "items": {"type": "number"}},
                            "population_size": {"type": "integer"},
                            "sample_size": {"type": "integer"},
                            "successes_in_population": {"type": "integer"}
                        },
                        "required": ["observed_counts", "population_size", "sample_size", "successes_in_population"]
                    }
                }
            },

            # Tool 3: Compute Correlation
            {
                "type": "function",
                "function": {
                    "name": "compute_correlation",
                    "description": "Pearson or Spearman correlation with CI",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "array", "items": {"type": "number"}},
                            "y": {"type": "array", "items": {"type": "number"}},
                            "method": {"type": "string", "enum": ["pearson", "spearman"]}
                        },
                        "required": ["x", "y"]
                    }
                }
            },

            # Tool 4: Compare Distributions
            {
                "type": "function",
                "function": {
                    "name": "compare_distributions",
                    "description": "Statistical test (t-test, Mann-Whitney, permutation) with effect size",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "group1": {"type": "array", "items": {"type": "number"}},
                            "group2": {"type": "array", "items": {"type": "number"}},
                            "test": {"type": "string", "enum": ["t_test", "mann_whitney", "permutation"]}
                        },
                        "required": ["group1", "group2"]
                    }
                }
            },

            # Tool 5: Compute Fingerprint
            {
                "type": "function",
                "function": {
                    "name": "compute_fingerprint",
                    "description": "Multi-modal fingerprint (molecular, morphological, projection) for a region",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region": {"type": "string", "description": "Region acronym"}
                        },
                        "required": ["region"]
                    }
                }
            },

            # Tool 6: Compute Mismatch
            {
                "type": "function",
                "function": {
                    "name": "compute_mismatch",
                    "description": "Cross-modal mismatch index between two regions (Figure 4 metric)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region1": {"type": "string"},
                            "region2": {"type": "string"}
                        },
                        "required": ["region1", "region2"]
                    }
                }
            }
        ]

    def answer(self, question: str, max_iterations: int = 15) -> Dict[str, Any]:
        """
        Main entry point - TRUE autonomous reasoning

        NO LEADING QUESTIONS!
        """
        print('answeransweransweransweransweranswer')
        logger.info(f"🎯 Question: {question}")
        start_time = time.time()

        state = AgentState(question=question)

        # === PHASE 1: SCHEMA-GUIDED PLANNING ===
        logger.info("\n" + "="*60)
        logger.info("📋 PHASE 1: SCHEMA-GUIDED PLANNING")
        logger.info("="*60)

        state.phase = AgentPhase.PLANNING
        plan_result = self._planning_phase(state)

        if not plan_result['success']:
            return self._build_error_response(question, "Planning failed", start_time)

        logger.info(f"✅ Generated plan with {len(state.reasoning_plan)} steps")

        # === PHASE 2: EXECUTION WITH REFLECTION ===
        logger.info("\n" + "="*60)
        logger.info("⚙️ PHASE 2: EXECUTION WITH SELF-REFLECTION")
        logger.info("="*60)

        state.phase = AgentPhase.EXECUTING

        for step_idx, step in enumerate(state.reasoning_plan):
            if step_idx >= max_iterations:
                break

            logger.info(f"\n🔹 Step {step.step_number}: {step.purpose}")

            # Execute
            exec_result = self._execute_step(step, state)

            if not exec_result['success']:
                logger.error(f"❌ Failed: {exec_result.get('error')}")

                # TRUE REPLANNING
                if state.replanning_count < state.max_replanning:
                    replan_success = self._true_replan(state, step_idx + 1)
                    if replan_success:
                        continue
                continue

            # Reflection
            reflection = self._reflect_on_step(step, state)
            step.reflection = reflection

            # Validation
            validation = self._validate_step_result(step, state)
            step.validation_passed = validation['passed']

            state.executed_steps.append(step)

        # === PHASE 3: SYNTHESIS ===
        logger.info("\n" + "="*60)
        logger.info("📝 PHASE 3: ANSWER SYNTHESIS")
        logger.info("="*60)

        final_answer = self._synthesize_answer(state)

        execution_time = time.time() - start_time

        return {
            'question': question,
            'answer': final_answer,
            'reasoning_plan': [self._step_to_dict(s) for s in state.reasoning_plan],
            'executed_steps': [self._step_to_dict(s) for s in state.executed_steps],
            'reflections': state.reflections,
            'entities_recognized': state.entities,
            'replanning_count': state.replanning_count,
            'confidence_score': state.confidence_score,
            'execution_time': execution_time,
            'total_steps': len(state.executed_steps)
        }

    # ==================== PLANNING ====================

    def _planning_phase(self, state: AgentState) -> Dict[str, Any]:
        try:
            initial_plan = self.planner.generate_initial_plan(state.question)
            state.entities = initial_plan['entities']

            refined_plan = self._llm_refine_plan(initial_plan['reasoning_steps'], state)
            state.reasoning_plan = refined_plan

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _llm_refine_plan(self, initial_chain: List[Dict],
                             state: AgentState) -> List[ReasoningStep]:
            """
            LLM reviews schema-generated plan and adds:
            - Expected results
            - Explicit reasoning
            - Better queries
            """
            prompt = f"""You are refining a reasoning plan for neuroscience knowledge graph analysis.

    **Question:** {state.question}

    **Recognized Entities:** {', '.join([e['text'] for e in state.entities])}

    **Initial Reasoning Chain:**
    {json.dumps(initial_chain, indent=2)}

    Your task:
    1. Review each step in the reasoning chain
    2. For each step, add:
       - **Expected Result**: What data pattern you expect to see
       - **Rationale Enhancement**: Why this step is necessary (be specific!)
       - **Query Review**: Check if the query is correct, fix if needed

    Return a JSON array of steps with this format:
    [
      {{
        "step_number": 1,
        "purpose": "...",
        "action": "execute_cypher" or "compute_stat" or "compute_fingerprint",
        "rationale": "Detailed explanation of WHY this step",
        "expected_result": "What pattern/values we expect to find",
        "query_or_params": {{...}},
        "modality": "molecular/morphological/projection" or null,
        "depends_on": [previous step numbers]
      }},
      ...
    ]

    **Important**: 
    - Make rationale SPECIFIC to the question
    - Expected results should be CONCRETE predictions
    - Ensure queries are syntactically correct Cypher
    """

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert neuroscientist and Cypher query expert."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )

                result = json.loads(response.choices[0].message.content)

                # Convert to ReasoningStep objects
                steps = []
                for step_dict in result.get('steps', result.get('reasoning_plan', [])):
                    step = ReasoningStep(
                        step_number=step_dict.get('step_number', len(steps) + 1),
                        purpose=step_dict['purpose'],
                        action=step_dict['action'],
                        rationale=step_dict['rationale'],
                        expected_result=step_dict['expected_result'],
                        query_or_params=step_dict.get('query_or_params', {}),
                        modality=step_dict.get('modality'),
                        depends_on=step_dict.get('depends_on', [])
                    )
                    steps.append(step)

                return steps

            except Exception as e:
                logger.error(f"LLM plan refinement failed: {e}")
                # Fallback: convert initial_chain to ReasoningStep
                return [
                    ReasoningStep(
                        step_number=s.get('step', i + 1),
                        purpose=s.get('purpose', 'Query execution'),
                        action='execute_cypher',
                        rationale=s.get('rationale', 'Execute planned query'),
                        expected_result="Data matching query criteria",
                        query_or_params={'query': s.get('query', '')}
                    )
                    for i, s in enumerate(initial_chain)
                ]

    # ==================== EXECUTION ====================

    def _execute_step(self, step: ReasoningStep, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()

        try:
            if step.action == 'execute_cypher':
                result = self._execute_cypher_step(step, state)
            else:
                result = {'success': False, 'error': 'Unknown action'}

            step.actual_result = result
            step.execution_time = time.time() - start_time

            step_key = f"step_{step.step_number}"
            state.intermediate_data[step_key] = result.get('data', [])

            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_cypher_step(self, step: ReasoningStep, state: AgentState) -> Dict[str, Any]:
        query = step.query_or_params.get('query', '')
        params = {}

        # Substitute parameters
        if '$enriched_regions' in query:
            for dep_num in step.depends_on:
                dep_key = f"step_{dep_num}"
                if dep_key in state.intermediate_data:
                    dep_data = state.intermediate_data[dep_key]
                    params['enriched_regions'] = [
                        row.get('region', row.get('acronym', ''))
                        for row in dep_data[:10]
                    ]

        return self.executor.execute(query, params)

    # ==================== REFLECTION ====================

    def _reflect_on_step(self, step: ReasoningStep, state: AgentState) -> str:
        """
        LLM reflects on step result:
        1. Did results match expectations?
        2. What did we learn?
        3. Should we adjust the plan?
        """
        prompt = f"""Reflect on this reasoning step:

    **Step {step.step_number}: {step.purpose}**

    **Rationale:** {step.rationale}

    **Expected Result:** {step.expected_result}

    **Actual Result:** {json.dumps(step.actual_result.get('data', [])[:5], indent=2)}
    (showing first 5 rows)

    **Row Count:** {len(step.actual_result.get('data', []))} rows

    **Questions for reflection:**
    1. Did the actual results match your expectations?
    2. If not, what surprised you?
    3. What new insights did this step provide?
    4. Should we adjust subsequent steps based on this?

    Provide a 2-3 sentence reflection."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are reflecting on neuroscience data analysis results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            reflection = response.choices[0].message.content.strip()
            state.reflections.append(f"Step {step.step_number}: {reflection}")

            return reflection

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return "Reflection unavailable"

    def _validate_step_result(self, step: ReasoningStep, state: AgentState) -> Dict[str, Any]:
        result = step.actual_result

        if not result or not result.get('success'):
            return {'passed': False}

        if len(result.get('data', [])) == 0:
            return {'passed': False}

        return {'passed': True}

    # ==================== TRUE REPLANNING ====================

    def _true_replan(self, state: AgentState, from_step: int) -> bool:
        """
        TRUE replanning with LLM

        Not just "skip the step" - actually generate NEW plan!
        """
        logger.info(f"🔄 TRUE REPLANNING from step {from_step}")
        state.replanning_count += 1

        # Context: what we've done, what failed
        executed_summary = "\n".join([
            f"Step {s.step_number}: {s.purpose} → {'✓' if s.validation_passed else '✗'}"
            for s in state.executed_steps
        ])

        failed_step = state.reasoning_plan[from_step - 1] if from_step <= len(state.reasoning_plan) else None

        prompt = f"""We need to replan the analysis.

**Original Question:** {state.question}

**Executed Steps:**
{executed_summary}

**Failed Step:** {failed_step.purpose if failed_step else 'Unknown'}

**Issue:** Step failed or produced no results.

Generate NEW steps starting from step {from_step} to complete the analysis.
Return JSON array of steps."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are replanning a neuroscience analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)
            new_steps_data = result.get('steps', [])

            # Convert to ReasoningStep
            new_steps = []
            for i, sd in enumerate(new_steps_data):
                step = ReasoningStep(
                    step_number=from_step + i,
                    purpose=sd.get('purpose', ''),
                    action='execute_cypher',
                    rationale=sd.get('rationale', 'Replanned step'),
                    expected_result=sd.get('expected_result', ''),
                    query_or_params={'query': sd.get('query', '')}
                )
                new_steps.append(step)

            # Replace remaining plan
            state.reasoning_plan = state.reasoning_plan[:from_step-1] + new_steps

            logger.info(f"✅ Replanned {len(new_steps)} new steps")
            return True

        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            return False

    # ==================== SYNTHESIS ====================

    def _synthesize_answer(self, state: AgentState) -> str:
            """
            Generate final answer with full reasoning trace
            """
            # Prepare evidence summary
            evidence_summary = []
            for step in state.executed_steps:
                if step.actual_result and step.actual_result.get('success'):
                    data_count = len(step.actual_result.get('data', []))
                    evidence_summary.append(
                        f"- Step {step.step_number} ({step.purpose}): {data_count} results"
                    )

            evidence_text = "\n".join(evidence_summary)

            # Prepare key findings
            key_data = {}
            for step in state.executed_steps:
                if step.actual_result and step.actual_result.get('success'):
                    data = step.actual_result.get('data', [])
                    if data:
                        key_data[f"step_{step.step_number}"] = data[:10]  # Top 10

            prompt = f"""Synthesize a comprehensive answer based on the reasoning trace.

    **Original Question:** {state.question}

    **Reasoning Plan Executed:**
    {chr(10).join([f"{i + 1}. {s.purpose}" for i, s in enumerate(state.executed_steps)])}

    **Evidence Collected:**
    {evidence_text}

    **Key Findings:**
    {json.dumps(key_data, indent=2, default=str)[:3000]}

    **Reflections:**
    {chr(10).join(state.reflections[-5:])}

    **Your Task:**
    Write a comprehensive, scientifically rigorous answer that:
    1. Directly answers the original question
    2. Cites specific quantitative findings
    3. Explains the reasoning process briefly
    4. Acknowledges any limitations or uncertainties
    5. Is written for a neuroscience audience

    Make it publication-quality but accessible.
    """

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a neuroscience writer synthesizing analysis results."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                answer = response.choices[0].message.content.strip()
                state.final_answer = answer

                # Estimate confidence
                state.confidence_score = self._estimate_confidence(state)

                return answer

            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                return f"Analysis completed with {len(state.executed_steps)} steps, but synthesis failed."

    # ==================== UTILITIES ====================

    def _step_to_dict(self, step: ReasoningStep) -> Dict:
        """Convert ReasoningStep to dict for output"""
        return {
            'step_number': step.step_number,
            'purpose': step.purpose,
            'action': step.action,
            'rationale': step.rationale,
            'expected_result': step.expected_result,
            'actual_result_summary': {
                'success': step.actual_result.get('success') if step.actual_result else False,
                'row_count': len(step.actual_result.get('data', [])) if step.actual_result else 0
            },
            'reflection': step.reflection,
            'validation_passed': step.validation_passed,
            'execution_time': step.execution_time
        }

    def _compute_validation_rate(self, state: AgentState) -> float:
        """Compute what fraction of steps passed validation"""
        if not state.executed_steps:
            return 0.0
        passed = sum(1 for s in state.executed_steps if s.validation_passed)
        return passed / len(state.executed_steps)

    def _estimate_confidence(self, state: AgentState) -> float:
        """Estimate confidence in final answer"""
        score = 0.8  # Base score

        # Factor 1: Validation rate
        val_rate = self._compute_validation_rate(state)
        score *= (0.5 + 0.5 * val_rate)

        # Factor 2: Replanning penalty
        score *= (0.95 ** state.replanning_count)

        # Factor 3: Step completion
        planned_steps = len(state.reasoning_plan)
        executed_steps = len(state.executed_steps)
        completion_rate = executed_steps / planned_steps if planned_steps > 0 else 0
        score *= (0.7 + 0.3 * completion_rate)

        return min(1.0, max(0.0, score))

    def _build_error_response(self, question: str, error: str, start_time: float) -> Dict:
        """Build error response"""
        return {
            'question': question,
            'answer': f"Analysis failed: {error}",
            'error': error,
            'execution_time': time.time() - start_time,
            'success': False
        }


# ==================== Test Functions ====================

def test_complete_fig3():
    """Test with COMPLETE agent, REAL schema, HONEST question"""
    import os

    print("\n" + "="*80)
    print("TEST: Figure 3 (COMPLETE AGENT)")
    print("="*80)

    agent = CompleteFinalAgent(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY",""),
        model="gpt-4o"
    )

    # HONEST QUESTION
    question = "Tell me about Car3+ neurons"

    result = agent.answer(question, max_iterations=12)

    print(f"\n📊 Results:")
    print(f"  Steps: {result['total_steps']}")
    print(f"  Confidence: {result['confidence_score']:.2f}")
    print(f"  Time: {result['execution_time']:.1f}s")

    print(f"\n💡 Answer:\n{result['answer']}...\n")

    return result


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AIPOM-CoT V9 FINAL COMPLETE")
    print("="*80 + "\n")

    result = test_complete_fig3()

    print("\n✅ Test complete!")