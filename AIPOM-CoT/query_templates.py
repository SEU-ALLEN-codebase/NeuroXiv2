from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from string import Template


@dataclass
class QueryTemplate:
    """A reusable KG query template with reasoning metadata."""

    name: str
    purpose: str
    modality: str  # 'molecular' | 'morphological' | 'projection' | 'spatial' | 'meta'
    cypher: str
    parameters: Dict[str, str] = field(default_factory=dict)  # param_name -> description
    expected_rows: str = "variable"  # "1", "N", "0-N", "variable"
    can_return_empty: bool = True
    reasoning_hint: str = ""  # Why/when to use this query

    def render(self, **kwargs) -> str:
        """Render the Cypher query with parameters."""
        # Replace $PARAM style placeholders
        result = self.cypher
        for key, value in kwargs.items():
            if isinstance(value, str):
                result = result.replace(f'${key}', f"'{value}'")
            else:
                result = result.replace(f'${key}', str(value))
        return result

    def get_neo4j_params(self, **kwargs) -> Dict[str, Any]:
        """Get parameters in Neo4j format (for parameterized queries)."""
        return kwargs


# =============================================================================
# META QUERIES - Schema and connectivity
# =============================================================================

Q_CHECK_CONNECTIVITY = QueryTemplate(
    name='check_connectivity',
    purpose='Verify Neo4j connection is alive',
    modality='meta',
    cypher='RETURN 1 AS ok',
    expected_rows='1',
    can_return_empty=False,
    reasoning_hint='Always run first to validate database connectivity'
)

Q_GET_SCHEMA_LABELS = QueryTemplate(
    name='get_schema_labels',
    purpose='Discover all node labels in the KG',
    modality='meta',
    cypher='CALL db.labels() YIELD label RETURN label ORDER BY label',
    expected_rows='variable',
    reasoning_hint='Use to understand what entity types exist before querying'
)

Q_GET_SCHEMA_RELATIONSHIPS = QueryTemplate(
    name='get_schema_relationships',
    purpose='Discover all relationship types in the KG',
    modality='meta',
    cypher='CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType',
    expected_rows='variable',
    reasoning_hint='Use to understand how entities are connected'
)

# =============================================================================
# MOLECULAR QUERIES - Subclass, gene markers, enrichment
# =============================================================================

Q_GET_ALL_SUBCLASSES = QueryTemplate(
    name='get_all_subclasses',
    purpose='Get all transcriptomic subclasses linked to regions',
    modality='molecular',
    cypher='''
        MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass)
        RETURN DISTINCT sc.name AS subclass_name, sc.markers AS markers
        ORDER BY sc.name
    ''',
    expected_rows='variable',
    reasoning_hint='Use to establish the feature space for molecular fingerprints'
)

Q_GET_SUBCLASS_BY_GENE = QueryTemplate(
    name='get_subclass_by_gene',
    purpose='Find transcriptomic subclass(es) marked by a specific gene',
    modality='molecular',
    cypher='''
        MATCH (sc:Subclass)
        WHERE sc.name CONTAINS $GENE OR sc.markers CONTAINS $GENE
        RETURN sc.name AS subclass_name, sc.markers AS markers, elementId(sc) AS subclass_eid
        ORDER BY sc.name
    ''',
    parameters={'GENE': 'Gene marker name (e.g., Car3, Pvalb)'},
    expected_rows='1-N',
    reasoning_hint='First step in gene-centric analysis - identifies the molecular identity'
)

Q_GET_REGION_ENRICHMENT_FOR_SUBCLASS = QueryTemplate(
    name='get_region_enrichment_for_subclass',
    purpose='Get pct_cells of a subclass across ALL regions (for enrichment bar chart)',
    modality='molecular',
    cypher='''
        MATCH (r:Region)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        WHERE sc.name = $SUBCLASS_NAME
        RETURN r.acronym AS region_name, hs.pct_cells AS pct_cells, hs.rank AS rank
        ORDER BY hs.pct_cells DESC
    ''',
    parameters={'SUBCLASS_NAME': 'Full subclass name (e.g., "001 CLA-EPd-CTX Car3 Glut")'},
    expected_rows='variable',
    reasoning_hint='Use after identifying subclass to find which regions are enriched'
)

Q_GET_MOLECULAR_SIGNATURE = QueryTemplate(
    name='get_molecular_signature',
    purpose='Get molecular fingerprint (subclass composition) for a region',
    modality='molecular',
    cypher='''
        MATCH (r:Region {acronym: $REGION})-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells, hs.rank AS rank
        ORDER BY hs.rank
    ''',
    parameters={'REGION': 'Region acronym (e.g., CLA, VISp)'},
    expected_rows='variable',
    reasoning_hint='Used for cross-region molecular similarity analysis'
)

Q_FIND_GENE_ENRICHED_POCKETS = QueryTemplate(
    name='find_gene_enriched_pockets',
    purpose='Find all spatial regions where a gene-marked subclass is DOMINANT (rank=1)',
    modality='molecular',
    cypher='''
        MATCH (spatial)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        WHERE (sc.name CONTAINS $GENE OR sc.markers CONTAINS $GENE)
          AND hs.rank = 1
        WITH spatial, sc, hs,
             CASE
               WHEN 'ME_Subregion' IN labels(spatial) THEN 1
               WHEN 'Subregion' IN labels(spatial) THEN 2
               ELSE 3
             END AS granularity
        MATCH (spatial)-[hs2:HAS_SUBCLASS]->(sc2:Subclass)
        WITH spatial, sc, hs, granularity, COUNT(DISTINCT sc2) AS n_subclasses
        RETURN elementId(spatial) AS pocket_eid,
               labels(spatial) AS spatial_type,
               COALESCE(spatial.acronym, spatial.name) AS spatial_name,
               sc.name AS dominant_subclass_name,
               sc.markers AS dominant_markers,
               hs.pct_cells AS dominant_pct,
               n_subclasses,
               granularity AS granularity_score
        ORDER BY granularity ASC, hs.pct_cells DESC
        LIMIT 50
    ''',
    parameters={'GENE': 'Gene marker name'},
    expected_rows='0-N',
    reasoning_hint='Core query for gene circuit analysis - finds where gene is dominant'
)

# =============================================================================
# MORPHOLOGICAL QUERIES - Neuron structure, axonal/dendritic features
# =============================================================================

Q_GET_MORPHOLOGY_SIGNATURE = QueryTemplate(
    name='get_morphology_signature',
    purpose='Get 8-dimensional morphology features for a region',
    modality='morphological',
    cypher='''
        MATCH (r:Region {acronym: $REGION})
        RETURN r.acronym AS region,
               r.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
               r.axonal_length AS axonal_length,
               r.axonal_branches AS axonal_branches,
               r.axonal_maximum_branch_order AS axonal_max_branch_order,
               r.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
               r.dendritic_length AS dendritic_length,
               r.dendritic_branches AS dendritic_branches,
               r.dendritic_maximum_branch_order AS dendritic_max_branch_order
    ''',
    parameters={'REGION': 'Region acronym'},
    expected_rows='1',
    can_return_empty=True,
    reasoning_hint='Use for morphological fingerprint - requires z-score normalization across regions'
)

Q_GET_NEURONS_IN_REGION = QueryTemplate(
    name='get_neurons_in_region',
    purpose='Get all neurons located in a specific region (via LOCATE_AT)',
    modality='morphological',
    cypher='''
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $REGION})
        RETURN n.neuron_id AS neuron_id, n.name AS neuron_name,
               n.celltype AS celltype, n.base_region AS base_region
    ''',
    parameters={'REGION': 'Region acronym'},
    expected_rows='variable',
    reasoning_hint='Use to find morphologically characterized neurons in a region'
)

Q_GET_NEURONS_IN_POCKET = QueryTemplate(
    name='get_neurons_in_pocket',
    purpose='Get neurons in a spatial pocket (Subregion/ME_Subregion level)',
    modality='morphological',
    cypher='''
        MATCH (spatial)
        WHERE elementId(spatial) = $POCKET_EID
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT_SUBREGION|LOCATE_AT_ME_SUBREGION]->(spatial)
        WITH n WHERE n IS NOT NULL
        RETURN n.neuron_id AS neuron_id, n.celltype AS celltype,
               n.axonal_total_length AS axonal_length, n.dendritic_total_length AS dendritic_length,
               n.axonal_number_of_bifurcations AS axonal_branches,
               n.dendritic_number_of_bifurcations AS dendritic_branches
    ''',
    parameters={'POCKET_EID': 'Element ID of the spatial pocket'},
    expected_rows='variable',
    reasoning_hint='Use for fine-grained neuron extraction at subregion level'
)

Q_GET_NEURONS_FALLBACK = QueryTemplate(
    name='get_neurons_fallback',
    purpose='Fallback neuron query using base_region or celltype match',
    modality='morphological',
    cypher='''
        MATCH (n:Neuron)
        WHERE n.base_region = $REGION OR n.celltype CONTAINS $REGION
        RETURN n.neuron_id AS neuron_id, n.name AS neuron_name,
               n.celltype AS celltype, n.base_region AS base_region
    ''',
    parameters={'REGION': 'Region acronym to match'},
    expected_rows='variable',
    reasoning_hint='Use when LOCATE_AT returns empty - some regions lack explicit location links'
)

# =============================================================================
# PROJECTION QUERIES - Connectivity, targets, circuit analysis
# =============================================================================

Q_GET_ALL_TARGET_SUBREGIONS = QueryTemplate(
    name='get_all_target_subregions',
    purpose='Get all projection target subregions (for fingerprint feature space)',
    modality='projection',
    cypher='''
        MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion)
        WHERE COALESCE(p.weight, p.projection_length, p.total, 0) > 0
        RETURN DISTINCT t.acronym AS target_acronym
        ORDER BY t.acronym
    ''',
    expected_rows='variable',
    reasoning_hint='Establishes the target space for projection fingerprints'
)

Q_GET_PROJECTION_SIGNATURE = QueryTemplate(
    name='get_projection_signature',
    purpose='Get projection fingerprint for neurons in a region',
    modality='projection',
    cypher='''
        MATCH (r:Region {acronym: $REGION})<-[:LOCATE_AT]-(n:Neuron)
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE COALESCE(p.weight, p.projection_length, p.total, 0) > 0
        WITH t.acronym AS target,
             SUM(COALESCE(p.weight, p.projection_length, p.total, 0)) AS total_weight
        RETURN target, total_weight
        ORDER BY total_weight DESC
    ''',
    parameters={'REGION': 'Source region acronym'},
    expected_rows='variable',
    reasoning_hint='Used for projection fingerprint - apply log transform and L1 normalization'
)

Q_GET_NEURON_PROJECTIONS = QueryTemplate(
    name='get_neuron_projections',
    purpose='Get individual neuron projection patterns for heatmap (Panel D)',
    modality='projection',
    cypher='''
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $REGION})
        MATCH (n)-[proj:PROJECT_TO]->(target)
        WHERE (target:Region OR target:Subregion OR target:ME_Subregion)
          AND COALESCE(proj.projection_length, proj.weight, proj.total, 0) > 0
        RETURN n.neuron_id AS neuron_id,
               COALESCE(target.acronym, target.name) AS target_acronym,
               labels(target) AS target_type,
               elementId(target) AS target_eid,
               COALESCE(proj.projection_length, proj.weight, proj.total, 0) AS proj_strength
        ORDER BY n.neuron_id, proj_strength DESC
    ''',
    parameters={'REGION': 'Source region acronym'},
    expected_rows='variable',
    reasoning_hint='Use for detailed projection matrix visualization'
)

Q_GET_POCKET_PROJECTIONS = QueryTemplate(
    name='get_pocket_projections',
    purpose='Get aggregated projections from a spatial pocket',
    modality='projection',
    cypher='''
        MATCH (spatial)
        WHERE elementId(spatial) = $POCKET_EID
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT_SUBREGION|LOCATE_AT_ME_SUBREGION]->(spatial)
        WITH n WHERE n IS NOT NULL
        MATCH (n)-[p:PROJECT_TO]->(target)
        WHERE COALESCE(p.projection_length, p.weight, p.total, 0) > 0
        WITH target,
             COLLECT(DISTINCT n.neuron_id) AS neuron_ids,
             COUNT(DISTINCT n) AS n_neurons,
             SUM(COALESCE(p.projection_length, p.weight, p.total, 0)) AS total_strength,
             AVG(COALESCE(p.projection_length, p.weight, p.total, 0)) AS avg_strength,
             CASE WHEN 'ME_Subregion' IN labels(target) THEN 1
                  WHEN 'Subregion' IN labels(target) THEN 2
                  ELSE 3 END AS target_granularity
        RETURN elementId(target) AS target_eid,
               labels(target) AS target_type,
               COALESCE(target.acronym, target.name) AS target_name,
               target_granularity,
               n_neurons AS n_contributing_neurons,
               total_strength AS total_projection_strength,
               avg_strength AS avg_projection_strength,
               neuron_ids[0..5] AS sample_neurons
        ORDER BY target_granularity ASC, total_strength DESC
        LIMIT 30
    ''',
    parameters={'POCKET_EID': 'Element ID of the source pocket'},
    expected_rows='variable',
    reasoning_hint='Use for circuit analysis - identifies major projection targets'
)

Q_GET_TARGET_MOLECULAR_PROFILE = QueryTemplate(
    name='get_target_molecular_profile',
    purpose='Get subclass composition of a projection target (Panel E)',
    modality='molecular',
    cypher='''
        MATCH (target)
        WHERE elementId(target) = $TARGET_EID
        MATCH (target)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN COALESCE(target.acronym, target.name) AS target_name,
               sc.name AS subclass_name,
               sc.markers AS markers,
               hs.pct_cells AS pct_cells,
               hs.rank AS rank
        ORDER BY hs.rank
        LIMIT 20
    ''',
    parameters={'TARGET_EID': 'Element ID of the target region/subregion'},
    expected_rows='variable',
    reasoning_hint='Use to understand the molecular composition of projection targets'
)

# =============================================================================
# SPATIAL QUERIES - Region selection, hierarchy
# =============================================================================

Q_GET_ALL_REGIONS = QueryTemplate(
    name='get_all_regions',
    purpose='Get all regions that have molecular data (HAS_SUBCLASS)',
    modality='spatial',
    cypher='''
        MATCH (r:Region)
        WHERE EXISTS { (r)-[:HAS_SUBCLASS]->() }
        RETURN r.acronym AS region_acronym, r.name AS region_name, r.full_name AS full_name
        ORDER BY r.acronym
    ''',
    expected_rows='variable',
    reasoning_hint='Use to establish the set of analyzable regions'
)

Q_SELECT_TOP_REGIONS_BY_NEURONS = QueryTemplate(
    name='select_top_regions_by_neurons',
    purpose='Select top N regions ranked by neuron count',
    modality='spatial',
    cypher='''
        MATCH (r:Region)
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
        WITH r, COUNT(n) AS neuron_count
        WHERE neuron_count > 0
        RETURN r.acronym AS region_acronym, neuron_count
        ORDER BY neuron_count DESC
        LIMIT $TOP_N
    ''',
    parameters={'TOP_N': 'Number of top regions to select'},
    expected_rows='N',
    reasoning_hint='Use for fingerprint analysis - focuses on regions with morphological data'
)

Q_GET_REGION_PROPERTIES = QueryTemplate(
    name='get_region_properties',
    purpose='Get all properties of a region node',
    modality='spatial',
    cypher='''
        MATCH (r:Region {acronym: $REGION})
        RETURN properties(r) AS props
    ''',
    parameters={'REGION': 'Region acronym'},
    expected_rows='1',
    reasoning_hint='Use to inspect what data is available for a region'
)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

TEMPLATES: Dict[str, QueryTemplate] = {
    # Meta
    'check_connectivity': Q_CHECK_CONNECTIVITY,
    'get_schema_labels': Q_GET_SCHEMA_LABELS,
    'get_schema_relationships': Q_GET_SCHEMA_RELATIONSHIPS,

    # Molecular
    'get_all_subclasses': Q_GET_ALL_SUBCLASSES,
    'get_subclass_by_gene': Q_GET_SUBCLASS_BY_GENE,
    'get_region_enrichment_for_subclass': Q_GET_REGION_ENRICHMENT_FOR_SUBCLASS,
    'get_molecular_signature': Q_GET_MOLECULAR_SIGNATURE,
    'find_gene_enriched_pockets': Q_FIND_GENE_ENRICHED_POCKETS,

    # Morphological
    'get_morphology_signature': Q_GET_MORPHOLOGY_SIGNATURE,
    'get_neurons_in_region': Q_GET_NEURONS_IN_REGION,
    'get_neurons_in_pocket': Q_GET_NEURONS_IN_POCKET,
    'get_neurons_fallback': Q_GET_NEURONS_FALLBACK,

    # Projection
    'get_all_target_subregions': Q_GET_ALL_TARGET_SUBREGIONS,
    'get_projection_signature': Q_GET_PROJECTION_SIGNATURE,
    'get_neuron_projections': Q_GET_NEURON_PROJECTIONS,
    'get_pocket_projections': Q_GET_POCKET_PROJECTIONS,
    'get_target_molecular_profile': Q_GET_TARGET_MOLECULAR_PROFILE,

    # Spatial
    'get_all_regions': Q_GET_ALL_REGIONS,
    'select_top_regions_by_neurons': Q_SELECT_TOP_REGIONS_BY_NEURONS,
    'get_region_properties': Q_GET_REGION_PROPERTIES,
}


def get_template(name: str) -> QueryTemplate:
    """Get a query template by name."""
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


def list_templates_by_modality(modality: str) -> List[QueryTemplate]:
    """List all templates for a given modality."""
    return [t for t in TEMPLATES.values() if t.modality == modality]


def get_reasoning_hints() -> Dict[str, str]:
    """Get all reasoning hints for agent decision-making."""
    return {name: t.reasoning_hint for name, t in TEMPLATES.items()}
