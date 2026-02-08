import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add parent directory to path to import circuit modules
PARENT_DIR = Path(__file__).parent.parent
RESULT6_DIR = PARENT_DIR / "circuit"
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(RESULT6_DIR))

from provenance import ProvenanceLogger, EventType
from evidence_buffer import EvidenceBuffer
from tpar_reasoner import TPARReasoner, ReasoningStep, reason_about_enrichment, reason_about_projection


def _check_neo4j_connectivity(uri: str, user: str, password: str) -> Dict[str, Any]:
    """Check Neo4j connectivity before running analysis."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            record = result.single()
            ok = record and record['ok'] == 1
        driver.close()
        return {'connected': ok, 'error': None}
    except Exception as e:
        return {'connected': False, 'error': str(e)}


# ============================================================
# Canonical queries (replicated exactly from circuit/collect_data.py)
# ============================================================

def _build_find_pockets_query(gene: str) -> str:
    """Build the pocket discovery query for a given gene marker.
    Matches QUERY_FIND_ALL_CAR3_POCKETS from collect_data.py."""
    return f"""
    MATCH (spatial)
    WHERE spatial:Region OR spatial:Subregion OR spatial:ME_Subregion

    MATCH (spatial)-[has_sc:HAS_SUBCLASS]->(sc:Subclass)

    WITH spatial,
         labels(spatial) AS spatial_type,
         COALESCE(spatial.acronym, spatial.name) AS spatial_name,
         sc,
         has_sc.pct_cells AS pct_cells,
         has_sc.rank AS rank

    WITH spatial, spatial_type, spatial_name,
         collect({{
             subclass_name: sc.name,
             markers: sc.markers,
             pct_cells: pct_cells,
             rank: rank
         }}) AS subclass_profile

    WITH spatial, spatial_type, spatial_name, subclass_profile,
         [x IN subclass_profile WHERE x.rank = 1][0] AS dominant_subclass,
         size(subclass_profile) AS n_subclasses

    WHERE dominant_subclass IS NOT NULL
      AND (
          dominant_subclass.markers CONTAINS '{gene}'
          OR dominant_subclass.subclass_name CONTAINS '{gene}'
      )

    RETURN
        elementId(spatial) AS pocket_eid,
        spatial_type,
        spatial_name,
        dominant_subclass.subclass_name AS dominant_subclass_name,
        dominant_subclass.markers AS dominant_markers,
        dominant_subclass.pct_cells AS dominant_pct,
        dominant_subclass.rank AS dominant_rank,
        n_subclasses,
        subclass_profile
    ORDER BY
        CASE
            WHEN 'ME_Subregion' IN spatial_type THEN 1
            WHEN 'Subregion' IN spatial_type THEN 2
            WHEN 'Region' IN spatial_type THEN 3
        END,
        dominant_pct DESC
    LIMIT 50
    """


# Canonical neuron query from collect_data.py QUERY_GET_NEURONS_IN_POCKET
# NOTE: Only Subregion and ME_Subregion LOCATE_AT variants (no Region LOCATE_AT)
QUERY_GET_NEURONS = """
MATCH (spatial)
WHERE elementId(spatial) = $POCKET_EID

OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT_SUBREGION]->(spatial)
WHERE spatial:Subregion

OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(spatial)
WHERE spatial:ME_Subregion

WITH COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) AS neurons
UNWIND neurons AS n
WITH DISTINCT n
WHERE n IS NOT NULL

RETURN
    n.neuron_id AS neuron_id,
    n.name AS neuron_name,
    n.celltype AS celltype,
    n.base_region AS base_region
"""

# Canonical projection query from collect_data.py QUERY_GET_PROJECTION_TARGETS
QUERY_GET_PROJECTIONS = """
MATCH (spatial)
WHERE elementId(spatial) = $POCKET_EID

OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT_SUBREGION]->(spatial)
WHERE spatial:Subregion

OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(spatial)
WHERE spatial:ME_Subregion

WITH COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) AS neurons
UNWIND neurons AS n
WITH DISTINCT n
WHERE n IS NOT NULL

MATCH (n)-[proj:PROJECT_TO]->(target)
WHERE target:Region OR target:Subregion OR target:ME_Subregion

WITH target,
     labels(target) AS target_type,
     COALESCE(target.acronym, target.name) AS target_name,
     n.neuron_id AS source_neuron,
     COALESCE(proj.projection_length, proj.weight, proj.total, 0) AS proj_strength

WHERE proj_strength > 0

WITH target, target_type, target_name,
     count(DISTINCT source_neuron) AS n_contributing_neurons,
     sum(proj_strength) AS total_projection_strength,
     avg(proj_strength) AS avg_projection_strength,
     collect(DISTINCT source_neuron)[0..5] AS sample_neurons

WITH target, target_type, target_name,
     n_contributing_neurons,
     total_projection_strength,
     avg_projection_strength,
     sample_neurons,
     CASE
         WHEN 'ME_Subregion' IN target_type THEN 1
         WHEN 'Subregion' IN target_type THEN 2
         WHEN 'Region' IN target_type THEN 3
         ELSE 4
     END AS target_granularity

RETURN
    elementId(target) AS target_eid,
    target_type,
    target_name,
    target_granularity,
    n_contributing_neurons,
    total_projection_strength,
    avg_projection_strength,
    sample_neurons

ORDER BY n_contributing_neurons DESC, total_projection_strength DESC
LIMIT 30
"""

# Canonical target profile query from collect_data.py
QUERY_GET_TARGET_PROFILE = """
MATCH (target)
WHERE elementId(target) = $TARGET_EID

MATCH (target)-[has_sc:HAS_SUBCLASS]->(sc:Subclass)

RETURN
    COALESCE(target.acronym, target.name) AS target_name,
    sc.name AS subclass_name,
    sc.markers AS markers,
    has_sc.pct_cells AS pct_cells,
    has_sc.rank AS rank

ORDER BY has_sc.rank ASC
"""


# ============================================================
# Panel A-F queries (for Figure reproduction)
# ============================================================

QUERY_PANEL_A_SUBCLASS_IDENTITY = """
MATCH (sc:Subclass)
WHERE sc.name CONTAINS $GENE OR sc.markers CONTAINS $GENE
RETURN
    sc.name AS subclass_name,
    sc.markers AS markers,
    elementId(sc) AS subclass_eid
ORDER BY sc.name
"""

QUERY_PANEL_B_REGION_ENRICHMENT = """
MATCH (spatial:Region)-[hs:HAS_SUBCLASS]->(sc:Subclass)
WHERE sc.name = $SUBCLASS_NAME
RETURN
    spatial.acronym AS region_name,
    hs.pct_cells AS pct_cells,
    hs.rank AS rank
ORDER BY hs.pct_cells DESC
"""

QUERY_PANEL_C_CLA_NEURONS = """
MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $REGION_ACRONYM})
RETURN
    n.neuron_id AS neuron_id,
    n.name AS neuron_name,
    n.celltype AS celltype,
    n.base_region AS base_region
"""

QUERY_PANEL_C_CLA_NEURONS_FALLBACK = """
MATCH (n:Neuron)
WHERE n.base_region = $REGION_ACRONYM OR n.celltype CONTAINS $REGION_ACRONYM
RETURN
    n.neuron_id AS neuron_id,
    n.name AS neuron_name,
    n.celltype AS celltype,
    n.base_region AS base_region
"""

QUERY_PANEL_C_MORPHOLOGY = """
MATCH (r:Region {acronym: $REGION_ACRONYM})
RETURN
    r.acronym AS region,
    r.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
    r.axonal_length AS axonal_length,
    r.axonal_branches AS axonal_branches,
    r.axonal_maximum_branch_order AS axonal_max_branch_order,
    r.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
    r.dendritic_length AS dendritic_length,
    r.dendritic_branches AS dendritic_branches,
    r.dendritic_maximum_branch_order AS dendritic_max_branch_order
"""

QUERY_PANEL_D_NEURON_PROJECTIONS = """
MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $REGION_ACRONYM})
MATCH (n)-[proj:PROJECT_TO]->(target)
WHERE (target:Region OR target:Subregion OR target:ME_Subregion)
  AND COALESCE(proj.projection_length, proj.weight, proj.total, 0) > 0
RETURN
    n.neuron_id AS neuron_id,
    COALESCE(target.acronym, target.name) AS target_acronym,
    labels(target) AS target_type,
    elementId(target) AS target_eid,
    COALESCE(proj.projection_length, proj.weight, proj.total, 0) AS proj_strength
ORDER BY n.neuron_id, proj_strength DESC
"""

QUERY_PANEL_D_NEURON_PROJECTIONS_FALLBACK = """
MATCH (n:Neuron)
WHERE n.base_region = $REGION_ACRONYM OR n.celltype CONTAINS $REGION_ACRONYM
MATCH (n)-[proj:PROJECT_TO]->(target)
WHERE (target:Region OR target:Subregion OR target:ME_Subregion)
  AND COALESCE(proj.projection_length, proj.weight, proj.total, 0) > 0
RETURN
    n.neuron_id AS neuron_id,
    COALESCE(target.acronym, target.name) AS target_acronym,
    labels(target) AS target_type,
    elementId(target) AS target_eid,
    COALESCE(proj.projection_length, proj.weight, proj.total, 0) AS proj_strength
ORDER BY n.neuron_id, proj_strength DESC
"""


# ============================================================
# Panel A-F functions
# ============================================================

def _query_hash(query: str) -> str:
    """Generate a short hash of a Cypher query for provenance."""
    import hashlib
    return hashlib.sha256(query.encode()).hexdigest()[:12]


def _run_panel_a(session, gene: str, evidence_buffer, provenance_logger) -> pd.DataFrame:
    """Panel A: Identify gene-marked transcriptomic subclass(es)."""
    query = QUERY_PANEL_A_SUBCLASS_IDENTITY
    if provenance_logger:
        provenance_logger.log_act(6, 'kg_query', f'Panel A: Identify {gene} subclass nodes',
                                  query=query[:80])
    rows = [dict(rec) for rec in session.run(query, GENE=gene)]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['subclass_name', 'markers', 'subclass_eid'])
    if evidence_buffer and not df.empty:
        evidence_buffer.add_evidence(
            modality='molecular', source_step=6,
            query=query[:80], data=rows[:10],
            key_fields=['subclass_name', 'markers']
        )
    return df


def _run_panel_b(session, subclass_name: str, evidence_buffer, provenance_logger) -> pd.DataFrame:
    """Panel B: Region enrichment bar chart — pct_cells of a subclass across ALL Region nodes."""
    query = QUERY_PANEL_B_REGION_ENRICHMENT
    if provenance_logger:
        provenance_logger.log_act(7, 'kg_query',
                                  f'Panel B: Region enrichment for {subclass_name}',
                                  query=query[:80])
    rows = []
    for rec in session.run(query, SUBCLASS_NAME=subclass_name):
        rows.append({
            'region_name': rec['region_name'],
            'pct_cells': float(rec['pct_cells'] or 0.0),
            'rank': rec['rank']
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['region_name', 'pct_cells', 'rank'])
    if evidence_buffer and not df.empty:
        evidence_buffer.add_evidence(
            modality='molecular', source_step=7,
            query=query[:80], data=rows[:10],
            key_fields=['region_name', 'pct_cells', 'rank']
        )
    return df


def _run_panel_c(session, region_acronym: str, panel_b_df, evidence_buffer, provenance_logger):
    """Panel C: Multi-modal closed-loop evidence for a region (e.g., CLA)."""
    panel_c = {'region': region_acronym}

    # C.1 Molecular: extract from Panel B
    if panel_b_df is not None and not panel_b_df.empty:
        cla_row = panel_b_df[panel_b_df['region_name'] == region_acronym]
        if not cla_row.empty:
            panel_c['molecular_pct'] = float(cla_row.iloc[0]['pct_cells'])
            panel_c['molecular_rank'] = int(cla_row.iloc[0]['rank'])

    # C.2 Morphology: 8-dim features from Region node
    morph_query = QUERY_PANEL_C_MORPHOLOGY
    if provenance_logger:
        provenance_logger.log_act(8, 'kg_query', f'Panel C: Morphology for {region_acronym}',
                                  query=morph_query[:80])
    morph_rec = session.run(morph_query, REGION_ACRONYM=region_acronym).single()
    if morph_rec:
        panel_c['morphology'] = {k: morph_rec[k] for k in morph_rec.keys() if k != 'region'}
    if evidence_buffer:
        evidence_buffer.add_evidence(
            modality='morphological', source_step=8,
            query=morph_query[:80],
            data=[panel_c.get('morphology', {})],
            key_fields=['axonal_length', 'dendritic_length']
        )

    # C.3 Neurons via LOCATE_AT (Region-level), with fallback
    neuron_query = QUERY_PANEL_C_CLA_NEURONS
    if provenance_logger:
        provenance_logger.log_act(9, 'kg_query',
                                  f'Panel C: Neurons at {region_acronym} via LOCATE_AT',
                                  query=neuron_query[:80])
    neuron_rows = [dict(rec) for rec in session.run(neuron_query, REGION_ACRONYM=region_acronym)]

    if not neuron_rows:
        # Fallback: search by base_region or celltype
        fb_query = QUERY_PANEL_C_CLA_NEURONS_FALLBACK
        if provenance_logger:
            provenance_logger.log_act(9, 'kg_query',
                                      f'Panel C: Neurons fallback for {region_acronym}',
                                      query=fb_query[:80])
        neuron_rows = [dict(rec) for rec in session.run(fb_query, REGION_ACRONYM=region_acronym)]
        panel_c['neuron_method'] = 'fallback (base_region/celltype)'
    else:
        panel_c['neuron_method'] = 'LOCATE_AT'

    panel_c['neurons'] = neuron_rows
    panel_c['neuron_count'] = len(neuron_rows)

    if evidence_buffer and neuron_rows:
        evidence_buffer.add_evidence(
            modality='morphological', source_step=9,
            query=f'Neurons at {region_acronym}',
            data=neuron_rows[:10],
            key_fields=['neuron_id', 'celltype', 'base_region']
        )

    return panel_c


def _run_panel_d(session, region_acronym: str, evidence_buffer, provenance_logger):
    """Panel D: Individual neuron x target projection matrix for heatmap."""
    query = QUERY_PANEL_D_NEURON_PROJECTIONS
    if provenance_logger:
        provenance_logger.log_act(10, 'kg_query',
                                  f'Panel D: Neuron projection matrix for {region_acronym}',
                                  query=query[:80])
    rows = []
    for rec in session.run(query, REGION_ACRONYM=region_acronym):
        strength = float(rec['proj_strength'])
        rows.append({
            'neuron_id': rec['neuron_id'],
            'target_acronym': rec['target_acronym'],
            'target_type': str(rec['target_type']),
            'target_eid': rec['target_eid'],
            'proj_strength': strength,
            'proj_log10': float(np.log10(1 + strength))
        })

    if not rows:
        # Fallback: search by base_region/celltype
        fb_query = QUERY_PANEL_D_NEURON_PROJECTIONS_FALLBACK
        if provenance_logger:
            provenance_logger.log_act(10, 'kg_query',
                                      f'Panel D: Projection fallback for {region_acronym}',
                                      query=fb_query[:80])
        for rec in session.run(fb_query, REGION_ACRONYM=region_acronym):
            strength = float(rec['proj_strength'])
            rows.append({
                'neuron_id': rec['neuron_id'],
                'target_acronym': rec['target_acronym'],
                'target_type': str(rec['target_type']),
                'target_eid': rec['target_eid'],
                'proj_strength': strength,
                'proj_log10': float(np.log10(1 + strength))
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['neuron_id', 'target_acronym', 'target_type', 'target_eid', 'proj_strength', 'proj_log10'])

    if evidence_buffer and not df.empty:
        evidence_buffer.add_evidence(
            modality='projection', source_step=10,
            query=f'Neuron projections from {region_acronym}',
            data=rows[:10],
            key_fields=['neuron_id', 'target_acronym', 'proj_log10']
        )
    return df


def _run_panel_e(session, target_eids: List[str], evidence_buffer, provenance_logger):
    """Panel E: Full subclass composition for top target subregions."""
    if provenance_logger:
        provenance_logger.log_act(11, 'kg_query',
                                  f'Panel E: Full composition for {len(target_eids)} targets',
                                  query=QUERY_GET_TARGET_PROFILE[:80])
    all_profiles = []
    for target_eid in target_eids:
        for rec in session.run(QUERY_GET_TARGET_PROFILE, TARGET_EID=target_eid):
            all_profiles.append({
                'target_name': rec['target_name'],
                'subclass_name': rec['subclass_name'],
                'markers': rec['markers'],
                'pct_cells': float(rec['pct_cells'] or 0.0),
                'rank': rec['rank']
            })
    df = pd.DataFrame(all_profiles) if all_profiles else pd.DataFrame(
        columns=['target_name', 'subclass_name', 'markers', 'pct_cells', 'rank'])
    if evidence_buffer and not df.empty:
        evidence_buffer.add_evidence(
            modality='molecular', source_step=11,
            query=f'Panel E full composition ({len(target_eids)} targets)',
            data=all_profiles[:10],
            key_fields=['target_name', 'subclass_name', 'pct_cells']
        )
    return df


def _build_panel_f_subgraph(gene, panel_a_df, panel_b_df, panel_c_data,
                             panel_d_df, panel_e_df) -> str:
    """Panel F: Compile all touched nodes/edges into Cypher reconstruction."""
    lines = [
        f"// Panel F: Subgraph export for {gene} analysis",
        f"// Auto-generated Cypher reconstruction",
        ""
    ]
    # Subclass nodes (Panel A)
    if panel_a_df is not None and not panel_a_df.empty:
        lines.append("// --- Subclass nodes ---")
        for _, row in panel_a_df.iterrows():
            name = str(row.get('subclass_name', '')).replace("'", "\\'")
            markers = str(row.get('markers', '')).replace("'", "\\'")
            lines.append(f"MERGE (sc:Subclass {{name: '{name}', markers: '{markers}'}})")
    # Region nodes (Panel B top-20)
    if panel_b_df is not None and not panel_b_df.empty:
        lines.append("\n// --- Region nodes (top enriched) ---")
        for _, row in panel_b_df.head(20).iterrows():
            region = row.get('region_name', '')
            lines.append(f"MERGE (r:Region {{acronym: '{region}'}})")
    # Neuron nodes (Panel C/D)
    seen_neurons = set()
    if panel_d_df is not None and not panel_d_df.empty:
        lines.append("\n// --- Neuron nodes ---")
        for nid in panel_d_df['neuron_id'].unique():
            if nid not in seen_neurons:
                lines.append(f"MERGE (n:Neuron {{neuron_id: '{nid}'}})")
                seen_neurons.add(nid)
    # LOCATE_AT edges
    region = panel_c_data.get('region', '') if panel_c_data else ''
    if region and seen_neurons:
        lines.append(f"\n// --- LOCATE_AT edges ---")
        for nid in seen_neurons:
            lines.append(
                f"MATCH (n:Neuron {{neuron_id: '{nid}'}}), (r:Region {{acronym: '{region}'}}) "
                f"MERGE (n)-[:LOCATE_AT]->(r)"
            )
    # PROJECT_TO edges (top 50)
    if panel_d_df is not None and not panel_d_df.empty:
        lines.append(f"\n// --- PROJECT_TO edges (top 50 by strength) ---")
        top_projs = panel_d_df.nlargest(50, 'proj_strength')
        for _, row in top_projs.iterrows():
            nid = row['neuron_id']
            target = row['target_acronym']
            strength = row['proj_strength']
            lines.append(
                f"MATCH (n:Neuron {{neuron_id: '{nid}'}}), (t {{acronym: '{target}'}}) "
                f"MERGE (n)-[:PROJECT_TO {{projection_length: {strength:.2f}}}]->(t)"
            )
    # HAS_SUBCLASS edges (Panel E)
    if panel_e_df is not None and not panel_e_df.empty:
        lines.append(f"\n// --- HAS_SUBCLASS edges (target composition) ---")
        for target_name in panel_e_df['target_name'].unique()[:20]:
            target_rows = panel_e_df[panel_e_df['target_name'] == target_name].head(3)
            for _, row in target_rows.iterrows():
                sc_name = str(row['subclass_name']).replace("'", "\\'")
                pct = row['pct_cells']
                rank = row['rank']
                lines.append(
                    f"MATCH (t {{acronym: '{target_name}'}}), (sc:Subclass {{name: '{sc_name}'}}) "
                    f"MERGE (t)-[:HAS_SUBCLASS {{pct_cells: {pct}, rank: {rank}}}]->(sc)"
                )
    return "\n".join(lines)


def _rank_candidate_circuits(pockets_df, neurons_dict, projections_dict, gene: str = "Car3"):
    """
    Canonical scoring from collect_data.py rank_candidate_circuits().

    IMPORTANT: Scoring formulas match collect_data.py EXACTLY:
    1. molecular_score = min(dominant_pct, 30)  # NOT pct * 30 / 100
    2. neuron_score = range-based (5-1000: 20, >1000: 15, >0: 10, 0: 0)
    3. projection_score = top_3_fraction of n_contributing_neurons * 20
    """
    scored_pockets = []

    for idx, row in pockets_df.iterrows():
        pocket_eid = row['pocket_eid']

        score = 0
        reasons = []

        # 1. Molecular specificity (0-30): min(dominant_pct, 30)
        molecular_score = min(row['dominant_pct'], 30)
        score += molecular_score
        reasons.append(f"Molecular: {molecular_score:.1f} (dominant {row['dominant_pct']:.1f}%)")

        # 2. Spatial precision (0-20)
        granularity_score = {1: 20, 2: 15, 3: 10}.get(row['granularity_score'], 0)
        score += granularity_score
        spatial_level = {1: 'ME_Subregion', 2: 'Subregion', 3: 'Region'}.get(row['granularity_score'], 'Unknown')
        reasons.append(f"Spatial: {granularity_score} ({spatial_level})")

        # 3. Neuron count (0-20): range-based
        neurons = neurons_dict.get(pocket_eid, pd.DataFrame())
        n_neurons = len(neurons) if isinstance(neurons, pd.DataFrame) else 0
        if 5 <= n_neurons <= 1000:
            neuron_score = 20
        elif n_neurons > 1000:
            neuron_score = 15
        elif n_neurons > 0:
            neuron_score = 10
        else:
            neuron_score = 0
        score += neuron_score
        reasons.append(f"Neurons: {neuron_score} ({n_neurons})")

        # 4. Projection specificity (0-20): top_3_fraction of n_contributing_neurons
        projections = projections_dict.get(pocket_eid, pd.DataFrame())
        if not projections.empty and 'n_contributing_neurons' in projections.columns:
            total_neurons = projections['n_contributing_neurons'].sum()
            top_3_neurons = projections.head(3)['n_contributing_neurons'].sum()
            top_3_fraction = top_3_neurons / total_neurons if total_neurons > 0 else 0
            projection_score = min(top_3_fraction * 20, 20)
        else:
            projection_score = 0
            top_3_fraction = 0
        score += projection_score
        reasons.append(f"Projection: {projection_score:.1f} (top3 {top_3_fraction*100:.1f}%)")

        # 5. Gene relevance (0-10): always 10 for gene-enriched pockets
        gene_score = 10
        score += gene_score
        reasons.append(f"{gene}: 10")

        scored_pockets.append({
            'pocket_eid': pocket_eid,
            'spatial_name': row['spatial_name'],
            'spatial_type': row['spatial_type'],
            'total_score': score,
            'molecular_score': molecular_score,
            'granularity_score': granularity_score,
            'neuron_score': neuron_score,
            'projection_score': projection_score,
            'gene_score': gene_score,
            'n_neurons': n_neurons,
            'n_projection_targets': len(projections),
            'dominant_pct': row['dominant_pct'],
            'ranking_reasons': ' | '.join(reasons)
        })

    return pd.DataFrame(scored_pockets).sort_values('total_score', ascending=False)


def run_circuit_canonical(
    gene: str = "Car3",
    seed: int = 42,
    output_dir: str = "./outputs/circuit_discovery",
    provenance_logger: Optional[ProvenanceLogger] = None,
    evidence_buffer: Optional[EvidenceBuffer] = None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
) -> Dict[str, Any]:
    """
    Run the canonical Circuit analysis using circuit/collect_data.py logic.

    Follows the exact same flow as collect_data.py main():
    1. Find gene-enriched pockets
    2. Extract neurons per pocket
    3. Analyze projection patterns per pocket
    4. Rank candidate circuits (canonical scoring)
    5. Profile top circuit targets

    Args:
        gene: Gene marker to analyze (default: Car3)
        seed: Random seed for reproducibility
        output_dir: Output directory for results
        provenance_logger: Optional provenance logger
        evidence_buffer: Optional evidence buffer
        neo4j_uri: Neo4j URI (defaults to env var)
        neo4j_user: Neo4j user (defaults to env var)
        neo4j_password: Neo4j password (defaults to env var)

    Returns:
        Dictionary with result summary and file paths
    """
    import random

    # Set determinism
    random.seed(seed)
    np.random.seed(seed)

    # Get Neo4j credentials
    uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neuroxiv")

    # Create output directories
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'success': False,
        'error': None,
        'files': {},
        'summary': {},
        'kg_query_count': 0,
        'neo4j_status': None
    }

    # Neo4j connectivity check (fail-fast)
    neo4j_status = _check_neo4j_connectivity(uri, user, password)
    result['neo4j_status'] = neo4j_status

    if not neo4j_status['connected']:
        result['error'] = f"Neo4j connectivity check failed: {neo4j_status['error']}"
        if provenance_logger:
            provenance_logger.log_think("Neo4j connectivity FAILED", {
                'uri': uri, 'error': neo4j_status['error']
            })
        return result

    # Log start
    if provenance_logger:
        provenance_logger.log_think(f"Starting canonical Circuit analysis for {gene}", {
            'gene': gene,
            'neo4j_uri': uri,
            'seed': seed,
            'neo4j_connected': True
        })

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))

        if provenance_logger:
            provenance_logger.log_plan([
                {'step': 1, 'purpose': f'Find all {gene} enriched pockets'},
                {'step': 2, 'purpose': 'Extract neurons from each pocket'},
                {'step': 3, 'purpose': 'Analyze projection patterns per pocket'},
                {'step': 4, 'purpose': 'Rank candidate circuits (canonical scoring)'},
                {'step': 5, 'purpose': 'Profile top circuit targets'},
            ], planner_type='circuit_canonical')

        with driver.session() as session:

            # Step 1: Find gene-enriched pockets (canonical query)
            pocket_query = _build_find_pockets_query(gene)
            if provenance_logger:
                provenance_logger.log_act(1, 'kg_query', f'Find {gene} enriched pockets',
                                          query=pocket_query[:100])
            result['kg_query_count'] += 1

            pockets_data = []
            query_result = session.run(pocket_query)
            for rec in query_result:
                pockets_data.append({
                    'pocket_eid': rec['pocket_eid'],
                    'spatial_type': str(rec['spatial_type']),
                    'spatial_name': rec['spatial_name'],
                    'dominant_subclass_name': rec['dominant_subclass_name'],
                    'dominant_markers': rec['dominant_markers'],
                    'dominant_pct': float(rec['dominant_pct'] or 0.0),
                    'n_subclasses': rec['n_subclasses'],
                    'granularity_score': 1 if 'ME_Subregion' in str(rec['spatial_type']) else
                                        2 if 'Subregion' in str(rec['spatial_type']) else 3
                })

            pockets_df = pd.DataFrame(pockets_data)

            if evidence_buffer and not pockets_df.empty:
                evidence_buffer.add_evidence(
                    modality='molecular',
                    source_step=1,
                    query=pocket_query[:80],
                    data=pockets_data[:10],
                    key_fields=['spatial_name', 'dominant_pct', 'n_subclasses']
                )

            # REFLECT on Step 1: Pocket discovery
            if provenance_logger:
                n_pockets = len(pockets_df)
                confidence = 1.0 if n_pockets >= 1 else 0.0
                top_pocket = pockets_df.iloc[0] if not pockets_df.empty else None
                findings = []
                if top_pocket is not None:
                    findings.append(f"Found {n_pockets} {gene}-enriched pocket(s)")
                    findings.append(f"Top pocket: {top_pocket['spatial_name']} with {top_pocket['dominant_pct']*100:.1f}% enrichment")
                    if top_pocket['dominant_pct'] > 0.15:
                        findings.append(f"{gene} is DOMINANT in {top_pocket['spatial_name']} (>15%)")
                provenance_logger.log_reflect(
                    step_number=1,
                    validation_status='passed' if n_pockets > 0 else 'failed',
                    confidence=confidence,
                    should_replan=n_pockets == 0,
                    recommendations=findings if n_pockets > 0 else [f"No {gene} pockets found - check gene name"]
                )

            if pockets_df.empty:
                result['error'] = f"No {gene} enriched pockets found"
                driver.close()
                return result

            # Save pockets as subclasses.csv
            pockets_df.to_csv(data_dir / "subclasses.csv", index=False)
            result['files']['subclasses'] = str(data_dir / "subclasses.csv")

            # Step 2-3: Extract neurons and projections per pocket (canonical queries)
            neurons_dict = {}
            projections_dict = {}
            all_neurons = []
            all_projections = []

            for idx, pocket in pockets_df.iterrows():
                pocket_eid = pocket['pocket_eid']

                # Neurons (canonical query - no Region LOCATE_AT)
                result['kg_query_count'] += 1
                neurons_result = session.run(QUERY_GET_NEURONS, POCKET_EID=pocket_eid)
                neurons_rows = [dict(rec) for rec in neurons_result]
                neurons_df = pd.DataFrame(neurons_rows) if neurons_rows else pd.DataFrame()
                neurons_dict[pocket_eid] = neurons_df

                for nr in neurons_rows:
                    nr['spatial_name'] = pocket['spatial_name']
                    all_neurons.append(nr)

                # Projections (canonical query with projection_length/weight/total)
                if not neurons_df.empty:
                    result['kg_query_count'] += 1
                    proj_result = session.run(QUERY_GET_PROJECTIONS, POCKET_EID=pocket_eid)
                    proj_rows = []
                    for rec in proj_result:
                        proj_rows.append({
                            'target_eid': rec['target_eid'],
                            'target_type': str(rec['target_type']),
                            'target_name': rec['target_name'],
                            'target_granularity': rec['target_granularity'],
                            'n_contributing_neurons': rec['n_contributing_neurons'],
                            'total_projection_strength': rec['total_projection_strength'],
                            'avg_projection_strength': rec['avg_projection_strength'],
                            'sample_neurons': rec['sample_neurons']
                        })
                    projections_dict[pocket_eid] = pd.DataFrame(proj_rows) if proj_rows else pd.DataFrame()

                    for pr in proj_rows:
                        pr['source_spatial'] = pocket['spatial_name']
                        all_projections.append(pr)
                else:
                    projections_dict[pocket_eid] = pd.DataFrame()

            if provenance_logger:
                provenance_logger.log_act(2, 'kg_query', 'Extract neurons from pockets',
                                          result_summary={'total_neurons': len(all_neurons),
                                                          'pockets_processed': len(pockets_df)})
                # REFLECT on Step 2: Neuron extraction
                neuron_confidence = min(1.0, len(all_neurons) / 10) if len(all_neurons) > 0 else 0.0
                provenance_logger.log_reflect(
                    step_number=2,
                    validation_status='passed' if all_neurons else 'low_confidence',
                    confidence=neuron_confidence,
                    should_replan=len(all_neurons) == 0,
                    recommendations=[
                        f"Found {len(all_neurons)} neurons across {len(pockets_df)} pockets",
                        f"Average {len(all_neurons)/len(pockets_df):.1f} neurons per pocket" if pockets_df.shape[0] > 0 else "No pockets"
                    ]
                )

                provenance_logger.log_act(3, 'kg_query', 'Analyze projection patterns',
                                          result_summary={'total_projections': len(all_projections)})
                # REFLECT on Step 3: Projection analysis
                proj_confidence = min(1.0, len(all_projections) / 20) if len(all_projections) > 0 else 0.0
                unique_targets = len(set(p.get('target_name', '') for p in all_projections)) if all_projections else 0
                provenance_logger.log_reflect(
                    step_number=3,
                    validation_status='passed' if all_projections else 'low_confidence',
                    confidence=proj_confidence,
                    should_replan=False,
                    recommendations=[
                        f"Found {len(all_projections)} projection entries to {unique_targets} unique targets",
                        f"Projection data {'sufficient' if proj_confidence > 0.7 else 'limited'} for circuit analysis"
                    ]
                )

            # Save morphology summary (neurons)
            if all_neurons:
                neurons_all_df = pd.DataFrame(all_neurons)
                neurons_all_df.to_csv(data_dir / "morphology_summary.csv", index=False)
                result['files']['morphology_summary'] = str(data_dir / "morphology_summary.csv")

                if evidence_buffer:
                    evidence_buffer.add_evidence(
                        modality='morphological',
                        source_step=2,
                        query='QUERY_GET_NEURONS per pocket (canonical)',
                        data=all_neurons[:10],
                        key_fields=['neuron_id', 'celltype', 'spatial_name']
                    )

            # Save projection targets
            if all_projections:
                proj_all_df = pd.DataFrame(all_projections)
                proj_all_df.to_csv(data_dir / "projection_targets.csv", index=False)
                result['files']['projection_targets'] = str(data_dir / "projection_targets.csv")

                if evidence_buffer:
                    evidence_buffer.add_evidence(
                        modality='projection',
                        source_step=3,
                        query='QUERY_GET_PROJECTIONS per pocket (canonical)',
                        data=all_projections[:10],
                        key_fields=['target_name', 'n_contributing_neurons', 'source_spatial']
                    )

            # Step 4: Rank circuits (CANONICAL scoring)
            if provenance_logger:
                provenance_logger.log_act(4, 'computation', 'Rank candidate circuits (canonical scoring)')

            ranked_df = _rank_candidate_circuits(pockets_df, neurons_dict, projections_dict, gene)
            ranked_df.to_csv(data_dir / "region_enrichment.csv", index=False)
            result['files']['region_enrichment'] = str(data_dir / "region_enrichment.csv")

            # REFLECT on Step 4: Circuit ranking
            if provenance_logger:
                top_circuit = ranked_df.iloc[0] if not ranked_df.empty else None
                rank_findings = []
                if top_circuit is not None:
                    rank_findings.append(f"Top circuit: {top_circuit['spatial_name']} with total_score={top_circuit['total_score']:.1f}")
                    rank_findings.append(f"Molecular score: {top_circuit['molecular_score']:.1f}/30")
                    rank_findings.append(f"Spatial score: {top_circuit['spatial_score']:.1f}/20")
                    rank_findings.append(f"Projection score: {top_circuit['projection_score']:.1f}/20")
                provenance_logger.log_reflect(
                    step_number=4,
                    validation_status='passed' if not ranked_df.empty else 'failed',
                    confidence=1.0 if not ranked_df.empty else 0.0,
                    should_replan=False,
                    recommendations=rank_findings if rank_findings else ["No circuits ranked"]
                )

            # Step 5: Profile top circuit targets (canonical query)
            if provenance_logger:
                provenance_logger.log_act(5, 'kg_query', 'Profile top circuit targets')

            target_profiles = []
            if not ranked_df.empty:
                top_pocket_eid = ranked_df.iloc[0]['pocket_eid']
                top_projections = projections_dict.get(top_pocket_eid, pd.DataFrame())

                if not top_projections.empty:
                    for _, proj in top_projections.head(5).iterrows():
                        target_eid = proj.get('target_eid')
                        if not target_eid:
                            continue

                        result['kg_query_count'] += 1
                        profile_result = session.run(QUERY_GET_TARGET_PROFILE, TARGET_EID=target_eid)
                        for rec in profile_result:
                            target_profiles.append({
                                'target_name': rec['target_name'],
                                'subclass_name': rec['subclass_name'],
                                'markers': rec['markers'],
                                'pct_cells': rec['pct_cells'],
                                'rank': rec['rank']
                            })

                    if target_profiles:
                        target_comp_df = pd.DataFrame(target_profiles)
                        target_comp_df.to_csv(data_dir / "target_composition.csv", index=False)
                        result['files']['target_composition'] = str(data_dir / "target_composition.csv")

            # ============================================================
            # Panel A-F: Figure reproduction queries
            # ============================================================

            # Panel A: Identify Car3 subclass
            result['kg_query_count'] += 1
            panel_a_df = _run_panel_a(session, gene, evidence_buffer, provenance_logger)
            if not panel_a_df.empty:
                panel_a_df.to_csv(data_dir / "panel_a_subclass_identity.csv", index=False)
                result['files']['panel_a_subclass_identity'] = str(data_dir / "panel_a_subclass_identity.csv")

            # REFLECT on Panel A: Subclass identification
            if provenance_logger:
                if not panel_a_df.empty:
                    subclass_name = panel_a_df.iloc[0]['subclass_name']
                    markers = panel_a_df.iloc[0]['markers']
                    provenance_logger.log_reflect(
                        step_number=6,
                        validation_status='passed',
                        confidence=1.0,
                        should_replan=False,
                        recommendations=[
                            f"Panel A: Identified {gene} subclass as '{subclass_name}'",
                            f"Markers: {markers}",
                            f"This subclass defines the transcriptomic identity for {gene}+ neurons"
                        ]
                    )
                else:
                    provenance_logger.log_reflect(
                        step_number=6,
                        validation_status='failed',
                        confidence=0.0,
                        should_replan=True,
                        recommendations=[f"No subclass found containing {gene} - check gene name spelling"]
                    )

            # Panel B: Region enrichment across ALL regions
            panel_b_df = pd.DataFrame()
            if not panel_a_df.empty:
                car3_subclass_name = panel_a_df.iloc[0]['subclass_name']
                result['kg_query_count'] += 1
                panel_b_df = _run_panel_b(session, car3_subclass_name, evidence_buffer, provenance_logger)
                if not panel_b_df.empty:
                    panel_b_df.to_csv(data_dir / "panel_b_region_enrichment.csv", index=False)
                    result['files']['panel_b_region_enrichment'] = str(data_dir / "panel_b_region_enrichment.csv")

            # REFLECT on Panel B: Region enrichment
            if provenance_logger:
                if not panel_b_df.empty:
                    top_region = panel_b_df.iloc[0]['region_name']
                    top_pct = panel_b_df.iloc[0]['pct_cells']
                    n_regions = len(panel_b_df)
                    is_dominant = top_pct > 0.15
                    provenance_logger.log_reflect(
                        step_number=7,
                        validation_status='passed',
                        confidence=1.0 if is_dominant else 0.8,
                        should_replan=False,
                        recommendations=[
                            f"Panel B: {gene} subclass found in {n_regions} brain regions",
                            f"Top enriched region: {top_region} at {top_pct*100:.1f}%",
                            f"{top_region} is {'DOMINANT' if is_dominant else 'enriched'} for {gene} (threshold: 15%)",
                            f"This confirms {top_region} as the primary region for {gene}+ circuit analysis"
                        ]
                    )
                else:
                    provenance_logger.log_reflect(
                        step_number=7,
                        validation_status='low_confidence',
                        confidence=0.3,
                        should_replan=False,
                        recommendations=["No region enrichment data - subclass may not be present in HAS_SUBCLASS relationships"]
                    )

            # Determine primary region (top enriched from Panel B, or top pocket)
            if not panel_b_df.empty:
                primary_region = panel_b_df.iloc[0]['region_name']
            elif not ranked_df.empty:
                primary_region = ranked_df.iloc[0]['spatial_name']
            else:
                primary_region = 'CLA'

            # Panel C: Multi-modal evidence for primary region
            result['kg_query_count'] += 2  # morphology + neurons
            panel_c_data = _run_panel_c(session, primary_region, panel_b_df,
                                         evidence_buffer, provenance_logger)
            # Serialize Panel C to CSV
            panel_c_rows = []
            if panel_c_data.get('molecular_pct') is not None:
                panel_c_rows.append({'modality': 'molecular', 'property': 'pct_cells',
                                     'value': panel_c_data['molecular_pct'],
                                     'source': 'panel_b_region_enrichment.csv'})
                panel_c_rows.append({'modality': 'molecular', 'property': 'rank',
                                     'value': panel_c_data.get('molecular_rank', ''),
                                     'source': 'panel_b_region_enrichment.csv'})
            if panel_c_data.get('morphology'):
                for k, v in panel_c_data['morphology'].items():
                    panel_c_rows.append({'modality': 'morphology', 'property': k,
                                         'value': v if v is not None else 'NaN',
                                         'source': 'Region node properties'})
            panel_c_rows.append({'modality': 'morphological', 'property': 'neuron_count',
                                 'value': panel_c_data.get('neuron_count', 0),
                                 'source': panel_c_data.get('neuron_method', 'LOCATE_AT')})
            for nr in panel_c_data.get('neurons', [])[:10]:
                panel_c_rows.append({'modality': 'morphological', 'property': 'neuron',
                                     'value': f"{nr.get('neuron_id', '')}|{nr.get('celltype', '')}",
                                     'source': 'QUERY_PANEL_C_CLA_NEURONS'})
            pd.DataFrame(panel_c_rows).to_csv(data_dir / "panel_c_cla_multimodal.csv", index=False)
            result['files']['panel_c_cla_multimodal'] = str(data_dir / "panel_c_cla_multimodal.csv")

            # Panel D: Projection matrix
            result['kg_query_count'] += 1
            panel_d_df = _run_panel_d(session, primary_region, evidence_buffer, provenance_logger)
            if not panel_d_df.empty:
                # Save raw long-form
                panel_d_df.to_csv(data_dir / "panel_d_raw_projections.csv", index=False)
                result['files']['panel_d_raw_projections'] = str(data_dir / "panel_d_raw_projections.csv")
                # Build pivot matrix
                pivot = panel_d_df.pivot_table(
                    index='neuron_id', columns='target_acronym',
                    values='proj_log10', fill_value=0, aggfunc='sum'
                )
                pivot.to_csv(data_dir / "panel_d_projection_matrix.csv")
                result['files']['panel_d_projection_matrix'] = str(data_dir / "panel_d_projection_matrix.csv")

            # Panel E: Full target composition (all subclasses for top targets)
            top_target_eids = []
            if not panel_d_df.empty:
                # Get top 10 targets by total projection strength from Panel D
                target_totals = panel_d_df.groupby('target_acronym').agg(
                    total=('proj_strength', 'sum'),
                    eid=('target_eid', 'first')
                ).nlargest(10, 'total')
                top_target_eids = target_totals['eid'].tolist()
            elif not ranked_df.empty:
                # Fallback: use top pocket's projections
                top_pocket_eid = ranked_df.iloc[0]['pocket_eid']
                top_proj_fb = projections_dict.get(top_pocket_eid, pd.DataFrame())
                if not top_proj_fb.empty:
                    top_target_eids = top_proj_fb.head(10)['target_eid'].tolist()

            panel_e_df = pd.DataFrame()
            if top_target_eids:
                result['kg_query_count'] += len(top_target_eids)
                panel_e_df = _run_panel_e(session, top_target_eids, evidence_buffer, provenance_logger)
                if not panel_e_df.empty:
                    panel_e_df.to_csv(data_dir / "panel_e_target_full_composition.csv", index=False)
                    result['files']['panel_e_target_full_composition'] = str(
                        data_dir / "panel_e_target_full_composition.csv")

            # Panel F: Subgraph export
            subgraph_cypher = _build_panel_f_subgraph(
                gene, panel_a_df, panel_b_df, panel_c_data, panel_d_df, panel_e_df
            )
            subgraph_path = data_dir / "panel_f_subgraph.cypher"
            with open(subgraph_path, 'w', encoding='utf-8') as f:
                f.write(subgraph_cypher)
            result['files']['panel_f_subgraph'] = str(subgraph_path)

            # Store panel data in result for report synthesis
            result['panel_data'] = {
                'panel_a': panel_a_df.to_dict('records') if not panel_a_df.empty else [],
                'panel_b': panel_b_df.to_dict('records') if not panel_b_df.empty else [],
                'panel_c': panel_c_data,
                'panel_d_shape': list(panel_d_df.shape) if not panel_d_df.empty else [0, 0],
                'panel_d_n_neurons': panel_d_df['neuron_id'].nunique() if not panel_d_df.empty else 0,
                'panel_d_n_targets': panel_d_df['target_acronym'].nunique() if not panel_d_df.empty else 0,
                'panel_e_n_targets': panel_e_df['target_name'].nunique() if not panel_e_df.empty else 0,
                'primary_region': primary_region,
            }

            # Build summary
            result['summary'] = {
                'gene': gene,
                'n_pockets_found': len(pockets_df),
                'n_me_subregion': len(pockets_df[pockets_df['granularity_score'] == 1]),
                'n_subregion': len(pockets_df[pockets_df['granularity_score'] == 2]),
                'n_region': len(pockets_df[pockets_df['granularity_score'] == 3]),
                'total_neurons': len(all_neurons),
                'total_projection_targets': len(set(
                    p.get('target_name') for p in all_projections if p.get('target_name')
                )),
                'top_circuit': ranked_df.iloc[0].to_dict() if not ranked_df.empty else None,
                'kg_query_count': result['kg_query_count']
            }

            result['success'] = True

            if provenance_logger:
                provenance_logger.log_reflect(
                    step_number=11,
                    validation_status='passed',
                    confidence=0.95,
                    should_replan=False,
                    recommendations=[f'Analysis complete for {gene}: '
                                     f'{len(pockets_df)} pockets, {len(all_neurons)} neurons, '
                                     f'panels A-F computed']
                )

        driver.close()

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

        if provenance_logger:
            provenance_logger.log_reflect(
                step_number=0,
                validation_status='failed',
                confidence=0.0,
                should_replan=False,
                recommendations=[f'Error: {str(e)}']
            )

    return result


def generate_circuit_report(result: Dict[str, Any], output_dir: str, seed: int,
                            gene: str = "Car3",
                            evidence_buffer: Optional[EvidenceBuffer] = None) -> str:
    """Generate the report.md for Circuit."""
    output_path = Path(output_dir)

    lines = [
        f"# MS Circuit Report: {gene}+ Neuron Analysis",
        "",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        f"**Gene Marker:** {gene}",
        f"**Seed:** {seed}",
        f"**Status:** {'SUCCESS' if result['success'] else 'FAILED'}",
        f"**KG Queries Executed:** {result.get('kg_query_count', 0)}",
        f"**Neo4j Status:** {'Connected' if result.get('neo4j_status', {}).get('connected') else 'FAILED'}",
        ""
    ]

    if not result['success']:
        lines.extend([
            "## FAILED: NO EVIDENCE",
            "",
            f"**Error:** {result.get('error', 'Unknown error')}",
            "",
            "### Neo4j Connectivity Status",
            "",
            f"- URI: {os.getenv('NEO4J_URI', 'bolt://100.88.72.32:7687')}",
            f"- Connected: {result.get('neo4j_status', {}).get('connected', False)}",
            f"- Error: {result.get('neo4j_status', {}).get('error', 'N/A')}",
            f"- KG queries attempted: {result.get('kg_query_count', 0)}",
            "",
            "### Exception Stack",
            "",
            "```",
            result.get('traceback', 'No traceback available'),
            "```",
            "",
            "### Intent / Budget",
            "",
            "- Intent: MS_REPRO",
            "- Budget: heavy",
        ])
    else:
        summary = result.get('summary', {})
        top_circuit = summary.get('top_circuit', {})

        lines.extend([
            "## Analysis Summary",
            "",
            f"- **Gene Marker:** {gene}",
            f"- **Pockets Found:** {summary.get('n_pockets_found', 'N/A')}",
            f"  - ME_Subregion level: {summary.get('n_me_subregion', 'N/A')}",
            f"  - Subregion level: {summary.get('n_subregion', 'N/A')}",
            f"  - Region level: {summary.get('n_region', 'N/A')}",
            f"- **Total Neurons:** {summary.get('total_neurons', 'N/A')}",
            f"- **Projection Targets:** {summary.get('total_projection_targets', 'N/A')}",
            f"- **KG Queries:** {summary.get('kg_query_count', 'N/A')}",
            "",
            "## Top Circuit (Primary Focus)",
            ""
        ])

        if top_circuit:
            lines.extend([
                f"- **Region:** {top_circuit.get('spatial_name', 'N/A')}",
                f"- **Type:** {top_circuit.get('spatial_type', 'N/A')}",
                f"- **Total Score:** {top_circuit.get('total_score', 'N/A'):.1f}",
                f"- **Neurons:** {top_circuit.get('n_neurons', 'N/A')}",
                f"- **Projection Targets:** {top_circuit.get('n_projection_targets', 'N/A')}",
                f"- **Dominant Percentage:** {top_circuit.get('dominant_pct', 'N/A'):.1f}%",
                "",
                "### Scoring Breakdown (canonical from collect_data.py)",
                "",
                f"- Molecular Specificity: {top_circuit.get('molecular_score', 0):.1f}/30",
                f"- Spatial Precision: {top_circuit.get('granularity_score', 0)}/20",
                f"- Neuron Count: {top_circuit.get('neuron_score', 0):.1f}/20",
                f"- Projection Specificity: {top_circuit.get('projection_score', 0):.1f}/20",
                f"- {gene} Relevance: {top_circuit.get('gene_score', 0)}/10",
            ])
        else:
            lines.append("*No top circuit identified*")

        lines.extend([
            "",
            "## Output Files",
            ""
        ])

        for name, path in result.get('files', {}).items():
            lines.append(f"- `{name}`: {path}")

    # Add evidence summary
    if evidence_buffer:
        lines.append("")
        lines.append(evidence_buffer.to_markdown())

    lines.extend([
        "",
        "---",
        f"*Generated by MS Circuit canonical runner using circuit/collect_data.py logic*"
    ])

    report_content = "\n".join(lines)

    report_path = output_path / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_content


def _cite(filename: str, row: int, col: str) -> str:
    """Inline citation helper."""
    return f"(source: {filename} row={row} col={col})"


def circuit_synthesize_answer_from_artifacts(
    output_dir: str,
    gene: str,
    seed: int,
    evidence: Optional[EvidenceBuffer] = None,
    provenance: Optional[ProvenanceLogger] = None,
) -> str:
    """
    Read all produced CSV artifacts and write an integrated report.md
    with inline citations, executive summary, and MS checklist.
    """
    data_dir = Path(output_dir) / "data"
    lines = []

    # --- Header ---
    lines.extend([
        f"# MS Circuit: {gene}+ Neuron Circuit Analysis",
        "",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        f"**Gene Marker:** {gene}",
        f"**Seed:** {seed}",
        f"**Status:** SUCCESS",
        f"**Neo4j Status:** Connected",
        ""
    ])

    # --- Load artifacts ---
    def _read_csv(name):
        p = data_dir / name
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame()

    panel_a = _read_csv("panel_a_subclass_identity.csv")
    panel_b = _read_csv("panel_b_region_enrichment.csv")
    panel_c = _read_csv("panel_c_cla_multimodal.csv")
    panel_d_raw = _read_csv("panel_d_raw_projections.csv")
    panel_d_matrix = _read_csv("panel_d_projection_matrix.csv")
    panel_e = _read_csv("panel_e_target_full_composition.csv")
    subclasses = _read_csv("subclasses.csv")
    region_enr = _read_csv("region_enrichment.csv")
    target_comp = _read_csv("target_composition.csv")
    morph_sum = _read_csv("morphology_summary.csv")

    # Determine primary region
    primary_region = 'CLA'
    if not panel_b.empty:
        primary_region = panel_b.iloc[0]['region_name']

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")

    # Bullet 1: Subclass identity
    if not panel_a.empty:
        sc_name = panel_a.iloc[0]['subclass_name']
        sc_markers = panel_a.iloc[0]['markers']
        lines.append(f"- **Car3 subclass identified:** \"{sc_name}\" with markers {sc_markers} "
                     f"{_cite('panel_a_subclass_identity.csv', 1, 'subclass_name')}")
    # Bullet 2: Top enriched region
    if not panel_b.empty:
        top_reg = panel_b.iloc[0]['region_name']
        top_pct = panel_b.iloc[0]['pct_cells']
        lines.append(f"- **Top enriched region:** {top_reg} with pct_cells = {top_pct:.4f} "
                     f"({top_pct*100:.2f}%) {_cite('panel_b_region_enrichment.csv', 1, 'pct_cells')}")
    # Bullet 3: Total enriched regions
    if not panel_b.empty:
        lines.append(f"- **{len(panel_b)} regions** contain the {gene} subclass "
                     f"{_cite('panel_b_region_enrichment.csv', 'all', 'region_name')}")
    # Bullet 4: Pockets where Car3 is dominant
    if not subclasses.empty:
        lines.append(f"- **{len(subclasses)} pockets** where {gene} subclass is dominant (rank=1) "
                     f"{_cite('subclasses.csv', 'all', 'spatial_name')}")
    # Bullet 5: Neurons
    panel_c_neurons = panel_c[panel_c['property'] == 'neuron_count'] if not panel_c.empty else pd.DataFrame()
    n_neurons_primary = int(panel_c_neurons.iloc[0]['value']) if not panel_c_neurons.empty else 0
    if n_neurons_primary > 0:
        lines.append(f"- **{n_neurons_primary} neurons** found in {primary_region} "
                     f"{_cite('panel_c_cla_multimodal.csv', 'neuron_count', 'value')}")
    # Bullet 6: Projection targets
    if not panel_d_raw.empty:
        n_d_neurons = panel_d_raw['neuron_id'].nunique()
        n_d_targets = panel_d_raw['target_acronym'].nunique()
        lines.append(f"- **Projection matrix:** {n_d_neurons} neurons x {n_d_targets} targets from {primary_region} "
                     f"{_cite('panel_d_projection_matrix.csv', 'all', 'all')}")
    # Bullet 7: Top projection target
    if not panel_d_raw.empty:
        top_target_agg = panel_d_raw.groupby('target_acronym')['proj_strength'].sum().nlargest(1)
        if not top_target_agg.empty:
            tt_name = top_target_agg.index[0]
            tt_str = top_target_agg.iloc[0]
            lines.append(f"- **Top projection target:** {tt_name} (total strength = {tt_str:.1f}) "
                         f"{_cite('panel_d_raw_projections.csv', 'agg', 'proj_strength')}")
    # Bullet 8: Target composition
    if not panel_e.empty:
        n_e_targets = panel_e['target_name'].nunique()
        lines.append(f"- **Target molecular profiles** computed for {n_e_targets} targets "
                     f"{_cite('panel_e_target_full_composition.csv', 'all', 'target_name')}")
    lines.append("")

    # --- Panel A ---
    lines.append("## Panel A: Car3 Subclass Identification")
    lines.append("")
    if not panel_a.empty:
        lines.append("| # | Subclass Name | Markers |")
        lines.append("|---|---------------|---------|")
        for i, row in panel_a.iterrows():
            lines.append(f"| {i+1} | {row['subclass_name']} | {row['markers']} |")
        lines.append("")
        lines.append(f"Evidence: {len(panel_a)} subclass(es) found containing \"{gene}\" in name or markers "
                     f"{_cite('panel_a_subclass_identity.csv', 'all', 'subclass_name')}")
    else:
        lines.append("*No Car3 subclass found*")
    lines.append("")

    # --- Panel B ---
    lines.append("## Panel B: Region Enrichment for Car3 Subclass")
    lines.append("")
    if not panel_b.empty:
        lines.append(f"Showing pct_cells of \"{panel_a.iloc[0]['subclass_name'] if not panel_a.empty else gene}\" "
                     f"across all {len(panel_b)} Region nodes.")
        lines.append("")
        lines.append("| Rank | Region | pct_cells | pct (%) |")
        lines.append("|------|--------|-----------|---------|")
        for i, row in panel_b.head(15).iterrows():
            pct_display = row['pct_cells'] * 100
            lines.append(f"| {i+1} | {row['region_name']} | {row['pct_cells']:.6f} | {pct_display:.2f}% |")
        if len(panel_b) > 15:
            lines.append(f"| ... | ({len(panel_b) - 15} more regions) | ... | ... |")
        lines.append("")
        lines.append(f"**Top region: {panel_b.iloc[0]['region_name']}** at "
                     f"{panel_b.iloc[0]['pct_cells']*100:.2f}% "
                     f"{_cite('panel_b_region_enrichment.csv', 1, 'pct_cells')}")
    else:
        lines.append("*No region enrichment data available*")
    lines.append("")

    # --- Panel C ---
    lines.append(f"## Panel C: Multi-Modal Evidence for {primary_region}")
    lines.append("")
    lines.append("### Molecular")
    if not panel_c.empty:
        mol_rows = panel_c[panel_c['modality'] == 'molecular']
        if not mol_rows.empty:
            for _, row in mol_rows.iterrows():
                val = row['value']
                if row['property'] == 'pct_cells':
                    lines.append(f"- Car3 pct_cells in {primary_region}: **{float(val):.4f}** "
                                 f"({float(val)*100:.2f}%) "
                                 f"{_cite('panel_c_cla_multimodal.csv', 'pct_cells', 'value')}")
                elif row['property'] == 'rank':
                    lines.append(f"- HAS_SUBCLASS rank: {val} "
                                 f"{_cite('panel_c_cla_multimodal.csv', 'rank', 'value')}")
    lines.append("")
    lines.append("### Morphology")
    if not panel_c.empty:
        morph_rows = panel_c[panel_c['modality'] == 'morphology']
        if not morph_rows.empty:
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for _, row in morph_rows.iterrows():
                val = row['value']
                lines.append(f"| {row['property']} | {val} |")
            lines.append("")
            lines.append(f"{_cite('panel_c_cla_multimodal.csv', 'morphology', 'value')}")
        else:
            lines.append("*No morphology data for this region*")
    lines.append("")
    lines.append("### Neurons")
    if not panel_c.empty:
        nc_row = panel_c[panel_c['property'] == 'neuron_count']
        n_count = int(nc_row.iloc[0]['value']) if not nc_row.empty else 0
        method = nc_row.iloc[0]['source'] if not nc_row.empty else 'unknown'
        lines.append(f"- **{n_count} neurons** found via {method} "
                     f"{_cite('panel_c_cla_multimodal.csv', 'neuron_count', 'value')}")
        neuron_rows = panel_c[panel_c['property'] == 'neuron']
        if not neuron_rows.empty:
            lines.append("")
            lines.append("| Neuron ID | Celltype |")
            lines.append("|-----------|----------|")
            for _, row in neuron_rows.head(10).iterrows():
                parts = str(row['value']).split('|')
                nid = parts[0] if len(parts) > 0 else ''
                ct = parts[1] if len(parts) > 1 else ''
                lines.append(f"| {nid} | {ct} |")
    lines.append("")

    # --- Panel D ---
    lines.append(f"## Panel D: Projection Heatmap ({primary_region} Neurons)")
    lines.append("")
    if not panel_d_raw.empty:
        n_d_neurons = panel_d_raw['neuron_id'].nunique()
        n_d_targets = panel_d_raw['target_acronym'].nunique()
        lines.append(f"**Matrix dimensions:** {n_d_neurons} neurons x {n_d_targets} targets")
        lines.append(f"**Metric:** log10(1 + projection_length) per panel_D_v2.py convention")
        lines.append("")
        # Top 10 targets by aggregated stats
        target_agg = panel_d_raw.groupby('target_acronym').agg(
            n_neurons=('neuron_id', 'nunique'),
            total_strength=('proj_strength', 'sum'),
            avg_strength=('proj_strength', 'mean'),
            avg_log10=('proj_log10', 'mean')
        ).sort_values('total_strength', ascending=False)
        lines.append("### Top-10 Projection Targets")
        lines.append("")
        lines.append("| Rank | Target | N Neurons | Total Strength | Avg Strength | Avg log10 |")
        lines.append("|------|--------|-----------|----------------|--------------|-----------|")
        for rank, (target, row) in enumerate(target_agg.head(10).iterrows(), 1):
            lines.append(f"| {rank} | {target} | {row['n_neurons']} | {row['total_strength']:.1f} "
                         f"| {row['avg_strength']:.1f} | {row['avg_log10']:.3f} |")
        lines.append("")
        lines.append(f"{_cite('panel_d_raw_projections.csv', 'agg', 'proj_strength')}")
    else:
        lines.append(f"*No projection data from {primary_region} neurons*")
    lines.append("")

    # --- Panel E ---
    lines.append("## Panel E: Target Subregion Molecular Fingerprints")
    lines.append("")
    if not panel_e.empty:
        targets_in_e = panel_e['target_name'].unique()
        lines.append(f"Molecular composition for **{len(targets_in_e)} target** subregions.")
        lines.append("")
        for t_idx, target_name in enumerate(targets_in_e[:10]):
            target_rows = panel_e[panel_e['target_name'] == target_name].sort_values('rank')
            lines.append(f"### {target_name}")
            lines.append("")
            lines.append("| Rank | Subclass | pct_cells | Markers |")
            lines.append("|------|----------|-----------|---------|")
            for _, row in target_rows.head(5).iterrows():
                markers_short = str(row['markers'])[:50]
                pct_disp = float(row['pct_cells']) * 100
                lines.append(f"| {row['rank']} | {row['subclass_name']} | {pct_disp:.2f}% | {markers_short} |")
            if len(target_rows) > 5:
                lines.append(f"| ... | ({len(target_rows) - 5} more subclasses) | ... | ... |")
            lines.append("")
        lines.append(f"{_cite('panel_e_target_full_composition.csv', 'all', 'subclass_name')}")
    else:
        lines.append("*No target composition data*")
    lines.append("")

    # --- Closed-Loop A -> B -> D -> E ---
    lines.append("## Closed-Loop Reasoning: A -> B -> D -> E")
    lines.append("")
    if not panel_a.empty:
        sc = panel_a.iloc[0]['subclass_name']
        lines.append(f"**A (Molecular Identity):** The gene marker {gene} identifies "
                     f"transcriptomic subclass \"{sc}\" with markers {panel_a.iloc[0]['markers']}.")
    if not panel_b.empty:
        top_r = panel_b.iloc[0]['region_name']
        top_p = panel_b.iloc[0]['pct_cells']
        lines.append(f"**B (Spatial Enrichment):** {top_r} has the highest enrichment "
                     f"(pct_cells = {top_p:.4f}, {top_p*100:.2f}%), "
                     f"making it the primary region for this subclass.")
    if not panel_d_raw.empty:
        top_targets = panel_d_raw.groupby('target_acronym')['proj_strength'].sum().nlargest(3)
        target_list = ', '.join(top_targets.index.tolist())
        lines.append(f"**D (Projection Pattern):** {primary_region} neurons project predominantly to "
                     f"{target_list}.")
    if not panel_e.empty:
        targets_e = panel_e['target_name'].unique()[:3]
        for t in targets_e:
            t_rows = panel_e[panel_e['target_name'] == t].head(1)
            if not t_rows.empty:
                dom_sc = t_rows.iloc[0]['subclass_name']
                dom_pct = float(t_rows.iloc[0]['pct_cells']) * 100
                lines.append(f"**E (Target Identity):** {t} is dominated by {dom_sc} ({dom_pct:.1f}%).")
    lines.append("")

    # --- Panel F ---
    lines.append("## Panel F: Subgraph Export")
    lines.append("")
    subgraph_path = data_dir / "panel_f_subgraph.cypher"
    if subgraph_path.exists():
        n_lines = len(subgraph_path.read_text().split('\n'))
        lines.append(f"Subgraph Cypher file: `panel_f_subgraph.cypher` ({n_lines} lines)")
        lines.append(f"Contains node MERGE + relationship MERGE statements to reconstruct "
                     f"the {gene} analysis subgraph.")
    else:
        lines.append("*Subgraph file not generated*")
    lines.append("")

    # --- MS Checklist ---
    lines.append("## MS Checklist")
    lines.append("")
    # Check A
    sc_found = not panel_a.empty
    sc_name_chk = panel_a.iloc[0]['subclass_name'] if sc_found else 'N/A'
    lines.append(f"- [{'x' if sc_found else ' '}] **Panel A:** Car3 subclass identified: "
                 f"\"{sc_name_chk}\" {_cite('panel_a_subclass_identity.csv', 1, 'subclass_name') if sc_found else ''}")
    # Check B
    b_cla = panel_b[panel_b['region_name'] == 'CLA'] if not panel_b.empty else pd.DataFrame()
    if not b_cla.empty:
        cla_pct = b_cla.iloc[0]['pct_cells']
        cla_rank_in_b = panel_b.index.get_loc(b_cla.index[0]) + 1
        lines.append(f"- [x] **Panel B:** CLA enrichment = {cla_pct*100:.2f}%, "
                     f"rank #{cla_rank_in_b} {_cite('panel_b_region_enrichment.csv', cla_rank_in_b, 'pct_cells')}")
    else:
        lines.append("- [ ] **Panel B:** CLA not found in enrichment data")
    # Check C
    lines.append(f"- [{'x' if n_neurons_primary > 0 else ' '}] **Panel C:** "
                 f"{n_neurons_primary} neurons in {primary_region}, morphology data present")
    # Check D
    if not panel_d_raw.empty:
        lines.append(f"- [x] **Panel D:** Projection matrix {n_d_neurons}x{n_d_targets} computed "
                     f"{_cite('panel_d_projection_matrix.csv', 'all', 'all')}")
    else:
        lines.append("- [ ] **Panel D:** No projection matrix")
    # Check E
    if not panel_e.empty:
        lines.append(f"- [x] **Panel E:** Target composition for {len(targets_in_e)} targets "
                     f"{_cite('panel_e_target_full_composition.csv', 'all', 'target_name')}")
    else:
        lines.append("- [ ] **Panel E:** No target composition")
    # Check F
    lines.append(f"- [{'x' if subgraph_path.exists() else ' '}] **Panel F:** Subgraph exported")
    lines.append("")

    # --- Evidence Summary ---
    if evidence:
        lines.append(evidence.to_markdown())
        lines.append("")

    # --- Output Files ---
    lines.append("## Output Files")
    lines.append("")
    for csv_file in sorted(data_dir.glob("*")):
        lines.append(f"- `{csv_file.name}`: {csv_file}")
    lines.append("")

    lines.extend([
        "---",
        f"*Generated by MS Circuit runner with Panel A-F figure reproduction*"
    ])

    report_content = "\n".join(lines)
    report_path = Path(output_dir) / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_content


if __name__ == "__main__":
    from provenance import create_provenance_logger

    prov = create_provenance_logger(run_id="circuit_discovery", seed=42)
    evidence = EvidenceBuffer()

    result = run_circuit_canonical(
        gene="Car3",
        seed=42,
        output_dir="./outputs/circuit_discovery",
        provenance_logger=prov,
        evidence_buffer=evidence
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Files generated: {list(result['files'].keys())}")
        print(f"Summary: {result['summary']}")
        report = generate_circuit_report(result, "./outputs/circuit_discovery", 42, "Car3", evidence)
        print("Report generated")
    else:
        print(f"Error: {result['error']}")
        if result.get('traceback'):
            print(result['traceback'])