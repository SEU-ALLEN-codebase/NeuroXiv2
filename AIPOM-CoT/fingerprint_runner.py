import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import zscore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from provenance import ProvenanceLogger, EventType
from evidence_buffer import EvidenceBuffer
from tpar_reasoner import TPARReasoner, reason_about_similarity

logger = logging.getLogger(__name__)

# Font config for Chinese support in plots
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# KG Extractor Functions  (self-contained, no fingerprint.py import)
# Cypher queries replicated exactly from fingerprint.py
# ============================================================

def _kg_get_all_subclasses(session) -> List[str]:
    """Get the global sorted list of all subclass names.

    Cypher: MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass)
            RETURN DISTINCT sc.name ORDER BY sc.name
    """
    query = """
    MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass)
    RETURN DISTINCT sc.name AS subclass_name
    ORDER BY subclass_name
    """
    result = session.run(query)
    return [rec['subclass_name'] for rec in result]


def _kg_get_all_target_subregions(session) -> List[str]:
    """Get the global sorted list of all projection target subregion acronyms.

    Cypher: MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion)
            WHERE p.weight IS NOT NULL AND p.weight > 0
            RETURN DISTINCT t.acronym ORDER BY t.acronym
    """
    query = """
    MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion)
    WHERE p.weight IS NOT NULL AND p.weight > 0
    RETURN DISTINCT t.acronym AS target_subregion
    ORDER BY target_subregion
    """
    result = session.run(query)
    return [rec['target_subregion'] for rec in result]


def _kg_get_all_regions(session) -> List[str]:
    """Get all region acronyms that have molecular data (HAS_SUBCLASS edges).

    Cypher: MATCH (r:Region) WHERE EXISTS((r)-[:HAS_SUBCLASS]->())
            RETURN r.acronym ORDER BY r.acronym
    """
    query = """
    MATCH (r:Region)
    WHERE EXISTS((r)-[:HAS_SUBCLASS]->())
    RETURN r.acronym AS acronym
    ORDER BY acronym
    """
    result = session.run(query)
    return [rec['acronym'] for rec in result]


def _kg_compute_molecular_signature(session, region: str,
                                     all_subclasses: List[str]) -> np.ndarray:
    """Compute molecular fingerprint for one region.

    Returns fixed-dimension vector (len=all_subclasses) of pct_cells values.
    Zero-fills missing subclasses.

    Cypher: MATCH (r:Region {acronym: $region})-[hs:HAS_SUBCLASS]->(sc:Subclass)
            RETURN sc.name, hs.pct_cells
    """
    query = """
    MATCH (r:Region {acronym: $region})
    MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
    RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
    ORDER BY subclass_name
    """
    result = session.run(query, region=region)
    data = {rec['subclass_name']: rec['pct_cells'] for rec in result}

    signature = np.zeros(len(all_subclasses))
    for i, sc in enumerate(all_subclasses):
        if sc in data:
            signature[i] = data[sc]
    return signature


def _kg_compute_morphology_signature(session, region: str) -> np.ndarray:
    """Compute 8-dim morphology fingerprint for one region (raw, pre-zscore).

    Features (exact same order as fingerprint.py):
      [0] axonal_bifurcation_remote_angle
      [1] axonal_length
      [2] axonal_branches
      [3] axonal_maximum_branch_order
      [4] dendritic_bifurcation_remote_angle
      [5] dendritic_length
      [6] dendritic_branches
      [7] dendritic_maximum_branch_order

    Returns NaN array if region node has no data.
    """
    query = """
    MATCH (r:Region {acronym: $region})
    RETURN
      r.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
      r.axonal_length AS axonal_length,
      r.axonal_branches AS axonal_branches,
      r.axonal_maximum_branch_order AS axonal_max_branch_order,
      r.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
      r.dendritic_length AS dendritic_length,
      r.dendritic_branches AS dendritic_branches,
      r.dendritic_maximum_branch_order AS dendritic_max_branch_order
    """
    result = session.run(query, region=region)
    record = result.single()

    if not record:
        return np.array([np.nan] * 8)

    features = [
        'axonal_bifurcation_remote_angle',
        'axonal_length',
        'axonal_branches',
        'axonal_max_branch_order',
        'dendritic_bifurcation_remote_angle',
        'dendritic_length',
        'dendritic_branches',
        'dendritic_max_branch_order',
    ]
    return np.array([record[f] if record[f] is not None else np.nan
                     for f in features])


def _kg_compute_projection_signature(session, region: str,
                                      all_targets: List[str]) -> np.ndarray:
    """Compute projection fingerprint for one region.

    1. Find all neurons in the region (LOCATE_AT / LOCATE_AT_SUBREGION / LOCATE_AT_ME_SUBREGION)
    2. For each neuron, aggregate PROJECT_TO weights per target subregion
    3. Apply log10(1+x) stabilisation then L1 normalize.

    Returns fixed-dimension vector (len=all_targets).
    """
    query = """
    MATCH (r:Region {acronym: $region})
    OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT]->(r)
    OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r)
    OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r)
    WITH r, (COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) + COLLECT(DISTINCT n3)) AS ns
    UNWIND ns AS n
    WITH DISTINCT n
    WHERE n IS NOT NULL
    MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
    WHERE p.weight IS NOT NULL AND p.weight > 0
    WITH t.acronym AS tgt_subregion, SUM(p.weight) AS total_weight_to_tgt
    RETURN tgt_subregion, total_weight_to_tgt
    ORDER BY total_weight_to_tgt DESC
    """
    result = session.run(query, region=region)
    data = {rec['tgt_subregion']: rec['total_weight_to_tgt'] for rec in result}

    raw_values = np.zeros(len(all_targets))
    for i, tgt in enumerate(all_targets):
        if tgt in data:
            raw_values[i] = data[tgt]

    # Log stabilisation
    log_values = np.log10(1 + raw_values)

    # L1 normalize → probability distribution
    total = log_values.sum()
    if total > 0:
        return log_values / (total + 1e-9)
    return log_values


def _kg_select_top_regions(session, n: int = 30) -> List[str]:
    """Select top N regions by neuron count (LOCATE_AT).

    Cypher: MATCH (r:Region) OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
            WITH r, COUNT(DISTINCT n) AS nc WHERE nc > 0
            RETURN r.acronym ORDER BY nc DESC LIMIT $n
    """
    query = """
    MATCH (r:Region)
    OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
    WITH r, COUNT(DISTINCT n) AS neuron_count
    WHERE neuron_count > 0
    RETURN r.acronym AS region, neuron_count
    ORDER BY neuron_count DESC
    LIMIT $n
    """
    result = session.run(query, n=n)
    return [rec['region'] for rec in result]


# ============================================================
# Normalization helpers (matching fingerprint.py exactly)
# ============================================================

def _apply_morphology_zscore(regions: List[str],
                              morph_signatures: Dict[str, np.ndarray]):
    """Apply cross-region per-feature z-score normalization on morphology.

    Exactly matches fingerprint.py compute_all_morphology_signatures():
    1. Fix dimension to 8
    2. Convert zero dendritic features (indices 4-7) to NaN
    3. Per-feature z-score (ignoring NaN)
    """
    # Ensure 8-dim
    for r in regions:
        sig = morph_signatures[r]
        if len(sig) != 8:
            fixed = np.array([np.nan] * 8)
            fixed[:min(len(sig), 8)] = sig[:min(len(sig), 8)]
            morph_signatures[r] = fixed

    all_sigs = np.array([morph_signatures[r] for r in regions])

    # Zero dendritic → NaN (indices 4-7)
    for i in [4, 5, 6, 7]:
        col = all_sigs[:, i].copy()
        col[np.abs(col) < 1e-6] = np.nan
        all_sigs[:, i] = col

    # Per-feature z-score
    for i in range(8):
        col = all_sigs[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            col[valid] = zscore(col[valid])
            all_sigs[:, i] = col

    # Write back
    for idx, r in enumerate(regions):
        morph_signatures[r] = all_sigs[idx]


def _minmax_normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Min-Max normalize a DataFrame (matching fingerprint.py canonical normalization)."""
    values = df.values
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return df
    vmin = values[valid].min()
    vmax = values[valid].max()
    if vmax - vmin < 1e-9:
        return pd.DataFrame(np.zeros_like(values), index=df.index, columns=df.columns)
    normalized = (values - vmin) / (vmax - vmin)
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def _compute_distance_matrices(valid_regions: List[str],
                                mol_sigs, morph_sigs, proj_sigs):
    """Compute cosine (mol, proj) and euclidean (morph) distance matrices."""
    n = len(valid_regions)
    mol_dist = np.zeros((n, n))
    morph_dist = np.zeros((n, n))
    proj_dist = np.zeros((n, n))

    for i, ra in enumerate(valid_regions):
        for j, rb in enumerate(valid_regions):
            if i == j:
                continue
            # Molecular: cosine distance
            try:
                mol_dist[i, j] = cosine(mol_sigs[ra], mol_sigs[rb])
            except Exception:
                mol_dist[i, j] = np.nan
            # Morphology: euclidean on z-scored
            sa, sb = morph_sigs[ra], morph_sigs[rb]
            if not np.any(np.isnan(sa)) and not np.any(np.isnan(sb)):
                morph_dist[i, j] = euclidean(sa, sb)
            else:
                morph_dist[i, j] = np.nan
            # Projection: cosine distance
            try:
                proj_dist[i, j] = cosine(proj_sigs[ra], proj_sigs[rb])
            except Exception:
                proj_dist[i, j] = np.nan

    return (
        pd.DataFrame(mol_dist, index=valid_regions, columns=valid_regions),
        pd.DataFrame(morph_dist, index=valid_regions, columns=valid_regions),
        pd.DataFrame(proj_dist, index=valid_regions, columns=valid_regions),
    )


def _extract_top_mismatch_pairs(mismatch_df: pd.DataFrame,
                                 regions: List[str], n: int = 10):
    """Extract top-N mismatch region pairs from upper triangle."""
    pairs = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            val = mismatch_df.iloc[i, j]
            if not np.isnan(val):
                pairs.append((regions[i], regions[j], float(val)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n]


def _plot_heatmaps(mol_sim, morph_sim, proj_sim,
                    mol_morph_mm, mol_proj_mm,
                    figures_dir: str):
    """Generate canonical heatmap plots (matching fingerprint.py visualize_matrices)."""
    fdir = Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    # Combined 2x3
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Brain Region Similarity and Mismatch Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    for ax, data, title in [
        (axes[0, 0], mol_sim, 'Molecular Similarity'),
        (axes[0, 1], morph_sim, 'Morphology Similarity'),
        (axes[0, 2], proj_sim, 'Projection Similarity'),
        (axes[1, 0], mol_morph_mm, 'Mol-Morph Mismatch'),
        (axes[1, 1], mol_proj_mm, 'Mol-Proj Mismatch'),
    ]:
        lbl = 'Mismatch' if 'Mismatch' in title else 'Similarity'
        sns.heatmap(data, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': lbl},
                    xticklabels=True, yticklabels=True)
        ax.set_title(title, fontsize=14, fontweight='bold')

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(str(fdir / 'all_matrices_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Individual plots
    _names = [
        ('1_molecular_similarity', mol_sim, 'Molecular fingerprint Similarity', 'Similarity'),
        ('2_morphology_similarity', morph_sim, 'Morphology fingerprint Similarity', 'Similarity'),
        ('3_projection_similarity', proj_sim, 'Projection fingerprint Similarity', 'Similarity'),
        ('4_mol_morph_mismatch', mol_morph_mm, 'Molecular-Morphology Mismatch', 'Mismatch'),
        ('5_mol_proj_mismatch', mol_proj_mm, 'Molecular-Projection Mismatch', 'Mismatch'),
    ]
    for fname, data, title, lbl in _names:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': lbl},
                    xticklabels=True, yticklabels=True, annot=False)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel('Region', fontsize=20, fontweight='bold')
        ax.set_ylabel('Region', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(fdir / f'{fname}.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# Neo4j connectivity check
# ============================================================

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
# Main runner
# ============================================================

def run_fingerprint_canonical(
    seed: int = 42,
    output_dir: str = "./outputs/fingerprint_analysis",
    provenance_logger: Optional[ProvenanceLogger] = None,
    evidence_buffer: Optional[EvidenceBuffer] = None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
    top_n_regions: int = 30
) -> Dict[str, Any]:
    """
    Run the canonical Fingerprint analysis via self-contained KG extractors.

    Same computation as fingerprint.py run_full_analysis() but using direct
    KG queries instead of importing fingerprint.BrainRegionFingerprints.

    Every KG query is recorded in evidence_buffer for provenance.
    """
    import random
    from neo4j import GraphDatabase

    # Determinism
    random.seed(seed)
    np.random.seed(seed)

    # Neo4j credentials
    uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neuroxiv")

    # Output dirs
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    figures_dir = output_path / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'success': False,
        'error': None,
        'files': {},
        'summary': {},
        'neo4j_status': None,
    }

    # Connectivity check (fail-fast)
    neo4j_status = _check_neo4j_connectivity(uri, user, password)
    result['neo4j_status'] = neo4j_status
    if not neo4j_status['connected']:
        result['error'] = f"Neo4j connectivity failed: {neo4j_status['error']}"
        if provenance_logger:
            provenance_logger.log_think("Neo4j connectivity FAILED",
                                        {'uri': uri, 'error': neo4j_status['error']})
        return result

    prov = provenance_logger  # shorthand
    eb = evidence_buffer

    if prov:
        prov.log_think("Starting canonical Fingerprint analysis (self-contained KG extractors)", {
            'neo4j_uri': uri, 'top_n_regions': top_n_regions, 'seed': seed,
        })
        prov.log_plan([
            {'step': 1, 'purpose': 'KG: Get global subclasses'},
            {'step': 2, 'purpose': 'KG: Get global target subregions'},
            {'step': 3, 'purpose': 'KG: Get all regions with HAS_SUBCLASS'},
            {'step': 4, 'purpose': 'KG: Compute molecular signatures (per-region)'},
            {'step': 5, 'purpose': 'KG: Compute morphology signatures (per-region) + z-score'},
            {'step': 6, 'purpose': 'KG: Compute projection signatures (per-region) + log+L1'},
            {'step': 7, 'purpose': 'KG: Select top regions by neuron count'},
            {'step': 8, 'purpose': 'Compute distance/similarity/mismatch matrices + heatmaps'},
            {'step': 9, 'purpose': 'Save CSV outputs'},
        ], planner_type='fingerprint_kg_extractors')

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:

            # --- Step 1: Global subclasses ---
            q1 = 'MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass) RETURN DISTINCT sc.name ORDER BY sc.name'
            if prov:
                prov.log_act(1, 'kg_query', 'Get all subclasses', query=q1)
            all_subclasses = _kg_get_all_subclasses(session)
            if eb:
                eb.add_evidence('molecular', 1, q1,
                                [{'subclass': s} for s in all_subclasses[:10]],
                                ['subclass'])
                eb.record_entity_count('Subclass', len(all_subclasses))

            # --- Step 2: Global target subregions ---
            q2 = 'MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion) WHERE p.weight>0 RETURN DISTINCT t.acronym'
            if prov:
                prov.log_act(2, 'kg_query', 'Get all target subregions', query=q2)
            all_targets = _kg_get_all_target_subregions(session)
            if eb:
                eb.add_evidence('projection', 2, q2,
                                [{'target': t} for t in all_targets[:10]],
                                ['target'])
                eb.record_entity_count('TargetSubregion', len(all_targets))

            # --- Step 3: All regions with molecular data ---
            q3 = 'MATCH (r:Region) WHERE EXISTS((r)-[:HAS_SUBCLASS]->()) RETURN r.acronym'
            if prov:
                prov.log_act(3, 'kg_query', 'Get all regions with HAS_SUBCLASS', query=q3)
            regions = _kg_get_all_regions(session)
            if eb:
                eb.add_evidence('molecular', 3, q3,
                                [{'region': r} for r in regions[:10]],
                                ['region'])
                eb.record_entity_count('Region', len(regions))

            # --- Step 4: Molecular signatures ---
            mol_sigs: Dict[str, np.ndarray] = {}
            q4 = 'MATCH (r:Region {acronym: $region})-[hs:HAS_SUBCLASS]->(sc:Subclass) RETURN sc.name, hs.pct_cells'
            if prov:
                prov.log_act(4, 'kg_query', 'Compute molecular signatures for all regions',
                             query=q4, params={'n_regions': len(regions)})
            for r in regions:
                mol_sigs[r] = _kg_compute_molecular_signature(session, r, all_subclasses)
            if eb:
                eb.add_evidence('molecular', 4,
                                f'molecular_signature per region ({len(regions)} queries)',
                                [{'region': r, 'n_nonzero': int(np.count_nonzero(mol_sigs[r]))}
                                 for r in regions[:10]],
                                ['region', 'n_nonzero'])

            # --- Step 5: Morphology signatures + z-score ---
            morph_sigs: Dict[str, np.ndarray] = {}
            q5 = 'MATCH (r:Region {acronym: $region}) RETURN r.axonal_* , r.dendritic_*'
            if prov:
                prov.log_act(5, 'kg_query', 'Compute morphology signatures + z-score',
                             query=q5, params={'n_regions': len(regions)})
            for r in regions:
                morph_sigs[r] = _kg_compute_morphology_signature(session, r)
            _apply_morphology_zscore(regions, morph_sigs)
            if eb:
                eb.add_evidence('morphological', 5,
                                f'morphology_signature per region ({len(regions)} queries, z-scored)',
                                [{'region': r} for r in regions[:10]],
                                ['region', '8d_zscore_vector'])

            # --- Step 6: Projection signatures ---
            proj_sigs: Dict[str, np.ndarray] = {}
            q6 = ('MATCH (r:Region {acronym:$region}) ... '
                  'MATCH (n)-[p:PROJECT_TO]->(t:Subregion) ... log+L1 norm')
            if prov:
                prov.log_act(6, 'kg_query', 'Compute projection signatures (log+L1)',
                             query=q6, params={'n_regions': len(regions)})
            for r in regions:
                proj_sigs[r] = _kg_compute_projection_signature(session, r, all_targets)
            if eb:
                eb.add_evidence('projection', 6,
                                f'projection_signature per region ({len(regions)} queries, log+L1)',
                                [{'region': r} for r in regions[:10]],
                                ['region', 'log_normalized_vector'])

            # --- Step 7: Top regions ---
            q7 = 'MATCH (r:Region) OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r) ... ORDER BY nc DESC'
            if prov:
                prov.log_act(7, 'kg_query', 'Select top regions by neuron count',
                             query=q7, params={'top_n': top_n_regions})
            top_regions = _kg_select_top_regions(session, top_n_regions)
            if eb:
                eb.add_evidence('molecular', 7,
                                f'SELECT top {top_n_regions} regions by neuron count',
                                [{'region': r} for r in top_regions[:10]],
                                ['region'])

            # REFLECT on Steps 1-7: Data extraction phase
            if prov:
                prov.log_reflect(
                    step_number=7,
                    validation_status='passed' if len(top_regions) > 0 else 'failed',
                    confidence=min(1.0, len(top_regions) / top_n_regions),
                    should_replan=len(top_regions) == 0,
                    recommendations=[
                        f"Extracted signatures for {len(regions)} total regions",
                        f"Selected top {len(top_regions)} regions by neuron count for analysis",
                        f"Molecular: {len(all_subclasses)} subclasses as feature dimensions",
                        f"Morphology: 8-dimensional z-scored feature vectors",
                        f"Projection: {len(all_targets)} target subregions as feature dimensions",
                        "Data extraction complete - proceeding to distance computation"
                    ]
                )

        driver.close()

        # --- Step 8: Distance / similarity / mismatch + heatmaps ---
        valid_regions = [r for r in top_regions if r in regions]
        if prov:
            prov.log_act(8, 'computation',
                         'Distance matrices + similarity + mismatch + heatmaps',
                         params={'n_valid_regions': len(valid_regions)})

        mol_dist_df, morph_dist_df, proj_dist_df = _compute_distance_matrices(
            valid_regions, mol_sigs, morph_sigs, proj_sigs)

        # Canonical normalization: similarity = 1 - minmax(distance)
        mol_sim = 1 - _minmax_normalize_df(mol_dist_df)
        morph_sim = 1 - _minmax_normalize_df(morph_dist_df)
        proj_sim = 1 - _minmax_normalize_df(proj_dist_df)

        # Mismatch = |sim1 - sim2| (on normalised distances, same as fingerprint.py)
        mol_norm = _minmax_normalize_df(mol_dist_df)
        morph_norm = _minmax_normalize_df(morph_dist_df)
        proj_norm = _minmax_normalize_df(proj_dist_df)
        mol_morph_mismatch = np.abs(mol_norm - morph_norm)
        mol_proj_mismatch = np.abs(mol_norm - proj_norm)

        # REFLECT on Step 8: Analyze similarity patterns
        if prov:
            # Compute insights from similarity matrices
            mol_mean = np.nanmean(mol_sim.values[~np.eye(mol_sim.shape[0], dtype=bool)])
            morph_mean = np.nanmean(morph_sim.values[~np.eye(morph_sim.shape[0], dtype=bool)])
            proj_mean = np.nanmean(proj_sim.values[~np.eye(proj_sim.shape[0], dtype=bool)])

            # Find highest mismatch
            mm_max = np.nanmax(mol_morph_mismatch.values[~np.eye(mol_morph_mismatch.shape[0], dtype=bool)])
            mp_max = np.nanmax(mol_proj_mismatch.values[~np.eye(mol_proj_mismatch.shape[0], dtype=bool)])

            prov.log_reflect(
                step_number=8,
                validation_status='passed',
                confidence=1.0,
                should_replan=False,
                recommendations=[
                    f"Computed {len(valid_regions)}x{len(valid_regions)} similarity matrices",
                    f"Mean molecular similarity: {mol_mean:.3f}",
                    f"Mean morphology similarity: {morph_mean:.3f}",
                    f"Mean projection similarity: {proj_mean:.3f}",
                    f"Max mol-morph mismatch: {mm_max:.3f} (regions with different molecular vs morphological profiles)",
                    f"Max mol-proj mismatch: {mp_max:.3f} (regions with different molecular vs projection profiles)",
                    "Mismatch highlights regions where cross-modal signatures diverge"
                ]
            )

        # Heatmaps
        _plot_heatmaps(mol_sim, morph_sim, proj_sim,
                       mol_morph_mismatch, mol_proj_mismatch,
                       str(figures_dir))
        for fig_file in figures_dir.glob("*.png"):
            result['files'][fig_file.stem] = str(fig_file)

        # --- Step 9: Save CSVs ---
        if prov:
            prov.log_act(9, 'save_csv', 'Save similarity and mismatch CSVs')

        mol_sim.to_csv(data_dir / "similarity_molecule.csv")
        result['files']['similarity_molecule'] = str(data_dir / "similarity_molecule.csv")

        morph_sim.to_csv(data_dir / "similarity_morphology.csv")
        result['files']['similarity_morphology'] = str(data_dir / "similarity_morphology.csv")

        proj_sim.to_csv(data_dir / "similarity_projection.csv")
        result['files']['similarity_projection'] = str(data_dir / "similarity_projection.csv")

        mol_morph_mismatch.to_csv(data_dir / "mismatch_mol_morph.csv")
        result['files']['mismatch_mol_morph'] = str(data_dir / "mismatch_mol_morph.csv")

        mol_proj_mismatch.to_csv(data_dir / "mismatch_mol_proj.csv")
        result['files']['mismatch_mol_proj'] = str(data_dir / "mismatch_mol_proj.csv")

        # Top mismatch pairs
        top_mm = _extract_top_mismatch_pairs(mol_morph_mismatch, valid_regions)
        top_mp = _extract_top_mismatch_pairs(mol_proj_mismatch, valid_regions)

        result['summary'] = {
            'n_regions': len(regions),
            'n_subclasses': len(all_subclasses),
            'n_target_subregions': len(all_targets),
            'top_regions_analyzed': len(valid_regions),
            'top_regions': valid_regions,
            'top_mismatch_pairs': {
                'mol_morph': top_mm,
                'mol_proj': top_mp,
            },
        }
        result['success'] = True

        if prov:
            prov.log_reflect(9, 'passed', 0.95, False,
                             [f'Analysis complete: {len(valid_regions)} regions, '
                              f'{len(result["files"])} files'])

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
        if prov:
            prov.log_reflect(0, 'failed', 0.0, False, [f'Error: {e}'])

    return result


# ============================================================
# Report generation
# ============================================================

def generate_fingerprint_report(result: Dict[str, Any], output_dir: str, seed: int,
                            evidence_buffer: Optional[EvidenceBuffer] = None) -> str:
    """Generate the report.md for Fingerprint."""
    output_path = Path(output_dir)

    lines = [
        "# MS Fingerprint Report: Brain Region Fingerprint Analysis",
        "",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        f"**Seed:** {seed}",
        f"**Status:** {'SUCCESS' if result['success'] else 'FAILED'}",
        f"**Neo4j Status:** {'Connected' if result.get('neo4j_status', {}).get('connected') else 'FAILED'}",
        "",
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
            "",
            "### Exception Stack",
            "",
            "```",
            result.get('traceback', 'No traceback available'),
            "```",
        ])
    else:
        summary = result.get('summary', {})
        lines.extend([
            "## Analysis Summary",
            "",
            f"- **Total Regions in KG:** {summary.get('n_regions', 'N/A')}",
            f"- **Top Regions Analyzed:** {summary.get('top_regions_analyzed', 'N/A')}",
            f"- **Subclasses:** {summary.get('n_subclasses', 'N/A')}",
            f"- **Target Subregions:** {summary.get('n_target_subregions', 'N/A')}",
            "",
            "## Fingerprint Types Computed",
            "",
            "1. **Molecular** - Cell type composition (subclass pct_cells)",
            "2. **Morphology** - 8-dim neuronal structure features (z-score normalised)",
            "3. **Projection** - Output connectivity patterns (log10 + L1 normalised)",
            "",
            "## Output Files",
            "",
        ])
        for name, path in result.get('files', {}).items():
            lines.append(f"- `{name}`: {path}")

        lines.extend(["", "## Top Mismatch Pairs", ""])
        top_pairs = summary.get('top_mismatch_pairs', {})

        lines.append("### Molecular-Morphology Mismatch")
        lines.append("")
        for i, (r1, r2, val) in enumerate(top_pairs.get('mol_morph', [])[:10], 1):
            lines.append(f"{i}. {r1} <-> {r2}  (mismatch = {val:.4f})")

        lines.append("")
        lines.append("### Molecular-Projection Mismatch")
        lines.append("")
        for i, (r1, r2, val) in enumerate(top_pairs.get('mol_proj', [])[:10], 1):
            lines.append(f"{i}. {r1} <-> {r2}  (mismatch = {val:.4f})")

    if evidence_buffer:
        lines.append("")
        lines.append(evidence_buffer.to_markdown())

    lines.extend([
        "",
        "---",
        "*Generated by MS Fingerprint runner using self-contained KG extractors*",
    ])

    report_content = "\n".join(lines)
    report_path = output_path / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    return report_content


# ============================================================
# Standalone invocation
# ============================================================

if __name__ == "__main__":
    from provenance import create_provenance_logger

    prov = create_provenance_logger(run_id="fingerprint_analysis", seed=42)
    evidence = EvidenceBuffer()

    result = run_fingerprint_canonical(
        seed=42,
        output_dir="./outputs/fingerprint_analysis",
        provenance_logger=prov,
        evidence_buffer=evidence,
        top_n_regions=30,
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Files generated: {list(result['files'].keys())}")
        report = generate_fingerprint_report(result, "./outputs/fingerprint_analysis", 42, evidence)
        print("Report generated")
        print(f"Evidence summary: KG queries={evidence.get_kg_query_count()}, "
              f"coverage={evidence.get_coverage_rate():.0%}")
    else:
        print(f"Error: {result['error']}")
        if result.get('traceback'):
            print(result['traceback'])
