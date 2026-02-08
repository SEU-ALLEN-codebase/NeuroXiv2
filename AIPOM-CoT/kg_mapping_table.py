from typing import Dict, List, Any

# ============================================================
# Result4 Mapping: Brain Region Fingerprint Analysis
# ============================================================

RESULT4_MAPPING: List[Dict[str, Any]] = [
    {
        "csv_output": "similarity_molecule.csv",
        "description": "Molecular (subclass) similarity matrix between top regions",
        "required_columns": "region (index) x region (columns) -> float similarity [0,1]",
        "neo4j_queries": [
            {
                "purpose": "Get global subclass list (fixed dimensionality)",
                "cypher": (
                    "MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass) "
                    "RETURN DISTINCT sc.name AS subclass_name "
                    "ORDER BY subclass_name"
                ),
                "output": "List[str] - sorted subclass names",
            },
            {
                "purpose": "Compute molecular signature per region",
                "cypher": (
                    "MATCH (r:Region {acronym: $region})"
                    "-[hs:HAS_SUBCLASS]->(sc:Subclass) "
                    "RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells"
                ),
                "output": "np.ndarray(len(all_subclasses)) - pct_cells, zero-filled",
            },
        ],
        "computation": (
            "1. Cosine distance between each pair of molecular signatures\n"
            "2. Min-Max normalize distance matrix to [0,1]\n"
            "3. Similarity = 1 - normalized_distance"
        ),
        "dataframe_schema": "pd.DataFrame, index=regions, columns=regions, dtype=float64",
    },
    {
        "csv_output": "similarity_morphology.csv",
        "description": "Morphology similarity matrix (8-dim z-scored features)",
        "required_columns": "region (index) x region (columns) -> float similarity [0,1]",
        "neo4j_queries": [
            {
                "purpose": "Get 8 morphology features per region",
                "cypher": (
                    "MATCH (r:Region {acronym: $region}) RETURN "
                    "r.axonal_bifurcation_remote_angle, r.axonal_length, "
                    "r.axonal_branches, r.axonal_maximum_branch_order, "
                    "r.dendritic_bifurcation_remote_angle, r.dendritic_length, "
                    "r.dendritic_branches, r.dendritic_maximum_branch_order"
                ),
                "output": "np.ndarray(8) - raw feature values or NaN",
            },
        ],
        "computation": (
            "1. Zero dendritic values (indices 4-7) -> NaN\n"
            "2. Per-feature z-score across all regions (ignoring NaN)\n"
            "3. Euclidean distance between each pair of z-scored vectors\n"
            "4. Min-Max normalize distance matrix to [0,1]\n"
            "5. Similarity = 1 - normalized_distance"
        ),
        "dataframe_schema": "pd.DataFrame, index=regions, columns=regions, dtype=float64",
    },
    {
        "csv_output": "similarity_projection.csv",
        "description": "Projection similarity matrix (log+L1 normalized target weights)",
        "required_columns": "region (index) x region (columns) -> float similarity [0,1]",
        "neo4j_queries": [
            {
                "purpose": "Get global target subregion list",
                "cypher": (
                    "MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion) "
                    "WHERE p.weight IS NOT NULL AND p.weight > 0 "
                    "RETURN DISTINCT t.acronym AS target_subregion "
                    "ORDER BY target_subregion"
                ),
                "output": "List[str] - sorted target subregion acronyms",
            },
            {
                "purpose": "Compute projection signature per region",
                "cypher": (
                    "MATCH (r:Region {acronym: $region}) "
                    "OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT]->(r) "
                    "OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r) "
                    "OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r) "
                    "WITH r, (COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) "
                    "+ COLLECT(DISTINCT n3)) AS ns "
                    "UNWIND ns AS n WITH DISTINCT n WHERE n IS NOT NULL "
                    "MATCH (n)-[p:PROJECT_TO]->(t:Subregion) "
                    "WHERE p.weight IS NOT NULL AND p.weight > 0 "
                    "WITH t.acronym AS tgt_subregion, "
                    "SUM(p.weight) AS total_weight_to_tgt "
                    "RETURN tgt_subregion, total_weight_to_tgt"
                ),
                "output": "np.ndarray(len(all_targets)) - log10(1+x) then L1 normalized",
            },
        ],
        "computation": (
            "1. log10(1 + raw_weight) per target\n"
            "2. L1 normalize -> probability distribution\n"
            "3. Cosine distance between each pair\n"
            "4. Min-Max normalize distance matrix to [0,1]\n"
            "5. Similarity = 1 - normalized_distance"
        ),
        "dataframe_schema": "pd.DataFrame, index=regions, columns=regions, dtype=float64",
    },
    {
        "csv_output": "mismatch_mol_morph.csv",
        "description": "Molecular-Morphology mismatch matrix",
        "required_columns": "region (index) x region (columns) -> float mismatch [0,1]",
        "neo4j_queries": [],  # Derived from similarity matrices
        "computation": (
            "mismatch = |minmax_norm(mol_distance) - minmax_norm(morph_distance)|"
        ),
        "dataframe_schema": "pd.DataFrame, index=regions, columns=regions, dtype=float64",
    },
    {
        "csv_output": "mismatch_mol_proj.csv",
        "description": "Molecular-Projection mismatch matrix",
        "required_columns": "region (index) x region (columns) -> float mismatch [0,1]",
        "neo4j_queries": [],
        "computation": (
            "mismatch = |minmax_norm(mol_distance) - minmax_norm(proj_distance)|"
        ),
        "dataframe_schema": "pd.DataFrame, index=regions, columns=regions, dtype=float64",
    },
]

# Shared query for selecting which regions go into the matrices
RESULT4_TOP_REGIONS_QUERY = {
    "purpose": "Select top N regions by neuron count",
    "cypher": (
        "MATCH (r:Region) "
        "OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r) "
        "WITH r, COUNT(DISTINCT n) AS neuron_count "
        "WHERE neuron_count > 0 "
        "RETURN r.acronym AS region, neuron_count "
        "ORDER BY neuron_count DESC "
        "LIMIT $n"
    ),
    "output": "List[str] - top region acronyms",
}


# ============================================================
# Result6 Mapping: Gene Neuron Profiling (default: Car3)
# ============================================================

RESULT6_MAPPING: List[Dict[str, Any]] = [
    {
        "csv_output": "subclasses.csv",
        "description": "Gene-enriched spatial pockets with dominant subclass info",
        "required_columns": [
            "pocket_eid", "spatial_type", "spatial_name",
            "dominant_subclass_name", "dominant_markers", "dominant_pct",
            "n_subclasses", "granularity_score",
        ],
        "neo4j_queries": [
            {
                "purpose": f"Find all gene-enriched pockets (parameterized by gene)",
                "cypher": (
                    "MATCH (spatial) "
                    "WHERE spatial:Region OR spatial:Subregion OR spatial:ME_Subregion "
                    "MATCH (spatial)-[has_sc:HAS_SUBCLASS]->(sc:Subclass) "
                    "WITH spatial, labels(spatial) AS spatial_type, "
                    "COALESCE(spatial.acronym, spatial.name) AS spatial_name, "
                    "sc, has_sc.pct_cells AS pct_cells, has_sc.rank AS rank "
                    "WITH spatial, spatial_type, spatial_name, "
                    "collect({subclass_name: sc.name, markers: sc.markers, "
                    "pct_cells: pct_cells, rank: rank}) AS subclass_profile "
                    "WITH spatial, spatial_type, spatial_name, subclass_profile, "
                    "[x IN subclass_profile WHERE x.rank = 1][0] AS dominant_subclass, "
                    "size(subclass_profile) AS n_subclasses "
                    "WHERE dominant_subclass IS NOT NULL "
                    "AND (dominant_subclass.markers CONTAINS $gene "
                    "OR dominant_subclass.subclass_name CONTAINS $gene) "
                    "RETURN elementId(spatial) AS pocket_eid, spatial_type, spatial_name, "
                    "dominant_subclass.subclass_name, dominant_subclass.markers, "
                    "dominant_subclass.pct_cells, n_subclasses "
                    "ORDER BY ... LIMIT 50"
                ),
                "output": "One row per enriched pocket",
            },
        ],
        "computation": (
            "granularity_score: ME_Subregion=1, Subregion=2, Region=3"
        ),
        "dataframe_schema": "pd.DataFrame, one row per pocket, sorted by granularity then pct",
    },
    {
        "csv_output": "morphology_summary.csv",
        "description": "Neurons found in gene-enriched pockets",
        "required_columns": [
            "neuron_id", "neuron_name", "celltype", "base_region", "spatial_name",
        ],
        "neo4j_queries": [
            {
                "purpose": "Extract neurons from each pocket (per pocket_eid)",
                "cypher": (
                    "MATCH (spatial) WHERE elementId(spatial) = $POCKET_EID "
                    "OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT_SUBREGION]->(spatial) "
                    "WHERE spatial:Subregion "
                    "OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(spatial) "
                    "WHERE spatial:ME_Subregion "
                    "WITH COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) AS neurons "
                    "UNWIND neurons AS n WITH DISTINCT n WHERE n IS NOT NULL "
                    "RETURN n.neuron_id, n.name, n.celltype, n.base_region"
                ),
                "output": "One row per neuron (run once per pocket)",
            },
        ],
        "computation": "Concatenate across all pockets, add spatial_name column",
        "dataframe_schema": "pd.DataFrame, one row per neuron",
    },
    {
        "csv_output": "projection_targets.csv",
        "description": "Projection targets from neurons in gene-enriched pockets",
        "required_columns": [
            "target_eid", "target_type", "target_name", "target_granularity",
            "n_contributing_neurons", "total_projection_strength",
            "avg_projection_strength", "sample_neurons", "source_spatial",
        ],
        "neo4j_queries": [
            {
                "purpose": "Analyze projections from each pocket (per pocket_eid)",
                "cypher": (
                    "MATCH (spatial) WHERE elementId(spatial) = $POCKET_EID "
                    "... find neurons ... "
                    "MATCH (n)-[proj:PROJECT_TO]->(target) "
                    "WHERE target:Region OR target:Subregion OR target:ME_Subregion "
                    "... aggregate by target ... "
                    "RETURN target_eid, target_type, target_name, "
                    "n_contributing_neurons, total_projection_strength, "
                    "avg_projection_strength "
                    "ORDER BY n_contributing_neurons DESC LIMIT 30"
                ),
                "output": "One row per target per pocket (run once per pocket)",
            },
        ],
        "computation": "Concatenate across pockets, add source_spatial column",
        "dataframe_schema": "pd.DataFrame, one row per (source_pocket, target) pair",
    },
    {
        "csv_output": "region_enrichment.csv",
        "description": "Ranked candidate circuits with multi-component scoring",
        "required_columns": [
            "pocket_eid", "spatial_name", "spatial_type", "total_score",
            "molecular_score", "granularity_score", "neuron_score",
            "projection_score", "gene_score", "n_neurons",
            "n_projection_targets", "dominant_pct", "ranking_reasons",
        ],
        "neo4j_queries": [],  # Derived from pockets, neurons, projections
        "computation": (
            "Canonical scoring (0-100):\n"
            "  molecular_score  = min(dominant_pct, 30)           # 0-30\n"
            "  granularity_score = {ME_Sub:20, Sub:15, Region:10} # 0-20\n"
            "  neuron_score     = {5-1000:20, >1000:15, >0:10}   # 0-20\n"
            "  projection_score = top_3_fraction * 20              # 0-20\n"
            "  gene_score       = 10 (always)                      # 0-10\n"
            "  total = sum of above"
        ),
        "dataframe_schema": "pd.DataFrame, one row per pocket, sorted by total_score DESC",
    },
    {
        "csv_output": "target_composition.csv",
        "description": "Molecular composition (subclass distribution) of top projection targets",
        "required_columns": [
            "target_name", "subclass_name", "markers", "pct_cells", "rank",
        ],
        "neo4j_queries": [
            {
                "purpose": "Get subclass profile of each top target",
                "cypher": (
                    "MATCH (target) WHERE elementId(target) = $TARGET_EID "
                    "MATCH (target)-[has_sc:HAS_SUBCLASS]->(sc:Subclass) "
                    "RETURN COALESCE(target.acronym, target.name) AS target_name, "
                    "sc.name AS subclass_name, sc.markers AS markers, "
                    "has_sc.pct_cells AS pct_cells, has_sc.rank AS rank "
                    "ORDER BY has_sc.rank ASC"
                ),
                "output": "One row per subclass per target (run for top 5 targets)",
            },
        ],
        "computation": "Concatenate across top 5 targets of highest-ranked pocket",
        "dataframe_schema": "pd.DataFrame, one row per (target, subclass) pair",
    },
]


def print_mapping_table():
    """Print a human-readable summary of all mappings."""
    print("=" * 90)
    print("KG MAPPING TABLE: CSV Source -> Neo4j Query -> DataFrame Schema")
    print("=" * 90)

    print("\n--- RESULT4: Brain Region Fingerprint Analysis ---\n")
    for entry in RESULT4_MAPPING:
        print(f"  CSV: {entry['csv_output']}")
        print(f"    Description: {entry['description']}")
        print(f"    Schema: {entry['dataframe_schema']}")
        if entry['neo4j_queries']:
            for q in entry['neo4j_queries']:
                print(f"    Query [{q['purpose']}]:")
                print(f"      {q['cypher'][:120]}...")
        print(f"    Computation: {entry['computation'][:100]}")
        print()

    print(f"  Top Regions Selection: {RESULT4_TOP_REGIONS_QUERY['cypher'][:100]}...\n")

    print("\n--- RESULT6: Gene Neuron Profiling ---\n")
    for entry in RESULT6_MAPPING:
        print(f"  CSV: {entry['csv_output']}")
        print(f"    Description: {entry['description']}")
        cols = entry['required_columns']
        if isinstance(cols, list):
            print(f"    Columns: {', '.join(cols)}")
        else:
            print(f"    Columns: {cols}")
        print(f"    Schema: {entry['dataframe_schema']}")
        if entry['neo4j_queries']:
            for q in entry['neo4j_queries']:
                print(f"    Query [{q['purpose']}]:")
                cypher = q['cypher']
                print(f"      {cypher[:120]}...")
        print(f"    Computation: {entry['computation'][:100]}")
        print()

    print("=" * 90)


if __name__ == "__main__":
    print_mapping_table()
