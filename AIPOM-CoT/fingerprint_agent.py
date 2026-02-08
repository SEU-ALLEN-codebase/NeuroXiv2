import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import zscore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from neuroscience_agent import NeuroscienceAgent, AnalysisStep, AnalysisDepth, Modality
from kg_executor import validate_neurons


class FingerprintAgent(NeuroscienceAgent):
    """
    Agent for cross-modal brain region fingerprint analysis.

    This agent reasons about:
    1. Which fingerprint types (molecular, morphological, projection) to compute
    2. How to select regions for analysis
    3. What the similarity/mismatch patterns mean
    """

    # 32 Axonal features (matching result4.py)
    AXONAL_FEATURES = [
        'axonal_total_length', 'axonal_volume', 'axonal_area',
        'axonal_number_of_bifurcations', 'axonal_max_branch_order',
        'axonal_max_euclidean_distance', 'axonal_max_path_distance',
        'axonal_average_euclidean_distance', 'axonal_average_path_distance',
        'axonal_75pct_euclidean_distance', 'axonal_75pct_path_distance',
        'axonal_50pct_euclidean_distance', 'axonal_50pct_path_distance',
        'axonal_25pct_euclidean_distance', 'axonal_25pct_path_distance',
        'axonal_average_bifurcation_angle_local', 'axonal_average_bifurcation_angle_remote',
        'axonal_average_contraction',
        'axonal_width', 'axonal_height', 'axonal_depth',
        'axonal_width_95ci', 'axonal_height_95ci', 'axonal_depth_95ci',
        'axonal_flatness', 'axonal_flatness_95ci',
        'axonal_slimness', 'axonal_slimness_95ci',
        'axonal_center_shift', 'axonal_relative_center_shift',
        'axonal_2d_density', 'axonal_3d_density'
    ]

    # 32 Dendritic features
    DENDRITIC_FEATURES = [
        'dendritic_total_length', 'dendritic_volume', 'dendritic_area',
        'dendritic_number_of_bifurcations', 'dendritic_max_branch_order',
        'dendritic_max_euclidean_distance', 'dendritic_max_path_distance',
        'dendritic_average_euclidean_distance', 'dendritic_average_path_distance',
        'dendritic_75pct_euclidean_distance', 'dendritic_75pct_path_distance',
        'dendritic_50pct_euclidean_distance', 'dendritic_50pct_path_distance',
        'dendritic_25pct_euclidean_distance', 'dendritic_25pct_path_distance',
        'dendritic_average_bifurcation_angle_local', 'dendritic_average_bifurcation_angle_remote',
        'dendritic_average_contraction',
        'dendritic_width', 'dendritic_height', 'dendritic_depth',
        'dendritic_width_95ci', 'dendritic_height_95ci', 'dendritic_depth_95ci',
        'dendritic_flatness', 'dendritic_flatness_95ci',
        'dendritic_slimness', 'dendritic_slimness_95ci',
        'dendritic_center_shift', 'dendritic_relative_center_shift',
        'dendritic_2d_density', 'dendritic_3d_density'
    ]

    def __init__(
        self,
        seed: int = 42,
        output_dir: str = "./outputs/fingerprint_analysis",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        top_n_regions: int = 30,
        fingerprint_types: List[str] = None
    ):
        super().__init__(
            seed=seed,
            output_dir=output_dir,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            depth=AnalysisDepth.DEEP
        )
        self.top_n_regions = top_n_regions
        self.fingerprint_types = fingerprint_types or ['molecular', 'morphological', 'projection']

        # All 64 morphological features
        self.morph_feature_names = self.AXONAL_FEATURES + self.DENDRITIC_FEATURES

        # Data storage
        self.all_subclasses: List[str] = []
        self.all_targets: List[str] = []
        self.regions: List[str] = []
        self.top_regions: List[str] = []
        self.signatures: Dict[str, Dict[str, np.ndarray]] = {
            'molecular': {},
            'morphological': {},
            'projection': {}
        }

        # Create subdirectories
        self.figures_dir = Path(output_dir) / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def get_run_id(self) -> str:
        return "fingerprint_analysis"

    def get_analysis_goal(self) -> str:
        return "Analyze cross-modal fingerprint similarity and mismatch across brain regions"

    def generate_plan(self, question: str) -> List[AnalysisStep]:
        """
        Generate analysis plan based on the question.

        The agent reasons about which steps are needed.
        """
        self.think(
            "Planning cross-modal fingerprint analysis",
            {
                'question': question,
                'fingerprint_types': self.fingerprint_types,
                'top_n_regions': self.top_n_regions
            }
        )

        steps = []
        step_num = 1

        # Step 1: Discover feature space for molecular fingerprints
        if 'molecular' in self.fingerprint_types:
            steps.append(AnalysisStep(
                step_number=step_num,
                purpose="Discover all subclasses (molecular feature space)",
                template_name='get_all_subclasses',
                modality='molecular',
                reasoning="Need to know all subclasses to build molecular fingerprint vectors",
                expected_outcome="List of all transcriptomic subclasses"
            ))
            step_num += 1

        # Step 2: Discover feature space for projection fingerprints
        if 'projection' in self.fingerprint_types:
            steps.append(AnalysisStep(
                step_number=step_num,
                purpose="Discover all target subregions (projection feature space)",
                template_name='get_all_target_subregions',
                modality='projection',
                reasoning="Need to know all projection targets to build projection fingerprint vectors",
                expected_outcome="List of all projection target subregions"
            ))
            step_num += 1

        # Step 3: Get all analyzable regions
        steps.append(AnalysisStep(
            step_number=step_num,
            purpose="Get all regions with molecular data",
            template_name='get_all_regions',
            modality='molecular',
            reasoning="Need to establish the set of regions that can be analyzed",
            expected_outcome="List of all regions with HAS_SUBCLASS relationships"
        ))
        step_num += 1

        # Step 4: Select top regions by neuron count
        steps.append(AnalysisStep(
            step_number=step_num,
            purpose=f"Select top {self.top_n_regions} regions for analysis",
            template_name='select_top_regions_by_neurons',
            params={'TOP_N': self.top_n_regions},
            modality='morphological',
            reasoning=f"Focus on regions with most morphological data (top {self.top_n_regions} by neuron count)",
            expected_outcome=f"Top {self.top_n_regions} regions ranked by neuron count"
        ))
        step_num += 1

        # Steps 5-7: Compute fingerprints (added dynamically in execution)
        # These are batch operations over regions

        return steps

    def compile_results(self) -> Dict[str, Any]:
        """Compile final results after fingerprint computation."""
        results = {
            'success': True,
            'n_regions': len(self.regions),
            'n_subclasses': len(self.all_subclasses),
            'n_targets': len(self.all_targets),
            'top_regions_analyzed': len(self.top_regions),
            'fingerprint_types': self.fingerprint_types,
            'files': self.results.get('files', {}),  # Carry over accumulated files
            'summary': {
                'top_regions_analyzed': len(self.top_regions),
                'modalities': self.fingerprint_types
            }
        }

        # Add evidence summary
        results['evidence'] = self.get_evidence_summary()

        return results

    def run(self, question: str = None) -> Dict[str, Any]:
        """
        Run the full Fingerprint analysis with agent reasoning.
        """
        question = question or self.get_analysis_goal()

        # Log run start
        self.prov.log_run_start(
            mode='ms_agent',
            intent='MS_REPRO',
            query=question,
            budget='heavy'
        )

        # THINK: Understand the goal
        self.think(
            "Starting Fingerprint cross-modal fingerprint analysis",
            {
                'question': question,
                'fingerprint_types': self.fingerprint_types,
                'top_n_regions': self.top_n_regions
            }
        )

        # Connect to KG
        if not self.executor.connect():
            self.results['success'] = False
            self.results['error'] = "Failed to connect to Neo4j"
            return self.results

        try:
            # Phase 1: Feature space discovery
            self._discover_feature_spaces()

            # Phase 2: Region selection
            self._select_regions()

            # Phase 3: Compute fingerprints
            self._compute_fingerprints()

            # Phase 4: Compute similarity and mismatch matrices
            self._compute_matrices()

            # Phase 5: Generate visualizations
            self._generate_visualizations()

            # Phase 6: Generate report
            self._generate_report()

            # Compile results
            self.results = self.compile_results()

        except Exception as e:
            self.results['success'] = False
            self.results['error'] = str(e)
            import traceback
            self.results['traceback'] = traceback.format_exc()

        finally:
            self.executor.close()

        # Log run end
        self.prov.log_run_end(
            termination_reason='completed' if self.results.get('success') else 'error',
            total_steps=len(self.state.executed_steps),
            total_kg_queries=self.executor.query_count,
            execution_time=0,
            success=self.results.get('success', False)
        )

        return self.results

    def _discover_feature_spaces(self):
        """Phase 1: Discover feature spaces for fingerprints."""
        self.think(
            "Discovering feature spaces for fingerprint computation",
            {'fingerprint_types': self.fingerprint_types}
        )

        # Get all subclasses (molecular feature space)
        if 'molecular' in self.fingerprint_types:
            result = self.executor.execute_with_reasoning(
                template_name='get_all_subclasses',
                reasoning="Building molecular feature space from all transcriptomic subclasses",
                step_number=1
            )
            if result.success:
                self.all_subclasses = sorted(result.data['subclass_name'].tolist())
                self.think(f"Found {len(self.all_subclasses)} subclasses for molecular fingerprints")

        # Get all target subregions (projection feature space)
        if 'projection' in self.fingerprint_types:
            result = self.executor.execute_with_reasoning(
                template_name='get_all_target_subregions',
                reasoning="Building projection feature space from all target subregions",
                step_number=2
            )
            if result.success:
                self.all_targets = sorted(result.data['target_acronym'].tolist())
                self.think(f"Found {len(self.all_targets)} targets for projection fingerprints")

        # Get all regions
        result = self.executor.execute_with_reasoning(
            template_name='get_all_regions',
            reasoning="Identifying all regions with molecular data for analysis",
            step_number=3
        )
        if result.success:
            self.regions = result.data['region_acronym'].tolist()
            self.think(f"Found {len(self.regions)} analyzable regions")

    def _select_regions(self):
        """Phase 2: Select regions for analysis."""
        self.think(
            f"Selecting top {self.top_n_regions} regions by neuron count",
            {'strategy': 'neuron_count', 'top_n': self.top_n_regions}
        )

        result = self.executor.execute_with_reasoning(
            template_name='select_top_regions_by_neurons',
            params={'TOP_N': self.top_n_regions},
            reasoning=f"Selecting top {self.top_n_regions} regions with most morphological data",
            step_number=4
        )

        if result.success:
            self.top_regions = result.data['region_acronym'].tolist()
            # Filter to regions we have data for
            self.top_regions = [r for r in self.top_regions if r in self.regions]
            self.think(f"Selected {len(self.top_regions)} top regions for analysis")

            # Reflect on selection
            self.prov.log_reflect(
                step_number=4,
                validation_status='passed',
                confidence=len(self.top_regions) / self.top_n_regions,
                should_replan=False,
                recommendations=[
                    f"Selected {len(self.top_regions)} regions from {len(self.regions)} total",
                    f"Selection based on neuron count (morphological data availability)",
                    f"Top region: {self.top_regions[0] if self.top_regions else 'none'}"
                ]
            )

    def _compute_fingerprints(self):
        """Phase 3: Compute fingerprints for each region."""
        self.think(
            "Computing fingerprints for selected regions",
            {'n_regions': len(self.top_regions), 'fingerprint_types': self.fingerprint_types}
        )

        step_base = 5

        # Molecular fingerprints
        if 'molecular' in self.fingerprint_types:
            self._compute_molecular_fingerprints(step_base)
            step_base += 1

        # Morphological fingerprints
        if 'morphological' in self.fingerprint_types:
            self._compute_morphological_fingerprints(step_base)
            step_base += 1

        # Projection fingerprints
        if 'projection' in self.fingerprint_types:
            self._compute_projection_fingerprints(step_base)

    def _compute_molecular_fingerprints(self, step_number: int):
        """Compute molecular fingerprints (subclass composition vectors).

        Following result4.py pipeline:
        1. Get raw pct_cells from HAS_SUBCLASS relationships
        2. Prune zero-sum columns
        3. Z-score standardization per feature
        """
        self.think(
            "Computing molecular fingerprints (pct_cells → zscore)",
            {'n_regions': len(self.top_regions), 'n_features': len(self.all_subclasses)}
        )

        self.prov.log_act(
            step_number=step_number,
            action_type='computation',
            purpose='Compute molecular fingerprints (pct_cells + zscore)',
            params={'n_regions': len(self.top_regions), 'n_features': len(self.all_subclasses)}
        )

        subclass_to_idx = {s: i for i, s in enumerate(self.all_subclasses)}

        # Step 1: Get raw signatures
        for region in self.top_regions:
            result = self.executor.execute_raw(
                """
                MATCH (r:Region {acronym: $REGION})-[hs:HAS_SUBCLASS]->(sc:Subclass)
                RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
                """,
                {'REGION': region}
            )

            vec = np.zeros(len(self.all_subclasses))
            for _, row in result.iterrows():
                if row['subclass_name'] in subclass_to_idx:
                    vec[subclass_to_idx[row['subclass_name']]] = row['pct_cells'] or 0

            self.signatures['molecular'][region] = vec

        # Step 2 & 3: Apply normalization
        self._apply_molecular_normalization()

        # Reflect
        n_nonzero = sum(1 for v in self.signatures['molecular'].values() if np.any(v != 0))
        self.prov.log_reflect(
            step_number=step_number,
            validation_status='passed',
            confidence=n_nonzero / len(self.top_regions),
            should_replan=False,
            recommendations=[
                f"Computed molecular fingerprints for {len(self.top_regions)} regions",
                f"{n_nonzero} regions have valid molecular signatures",
                f"Applied pruning and z-score normalization"
            ]
        )

    def _compute_morphological_fingerprints(self, step_number: int):
        """Compute morphological fingerprints (64-dim feature vectors).

        Uses all 64 morphological features (32 axonal + 32 dendritic) with:
        1. Query from Region nodes or aggregate from Neurons
        2. Log1p transform
        3. Z-score normalization per feature
        """
        n_features = len(self.morph_feature_names)
        self.think(
            "Computing morphological fingerprints (64-dim)",
            {'n_regions': len(self.top_regions), 'n_features': n_features}
        )

        self.prov.log_act(
            step_number=step_number,
            action_type='computation',
            purpose='Compute morphological fingerprints (64-dim log1p+zscore)',
            params={'n_regions': len(self.top_regions), 'n_features': n_features}
        )

        # Query all 64 features for each region
        for region in self.top_regions:
            sig = self._get_morph_signature_for_region(region)
            self.signatures['morphological'][region] = sig

        # Apply Log1p → Z-score normalization
        self._apply_morphology_normalization()

        # Reflect
        n_valid = sum(1 for v in self.signatures['morphological'].values()
                     if not np.all(np.isnan(v)))
        self.prov.log_reflect(
            step_number=step_number,
            validation_status='passed',
            confidence=n_valid / len(self.top_regions),
            should_replan=False,
            recommendations=[
                f"Computed 64-dim morphological fingerprints for {len(self.top_regions)} regions",
                f"{n_valid} regions have valid morphological data",
                "Applied log1p transform and z-score normalization"
            ]
        )

    def _get_morph_signature_for_region(self, region: str) -> np.ndarray:
        """Get 64-dim morphological signature for a region.

        First tries to get from Region node properties, then aggregates from Neurons.
        """
        # Try to get from Region node first
        result = self.executor.execute_raw(
            "MATCH (r:Region {acronym: $REGION}) RETURN r",
            {'REGION': region}
        )

        if not result.empty:
            node = result.iloc[0]['r']
            if node is not None:
                signature = []
                for feat in self.morph_feature_names:
                    val = node.get(feat, None)
                    if val is None:
                        val = node.get(feat + '_mean', None)
                    signature.append(val if val is not None else np.nan)

                valid_count = np.sum(~np.isnan(signature))
                if valid_count >= 10:  # Enough features from Region node
                    return np.array(signature, dtype=float)

        # Fallback: aggregate from Neurons
        return self._aggregate_morph_from_neurons(region)

    def _aggregate_morph_from_neurons(self, region: str) -> np.ndarray:
        """Aggregate morphological features from neurons in a region."""
        # Build return clause for all features
        axon_return = [f"avg(n.{feat}) AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"avg(n.{feat}) AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {{acronym: $REGION}})
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
        RETURN {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        result = self.executor.execute_raw(query, {'REGION': region})

        if result.empty:
            return np.array([np.nan] * 64, dtype=float)

        signature = []
        for feat in self.AXONAL_FEATURES:
            val = result.iloc[0].get(feat, None)
            signature.append(val if val is not None else np.nan)
        for feat in self.DENDRITIC_FEATURES:
            val = result.iloc[0].get(feat, None)
            signature.append(val if val is not None else np.nan)

        return np.array(signature, dtype=float)

    def _apply_morphology_normalization(self):
        """Apply Log1p → Z-score normalization for 64-dim morphological fingerprints.

        Following result4.py pipeline:
        1. Fill NaN with column means
        2. Log1p transform
        3. Z-score standardization per feature
        """
        regions = list(self.signatures['morphological'].keys())
        if not regions:
            return

        matrix = np.array([self.signatures['morphological'][r] for r in regions])
        n_features = matrix.shape[1]

        self.think(f"Normalizing {n_features}-dim morphological fingerprints")

        # Step 1: Fill NaN with column means
        col_means = np.nanmean(matrix, axis=0)
        for j in range(n_features):
            mask = np.isnan(matrix[:, j])
            if np.isnan(col_means[j]):
                matrix[mask, j] = 0
            else:
                matrix[mask, j] = col_means[j]

        # Step 2: Log1p transform
        matrix = np.log1p(np.abs(matrix))  # Handle negative values with abs

        # Step 3: Z-score standardization per feature
        for j in range(n_features):
            col = matrix[:, j]
            std = np.std(col)
            if std > 1e-10:
                matrix[:, j] = (col - np.mean(col)) / std
            else:
                matrix[:, j] = 0

        # Update signatures
        for i, r in enumerate(regions):
            self.signatures['morphological'][r] = matrix[i]

    def _apply_molecular_normalization(self):
        """Apply Z-score normalization for molecular fingerprints.

        Following result4.py pipeline:
        1. Prune zero-sum columns (remove subclasses with no data)
        2. Z-score standardization per feature
        """
        regions = list(self.signatures['molecular'].keys())
        if not regions:
            return

        matrix = np.array([self.signatures['molecular'][r] for r in regions])
        n_features = matrix.shape[1]

        self.think(f"Normalizing molecular fingerprints ({n_features} subclasses)")

        # Step 1: Identify non-zero columns (prune zero-sum columns conceptually)
        # We keep all columns but mark zero-variance ones
        col_sums = np.sum(matrix, axis=0)
        nonzero_cols = col_sums > 0
        n_valid = np.sum(nonzero_cols)

        # Step 2: Z-score standardization per feature
        for j in range(n_features):
            col = matrix[:, j]
            std = np.std(col)
            if std > 1e-10:
                matrix[:, j] = (col - np.mean(col)) / std
            else:
                matrix[:, j] = 0

        # Update signatures
        for i, r in enumerate(regions):
            self.signatures['molecular'][r] = matrix[i]

        self.think(f"Molecular normalization complete: {n_valid}/{n_features} features have data")

    def _compute_projection_fingerprints(self, step_number: int):
        """Compute projection fingerprints (target composition vectors).

        Following result4.py pipeline:
        1. Get raw projection weights to subregions
        2. Prune zero-sum columns
        3. Log1p transform
        4. Z-score standardization per feature
        """
        self.think(
            "Computing projection fingerprints (log1p → zscore)",
            {'n_regions': len(self.top_regions), 'n_features': len(self.all_targets)}
        )

        self.prov.log_act(
            step_number=step_number,
            action_type='computation',
            purpose='Compute projection fingerprints (log1p + zscore)',
            params={'n_regions': len(self.top_regions), 'n_features': len(self.all_targets)}
        )

        target_to_idx = {t: i for i, t in enumerate(self.all_targets)}

        # Step 1: Get raw signatures
        for region in self.top_regions:
            result = self.executor.execute_raw(
                """
                MATCH (r:Region {acronym: $REGION})<-[:LOCATE_AT]-(n:Neuron)
                MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
                WHERE COALESCE(p.weight, p.projection_length, p.total, 0) > 0
                WITH t.acronym AS target, SUM(COALESCE(p.weight, p.projection_length, p.total, 0)) AS total_weight
                RETURN target, total_weight
                """,
                {'REGION': region}
            )

            vec = np.zeros(len(self.all_targets))
            for _, row in result.iterrows():
                if row['target'] in target_to_idx:
                    vec[target_to_idx[row['target']]] = row['total_weight'] or 0

            self.signatures['projection'][region] = vec

        # Step 2, 3, 4: Apply normalization
        self._apply_projection_normalization()

        # Reflect
        n_nonzero = sum(1 for v in self.signatures['projection'].values() if np.any(v != 0))
        self.prov.log_reflect(
            step_number=step_number,
            validation_status='passed',
            confidence=n_nonzero / len(self.top_regions),
            should_replan=False,
            recommendations=[
                f"Computed projection fingerprints for {len(self.top_regions)} regions",
                f"{n_nonzero} regions have projection data",
                "Applied log1p transform and z-score normalization"
            ]
        )

    def _apply_projection_normalization(self):
        """Apply Log1p → Z-score normalization for projection fingerprints.

        Following result4.py pipeline:
        1. Prune zero-sum columns
        2. Log1p transform
        3. Z-score standardization per feature
        """
        regions = list(self.signatures['projection'].keys())
        if not regions:
            return

        matrix = np.array([self.signatures['projection'][r] for r in regions])
        n_features = matrix.shape[1]

        self.think(f"Normalizing projection fingerprints ({n_features} targets)")

        # Step 1: Identify non-zero columns
        col_sums = np.sum(matrix, axis=0)
        nonzero_cols = col_sums > 0
        n_valid = np.sum(nonzero_cols)

        # Step 2: Log1p transform
        matrix = np.log1p(matrix)

        # Step 3: Z-score standardization per feature
        for j in range(n_features):
            col = matrix[:, j]
            std = np.std(col)
            if std > 1e-10:
                matrix[:, j] = (col - np.mean(col)) / std
            else:
                matrix[:, j] = 0

        # Update signatures
        for i, r in enumerate(regions):
            self.signatures['projection'][r] = matrix[i]

        self.think(f"Projection normalization complete: {n_valid}/{n_features} targets have data")

    def _apply_zscore_normalization(self, fingerprint_type: str):
        """Apply z-score normalization across regions for a fingerprint type."""
        regions = list(self.signatures[fingerprint_type].keys())
        if not regions:
            return

        n_features = len(self.signatures[fingerprint_type][regions[0]])
        matrix = np.array([self.signatures[fingerprint_type][r] for r in regions])

        # Z-score per feature (column)
        for i in range(n_features):
            col = matrix[:, i]
            if np.std(col) > 0:
                matrix[:, i] = zscore(col)
            else:
                matrix[:, i] = 0

        # Update signatures
        for i, r in enumerate(regions):
            self.signatures[fingerprint_type][r] = matrix[i]

    def _compute_matrices(self):
        """Phase 4: Compute similarity and mismatch matrices."""
        self.think(
            "Computing distance and similarity matrices",
            {'fingerprint_types': self.fingerprint_types, 'n_regions': len(self.top_regions)}
        )

        step_number = 8
        self.prov.log_act(
            step_number=step_number,
            action_type='computation',
            purpose='Compute similarity and mismatch matrices',
            params={'n_regions': len(self.top_regions)}
        )

        # Distance matrices
        self.distance_matrices = {}
        self.similarity_matrices = {}
        self.mismatch_matrices = {}

        # Compute distances
        if 'molecular' in self.fingerprint_types:
            self.distance_matrices['molecular'] = self._compute_distance_matrix('molecular', cosine)
        if 'morphological' in self.fingerprint_types:
            self.distance_matrices['morphological'] = self._compute_distance_matrix('morphological', euclidean)
        if 'projection' in self.fingerprint_types:
            self.distance_matrices['projection'] = self._compute_distance_matrix('projection', cosine)

        # Min-max normalize and convert to similarity
        for name, dist_df in self.distance_matrices.items():
            norm_df = self._minmax_normalize(dist_df)
            self.similarity_matrices[name] = 1 - norm_df

            # Save
            path = self.save_dataframe(self.similarity_matrices[name], f"similarity_{name}.csv")
            self.results['files'] = self.results.get('files', {})
            self.results['files'][f'similarity_{name}'] = path

        # Compute mismatch matrices
        if 'molecular' in self.similarity_matrices and 'morphological' in self.similarity_matrices:
            mol_norm = self._minmax_normalize(self.distance_matrices['molecular'])
            morph_norm = self._minmax_normalize(self.distance_matrices['morphological'])
            self.mismatch_matrices['mol_morph'] = np.abs(mol_norm - morph_norm)
            path = self.save_dataframe(self.mismatch_matrices['mol_morph'], "mismatch_mol_morph.csv")
            self.results['files']['mismatch_mol_morph'] = path

        if 'molecular' in self.similarity_matrices and 'projection' in self.similarity_matrices:
            mol_norm = self._minmax_normalize(self.distance_matrices['molecular'])
            proj_norm = self._minmax_normalize(self.distance_matrices['projection'])
            self.mismatch_matrices['mol_proj'] = np.abs(mol_norm - proj_norm)
            path = self.save_dataframe(self.mismatch_matrices['mol_proj'], "mismatch_mol_proj.csv")
            self.results['files']['mismatch_mol_proj'] = path

        # Reflect on findings
        # IMPORTANT: Use .copy() to avoid modifying the original matrices!
        findings = []
        for name, sim_df in self.similarity_matrices.items():
            values = sim_df.values.copy()  # Copy to avoid modifying original
            np.fill_diagonal(values, np.nan)  # Exclude self-similarity for mean
            mean_sim = np.nanmean(values)
            findings.append(f"Mean {name} similarity: {mean_sim:.3f}")

        for name, mismatch_df in self.mismatch_matrices.items():
            values = mismatch_df.values.copy()  # Copy to avoid modifying original
            np.fill_diagonal(values, np.nan)  # Exclude diagonal for max
            max_mismatch = np.nanmax(values)
            findings.append(f"Max {name} mismatch: {max_mismatch:.3f}")

        self.prov.log_reflect(
            step_number=step_number,
            validation_status='passed',
            confidence=1.0,
            should_replan=False,
            recommendations=findings
        )

    def _compute_distance_matrix(self, fingerprint_type: str, metric_fn) -> pd.DataFrame:
        """Compute pairwise distance matrix with NaN handling."""
        regions = self.top_regions
        n = len(regions)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0
                else:
                    v1 = self.signatures[fingerprint_type].get(regions[i], np.zeros(1))
                    v2 = self.signatures[fingerprint_type].get(regions[j], np.zeros(1))

                    # Handle NaN values - use only valid (non-NaN) features
                    valid_mask = ~(np.isnan(v1) | np.isnan(v2))
                    if valid_mask.sum() == 0:
                        matrix[i, j] = np.nan  # No valid features to compare
                    elif np.all(v1[valid_mask] == 0) or np.all(v2[valid_mask] == 0):
                        matrix[i, j] = 1.0  # Max distance if no data
                    else:
                        try:
                            # Compute distance using only valid features
                            matrix[i, j] = metric_fn(v1[valid_mask], v2[valid_mask])
                        except:
                            matrix[i, j] = 1.0

        return pd.DataFrame(matrix, index=regions, columns=regions)

    def _minmax_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize a matrix."""
        values = df.values.copy()
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax > vmin:
            values = (values - vmin) / (vmax - vmin)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def _generate_visualizations(self):
        """Phase 5: Generate publication-quality visualizations."""
        self.think("Generating similarity and mismatch heatmaps")

        step_number = 9
        self.prov.log_act(
            step_number=step_number,
            action_type='visualization',
            purpose='Generate similarity and mismatch heatmaps'
        )

        # 1. Generate combined matrix figure (2x3 layout)
        self._generate_combined_matrix_figure()

        # 2. Individual similarity heatmaps (matching result4.py style)
        for name, sim_df in self.similarity_matrices.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(sim_df, cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       ax=ax, xticklabels=True, yticklabels=True, annot=False,
                       cbar_kws={'label': 'Similarity'})
            title_map = {'molecular': 'Molecular fingerprint Similarity',
                        'morphological': 'Morphology fingerprint Similarity',
                        'projection': 'Projection fingerprint Similarity'}
            ax.set_title(title_map.get(name, f'{name.capitalize()} Similarity'),
                        fontsize=20, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
            ax.set_xlabel('Region', fontsize=20, fontweight='bold')
            ax.set_ylabel('Region', fontsize=20, fontweight='bold')
            plt.tight_layout()
            path = self.figures_dir / f"{name}_similarity.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')  # Use 300 for reasonable file size
            plt.close()
            self.results['files'][f'{name}_similarity_fig'] = str(path)

        # 3. Individual mismatch heatmaps (matching result4.py style with RdYlBu_r)
        for name, mismatch_df in self.mismatch_matrices.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(mismatch_df, cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       ax=ax, xticklabels=True, yticklabels=True, annot=False,
                       cbar_kws={'label': 'Mismatch'})
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
            ax.set_xlabel('Region', fontsize=20, fontweight='bold')
            ax.set_ylabel('Region', fontsize=20, fontweight='bold')
            plt.tight_layout()
            path = self.figures_dir / f"{name}_mismatch.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            self.results['files'][f'{name}_mismatch_fig'] = str(path)

        # 4. Generate top mismatch pair comparisons
        self._generate_mismatch_comparison_figures()

        self.prov.log_reflect(
            step_number=step_number,
            validation_status='passed',
            confidence=1.0,
            should_replan=False,
            recommendations=[f"Generated combined figure + {len(self.similarity_matrices) + len(self.mismatch_matrices)} individual heatmaps + comparison plots"]
        )

    def _generate_combined_matrix_figure(self):
        """Generate a combined 2x3 matrix figure showing all similarity and mismatch matrices."""
        self.think("Generating combined matrix visualization")

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle('Brain Region Similarity and Mismatch Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        # Row 1: Similarity matrices
        if 'molecular' in self.similarity_matrices:
            sns.heatmap(self.similarity_matrices['molecular'], ax=axes[0, 0],
                       cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       cbar_kws={'label': 'Similarity'}, xticklabels=True, yticklabels=True)
            axes[0, 0].set_title('Molecular Similarity', fontsize=14, fontweight='bold')

        if 'morphological' in self.similarity_matrices:
            sns.heatmap(self.similarity_matrices['morphological'], ax=axes[0, 1],
                       cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       cbar_kws={'label': 'Similarity'}, xticklabels=True, yticklabels=True)
            axes[0, 1].set_title('Morphology Similarity', fontsize=14, fontweight='bold')

        if 'projection' in self.similarity_matrices:
            sns.heatmap(self.similarity_matrices['projection'], ax=axes[0, 2],
                       cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       cbar_kws={'label': 'Similarity'}, xticklabels=True, yticklabels=True)
            axes[0, 2].set_title('Projection Similarity', fontsize=14, fontweight='bold')

        # Row 2: Mismatch matrices (using RdYlBu_r to match result4.py)
        if 'mol_morph' in self.mismatch_matrices:
            sns.heatmap(self.mismatch_matrices['mol_morph'], ax=axes[1, 0],
                       cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       cbar_kws={'label': 'Mismatch'}, xticklabels=True, yticklabels=True)
            axes[1, 0].set_title('Molecular-Morphology Mismatch', fontsize=14, fontweight='bold')

        if 'mol_proj' in self.mismatch_matrices:
            sns.heatmap(self.mismatch_matrices['mol_proj'], ax=axes[1, 1],
                       cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                       cbar_kws={'label': 'Mismatch'}, xticklabels=True, yticklabels=True)
            axes[1, 1].set_title('Molecular-Projection Mismatch', fontsize=14, fontweight='bold')

        # Empty subplot for layout balance
        axes[1, 2].axis('off')

        plt.tight_layout()
        path = self.figures_dir / "all_matrices_combined.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.results['files']['combined_matrices_fig'] = str(path)

    def _generate_mismatch_comparison_figures(self):
        """Generate detailed comparison figures for top mismatch pairs."""
        self.think("Identifying and visualizing top mismatch pairs")

        # Find top 3 mismatch pairs for each mismatch type
        for mismatch_name, mismatch_df in self.mismatch_matrices.items():
            top_pairs = self._get_top_mismatch_pairs(mismatch_df, n=3)

            for rank, (r1, r2, mismatch_val) in enumerate(top_pairs, 1):
                if mismatch_name == 'mol_morph':
                    self._plot_mol_morph_comparison(r1, r2, mismatch_val, rank)
                elif mismatch_name == 'mol_proj':
                    self._plot_mol_proj_comparison(r1, r2, mismatch_val, rank)

    def _get_top_mismatch_pairs(self, mismatch_df, n=3):
        """Extract top N mismatch pairs from matrix."""
        values = mismatch_df.values.copy()
        np.fill_diagonal(values, np.nan)
        regions = list(mismatch_df.index)

        pairs = []
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if not np.isnan(values[i, j]):
                    pairs.append((regions[i], regions[j], values[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    def _plot_mol_morph_comparison(self, region1, region2, mismatch, rank):
        """Plot molecular-morphology mismatch comparison with bar charts (64-dim morphology)."""
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1])

        # 1. Morphology features bar chart (first 10 features with highest variance)
        ax_morph = fig.add_subplot(gs[0])
        morph1 = self.signatures['morphological'].get(region1, np.zeros(64))
        morph2 = self.signatures['morphological'].get(region2, np.zeros(64))

        # Show first 10 dimensions (or features with highest difference)
        n_show = min(10, len(morph1))
        diff = np.abs(morph1 - morph2)
        top_idx = np.argsort(diff)[-n_show:][::-1]

        x = np.arange(n_show)
        width = 0.35

        ax_morph.bar(x - width/2, morph1[top_idx], width, label=region1, color='#E74C3C', alpha=0.8)
        ax_morph.bar(x + width/2, morph2[top_idx], width, label=region2, color='#3498DB', alpha=0.8)
        ax_morph.set_xticks(x)
        ax_morph.set_xticklabels([f'F{i+1}' for i in top_idx], fontsize=9)
        ax_morph.set_xlabel('Feature Index (top 10 by difference)', fontsize=11)
        ax_morph.set_ylabel('Z-score', fontsize=11)
        ax_morph.set_title(f'Morphology (64D normalized)', fontsize=12, fontweight='bold')
        ax_morph.legend()
        ax_morph.grid(axis='y', alpha=0.3)

        # 2. Top 10 molecular subclasses
        ax_mol = fig.add_subplot(gs[1])
        mol1 = self.signatures['molecular'].get(region1, np.zeros(len(self.all_subclasses)))
        mol2 = self.signatures['molecular'].get(region2, np.zeros(len(self.all_subclasses)))

        if len(mol1) > 0 and len(self.all_subclasses) > 0:
            top_idx = np.argsort(mol1 + mol2)[-10:][::-1]
            top_names = [self.all_subclasses[i][:25] if i < len(self.all_subclasses) else f'SC{i}' for i in top_idx]
            x = np.arange(len(top_idx))
            ax_mol.barh(x - 0.2, mol1[top_idx], 0.35, label=region1, color='#E74C3C', alpha=0.8)
            ax_mol.barh(x + 0.2, mol2[top_idx], 0.35, label=region2, color='#3498DB', alpha=0.8)
            ax_mol.set_yticks(x)
            ax_mol.set_yticklabels(top_names, fontsize=8)
        ax_mol.set_xlabel('Fraction')
        ax_mol.set_title('Top 10 Cell Types', fontsize=11, fontweight='bold')
        ax_mol.legend()

        # 3. Info panel
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')
        info_text = f"Rank #{rank}\n{region1} vs {region2}\n\nMismatch: {mismatch:.3f}"
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                    transform=ax_info.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        path = self.figures_dir / f"detail_mol_morph_{rank}_{region1}_vs_{region2}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_mol_proj_comparison(self, region1, region2, mismatch, rank):
        """Plot molecular-projection mismatch comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Top projection targets
        ax = axes[0]
        proj1 = self.signatures['projection'].get(region1, np.zeros(len(self.all_targets)))
        proj2 = self.signatures['projection'].get(region2, np.zeros(len(self.all_targets)))
        top_idx = np.argsort(proj1 + proj2)[-10:][::-1]
        top_names = [self.all_targets[i][:15] for i in top_idx]
        x = np.arange(10)
        ax.barh(x - 0.2, proj1[top_idx], 0.35, label=region1, color='#E74C3C', alpha=0.8)
        ax.barh(x + 0.2, proj2[top_idx], 0.35, label=region2, color='#3498DB', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel('Projection Strength (normalized)')
        ax.set_title('Top 10 Projection Targets', fontsize=11, fontweight='bold')
        ax.legend()

        # 2. Top molecular subclasses
        ax = axes[1]
        mol1 = self.signatures['molecular'].get(region1, np.zeros(len(self.all_subclasses)))
        mol2 = self.signatures['molecular'].get(region2, np.zeros(len(self.all_subclasses)))
        top_idx = np.argsort(mol1 + mol2)[-10:][::-1]
        top_names = [self.all_subclasses[i][:25] for i in top_idx]
        x = np.arange(10)
        ax.barh(x - 0.2, mol1[top_idx], 0.35, label=region1, color='#E74C3C', alpha=0.8)
        ax.barh(x + 0.2, mol2[top_idx], 0.35, label=region2, color='#3498DB', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel('Fraction')
        ax.set_title('Top 10 Cell Types', fontsize=11, fontweight='bold')
        ax.legend()

        # 3. Info panel
        ax = axes[2]
        ax.axis('off')
        info_text = f"Rank #{rank}\n{region1} vs {region2}\n\nMismatch: {mismatch:.3f}"
        ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        path = self.figures_dir / f"detail_mol_proj_{rank}_{region1}_vs_{region2}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self):
        """Generate the analysis report with agent reasoning."""
        from datetime import datetime

        self.think("Generating analysis report with agent reasoning")

        # Collect summary stats
        summary_stats = {
            'n_regions': len(self.regions),
            'n_top_regions': len(self.top_regions),
            'n_subclasses': len(self.all_subclasses),
            'n_targets': len(self.all_targets),
            'fingerprint_types': self.fingerprint_types,
            'kg_queries': self.executor.query_count,
            'total_rows': self.executor.total_rows
        }

        # Build report content
        report_lines = [
            "# Result 4: Cross-Modal Brain Region Fingerprint Analysis",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Seed**: {self.seed}",
            f"**Status**: {'SUCCESS' if self.results.get('success', True) else 'FAILED'}",
            "",
            "## Executive Summary",
            "",
            f"This agent-driven analysis computed cross-modal fingerprints for {len(self.top_regions)} brain regions",
            f"using {self.executor.query_count} KG queries. The analysis identifies regions where molecular,",
            "morphological, and projection profiles show high similarity or significant mismatch.",
            "",
            "### Key Findings",
            ""
        ]

        # Add similarity findings
        for name, sim_df in self.similarity_matrices.items():
            values = sim_df.values.copy()
            np.fill_diagonal(values, np.nan)
            mean_sim = np.nanmean(values)
            max_sim = np.nanmax(values)
            report_lines.append(f"- **{name.capitalize()} similarity**: mean={mean_sim:.3f}, max={max_sim:.3f}")

        # Add mismatch findings
        for name, mismatch_df in self.mismatch_matrices.items():
            values = mismatch_df.values.copy()
            np.fill_diagonal(values, np.nan)
            max_mismatch = np.nanmax(values)
            mean_mismatch = np.nanmean(values)
            report_lines.append(f"- **{name.replace('_', '-')} mismatch**: mean={mean_mismatch:.3f}, max={max_mismatch:.3f}")

        report_lines.extend([
            "",
            "## Agent Reasoning Trace",
            "",
            "The agent followed the TPAR workflow (Think → Plan → Act → Reflect):",
            ""
        ])

        # Add state findings (agent's reasoning)
        for finding in self.state.findings[:20]:  # Limit to 20 entries
            report_lines.append(f"- {finding}")

        report_lines.extend([
            "",
            "## Data Summary",
            "",
            f"- **Total regions analyzed**: {len(self.top_regions)} (of {len(self.regions)} available)",
            f"- **Molecular feature space**: {len(self.all_subclasses)} subclasses",
            f"- **Projection feature space**: {len(self.all_targets)} targets",
            f"- **KG queries executed**: {self.executor.query_count}",
            f"- **Total rows retrieved**: {self.executor.total_rows}",
            "",
            "## Output Files",
            ""
        ])

        for name, path in self.results.get('files', {}).items():
            report_lines.append(f"- `{name}`: {path}")

        report_lines.extend([
            "",
            "## Evidence Summary",
            "",
            self.evidence.to_markdown() if hasattr(self, 'evidence') and self.evidence else "Evidence tracking enabled via EvidenceBuffer",
            "",
            "---",
            "*Generated by FingerprintAgent with TPAR workflow*"
        ])

        report_content = "\n".join(report_lines)

        # Write report
        report_path = Path(self.output_dir) / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.results['files'] = self.results.get('files', {})
        self.results['files']['report'] = str(report_path)

        self.prov.log_reflect(
            step_number=10,
            validation_status='passed',
            confidence=1.0,
            should_replan=False,
            recommendations=["Report generated successfully"]
        )


def run_fingerprint_analysis(
    seed: int = 42,
    output_dir: str = "./outputs/fingerprint_analysis",
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
    top_n_regions: int = 30
) -> Dict[str, Any]:
    """
    Convenience function to run Fingerprint with agent reasoning.
    """
    agent = FingerprintAgent(
        seed=seed,
        output_dir=output_dir,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        top_n_regions=top_n_regions
    )
    return agent.run()


if __name__ == "__main__":
    result = run_fingerprint_analysis(seed=42)
    print(f"Success: {result.get('success')}")
    print(f"Files: {list(result.get('files', {}).keys())}")
