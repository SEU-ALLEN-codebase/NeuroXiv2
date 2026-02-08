import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from neuroscience_agent import NeuroscienceAgent, AnalysisStep, AnalysisDepth
from kg_executor import validate_enrichment, validate_neurons, validate_projections


class CircuitAgent(NeuroscienceAgent):
    """
    Agent for gene-centric circuit discovery analysis.

    This agent reasons about:
    1. Gene marker transcriptomic identity
    2. Spatial enrichment patterns
    3. Circuit ranking strategies
    4. Projection target analysis
    """

    def __init__(
        self,
        gene: str = "Car3",
        seed: int = 42,
        output_dir: str = "./outputs/circuit_discovery",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        # Set gene before super().__init__() because get_run_id() needs it
        self.gene = gene
        super().__init__(
            seed=seed,
            output_dir=output_dir,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            depth=AnalysisDepth.DEEP
        )

        # Data storage
        self.subclass_identity: pd.DataFrame = pd.DataFrame()
        self.region_enrichment: pd.DataFrame = pd.DataFrame()
        self.primary_region: str = ""
        self.neurons: pd.DataFrame = pd.DataFrame()
        self.projections: pd.DataFrame = pd.DataFrame()
        self.target_composition: pd.DataFrame = pd.DataFrame()

    def get_run_id(self) -> str:
        return f"circuit_discovery_{self.gene}"

    def get_analysis_goal(self) -> str:
        return f"Discover and characterize {self.gene}+ neuron circuits"

    def generate_plan(self, question: str) -> List[AnalysisStep]:
        """Generate analysis plan for gene circuit discovery."""
        self.think(
            f"Planning {self.gene} circuit analysis",
            {'gene': self.gene, 'question': question}
        )

        steps = [
            # Panel A: Identify gene-marked subclass
            AnalysisStep(
                step_number=1,
                purpose=f"Identify {self.gene}-marked transcriptomic subclass",
                template_name='get_subclass_by_gene',
                params={'GENE': self.gene},
                modality='molecular',
                reasoning=f"Need to find which subclass is marked by {self.gene} gene",
                expected_outcome=f"Subclass name and markers for {self.gene}"
            ),
            # Panel B: Region enrichment
            AnalysisStep(
                step_number=2,
                purpose=f"Find regions enriched for {self.gene} subclass",
                template_name='get_region_enrichment_for_subclass',
                # params set dynamically after step 1
                modality='molecular',
                reasoning=f"Need spatial distribution of {self.gene} subclass",
                depends_on=[1],
                expected_outcome="Ranked list of regions by enrichment"
            ),
            # Panel C: Neurons in primary region
            AnalysisStep(
                step_number=3,
                purpose="Get neurons in primary enriched region",
                template_name='get_neurons_in_region',
                # params set dynamically after step 2
                modality='morphological',
                reasoning="Need morphological data for neurons in top region",
                depends_on=[2],
                expected_outcome="List of neurons with morphology data"
            ),
            # Panel D: Projection patterns
            AnalysisStep(
                step_number=4,
                purpose="Analyze projection patterns from primary region",
                template_name='get_neuron_projections',
                # params set dynamically
                modality='projection',
                reasoning="Need to understand where neurons project to",
                depends_on=[3],
                expected_outcome="Neuron x target projection matrix"
            ),
            # Panel E: Target composition
            AnalysisStep(
                step_number=5,
                purpose="Profile molecular composition of projection targets",
                template_name='get_target_molecular_profile',
                modality='molecular',
                reasoning="Need to understand what cell types receive projections",
                depends_on=[4],
                expected_outcome="Subclass composition for each target"
            ),
        ]

        return steps

    def compile_results(self) -> Dict[str, Any]:
        """Compile final results with panel data."""
        results = {
            'success': True,
            'gene': self.gene,
            'primary_region': self.primary_region,
            'files': self.results.get('files', {}),
            'summary': {
                'subclass_found': not self.subclass_identity.empty,
                'n_enriched_regions': len(self.region_enrichment),
                'n_neurons': len(self.neurons),
                'n_projection_targets': self.projections['target_acronym'].nunique() if not self.projections.empty and 'target_acronym' in self.projections.columns else 0,
            },
            'evidence': self.get_evidence_summary()
        }
        return results

    def run(self, question: str = None) -> Dict[str, Any]:
        """Run the full Circuit analysis with agent reasoning."""
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
            f"Starting {self.gene} circuit discovery analysis",
            {'gene': self.gene, 'question': question}
        )

        # Connect to KG
        if not self.executor.connect():
            self.results['success'] = False
            self.results['error'] = "Failed to connect to Neo4j"
            return self.results

        try:
            # Panel A: Identify subclass
            self._panel_a_subclass_identity()

            # Panel B: Region enrichment
            self._panel_b_region_enrichment()

            # Panel C: Multi-modal evidence (neurons + morphology)
            self._panel_c_multimodal_evidence()

            # Panel D: Projection patterns
            self._panel_d_projections()

            # Panel E: Target composition
            self._panel_e_target_composition()

            # Panel F: Subgraph export
            self._panel_f_subgraph()

            # Generate visualizations
            self._generate_panel_b_figure()

            # Generate report
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

    def _panel_a_subclass_identity(self):
        """Panel A: Identify gene-marked subclass."""
        self.think(
            f"Panel A: Identifying transcriptomic subclass for {self.gene}",
            {'step': 'subclass_identity', 'gene': self.gene}
        )

        result = self.executor.execute_with_reasoning(
            template_name='get_subclass_by_gene',
            params={'GENE': self.gene},
            reasoning=f"Finding which subclass is marked by {self.gene}",
            step_number=1,
            validate_fn=lambda df: (1.0 if not df.empty else 0.0,
                                    [f"Found {len(df)} subclass(es)"] if not df.empty else ["No subclass found"])
        )

        if result.success and not result.data.empty:
            self.subclass_identity = result.data
            subclass_name = result.data.iloc[0]['subclass_name']
            markers = result.data.iloc[0]['markers']

            # Save
            path = self.save_dataframe(result.data, "panel_a_subclass_identity.csv")
            self.results['files'] = self.results.get('files', {})
            self.results['files']['panel_a'] = path

            # Reflect
            self.prov.log_reflect(
                step_number=1,
                validation_status='passed',
                confidence=1.0,
                should_replan=False,
                recommendations=[
                    f"Panel A: Identified {self.gene} subclass as '{subclass_name}'",
                    f"Markers: {markers}",
                    f"This defines the transcriptomic identity for {self.gene}+ neurons"
                ]
            )
        else:
            self.think(f"WARNING: No subclass found for {self.gene}")

    def _panel_b_region_enrichment(self):
        """Panel B: Region enrichment analysis."""
        if self.subclass_identity.empty:
            self.think("Skipping Panel B - no subclass identified")
            return

        subclass_name = self.subclass_identity.iloc[0]['subclass_name']

        self.think(
            f"Panel B: Finding regions enriched for '{subclass_name}'",
            {'subclass': subclass_name}
        )

        result = self.executor.execute_with_reasoning(
            template_name='get_region_enrichment_for_subclass',
            params={'SUBCLASS_NAME': subclass_name},
            reasoning=f"Finding spatial distribution of {subclass_name}",
            step_number=2,
            validate_fn=validate_enrichment
        )

        if result.success and not result.data.empty:
            self.region_enrichment = result.data
            self.primary_region = result.data.iloc[0]['region_name']
            top_pct = result.data.iloc[0]['pct_cells']
            n_regions = len(result.data)

            # Save
            path = self.save_dataframe(result.data, "panel_b_region_enrichment.csv")
            self.results['files']['panel_b'] = path

            # Reflect with detailed findings
            is_dominant = top_pct > 0.15
            self.prov.log_reflect(
                step_number=2,
                validation_status='passed',
                confidence=1.0 if is_dominant else 0.8,
                should_replan=False,
                recommendations=[
                    f"Panel B: {self.gene} subclass found in {n_regions} brain regions",
                    f"Top enriched region: {self.primary_region} at {top_pct*100:.1f}%",
                    f"{self.primary_region} is {'DOMINANT' if is_dominant else 'enriched'} for {self.gene}",
                    f"This confirms {self.primary_region} as the primary region for circuit analysis"
                ]
            )

    def _panel_c_multimodal_evidence(self):
        """Panel C: Multi-modal evidence (neurons + morphology)."""
        if not self.primary_region:
            self.think("Skipping Panel C - no primary region identified")
            return

        self.think(
            f"Panel C: Getting multi-modal evidence for {self.primary_region}",
            {'region': self.primary_region}
        )

        # Get neurons
        result = self.executor.execute_with_reasoning(
            template_name='get_neurons_in_region',
            params={'REGION': self.primary_region},
            reasoning=f"Finding neurons in {self.primary_region}",
            step_number=3,
            validate_fn=validate_neurons
        )

        if result.success and not result.data.empty:
            self.neurons = result.data
            n_neurons = len(result.data)
            neuron_method = 'LOCATE_AT'
        else:
            # Fallback query
            self.think(f"Primary query returned 0 neurons, trying fallback for {self.primary_region}")
            result = self.executor.execute_with_reasoning(
                template_name='get_neurons_fallback',
                params={'REGION': self.primary_region},
                reasoning=f"Fallback: searching neurons by base_region/celltype for {self.primary_region}",
                step_number=3
            )
            if result.success and not result.data.empty:
                self.neurons = result.data
                n_neurons = len(result.data)
                neuron_method = 'fallback'
            else:
                n_neurons = 0
                neuron_method = 'none'

        # Get morphology data for region
        morph_result = self.executor.execute_with_reasoning(
            template_name='get_morphology_signature',
            params={'REGION': self.primary_region},
            reasoning=f"Getting morphological properties for {self.primary_region}",
            step_number=3
        )

        # Build Panel C data
        panel_c_rows = []

        # Molecular evidence from Panel B
        if not self.region_enrichment.empty:
            region_row = self.region_enrichment[self.region_enrichment['region_name'] == self.primary_region]
            if not region_row.empty:
                panel_c_rows.append({
                    'modality': 'molecular',
                    'property': 'pct_cells',
                    'value': region_row.iloc[0]['pct_cells'],
                    'source': 'HAS_SUBCLASS relationship'
                })
                panel_c_rows.append({
                    'modality': 'molecular',
                    'property': 'rank',
                    'value': region_row.iloc[0]['rank'],
                    'source': 'HAS_SUBCLASS relationship'
                })

        # Morphology evidence
        if morph_result.success and not morph_result.data.empty:
            morph_props = ['axonal_bifurcation_remote_angle', 'axonal_length', 'axonal_branches',
                          'axonal_max_branch_order', 'dendritic_bifurcation_remote_angle',
                          'dendritic_length', 'dendritic_branches', 'dendritic_max_branch_order']
            for prop in morph_props:
                if prop in morph_result.data.columns:
                    panel_c_rows.append({
                        'modality': 'morphology',
                        'property': prop,
                        'value': morph_result.data.iloc[0].get(prop, 'NaN'),
                        'source': 'Region node properties'
                    })

        # Neuron count
        panel_c_rows.append({
            'modality': 'morphological',
            'property': 'neuron_count',
            'value': n_neurons,
            'source': neuron_method
        })

        # Sample neurons
        for i, row in self.neurons.head(10).iterrows():
            panel_c_rows.append({
                'modality': 'morphological',
                'property': 'neuron',
                'value': f"{row.get('neuron_id', '')}|{row.get('celltype', '')}",
                'source': 'Neuron nodes'
            })

        # Save
        panel_c_df = pd.DataFrame(panel_c_rows)
        path = self.save_dataframe(panel_c_df, "panel_c_cla_multimodal.csv")
        self.results['files']['panel_c'] = path

        # Reflect
        self.prov.log_reflect(
            step_number=3,
            validation_status='passed' if n_neurons > 0 else 'low_confidence',
            confidence=min(1.0, n_neurons / 10),
            should_replan=False,
            recommendations=[
                f"Panel C: Found {n_neurons} neurons in {self.primary_region}",
                f"Neuron query method: {neuron_method}",
                f"Morphology data: {'available' if morph_result.success else 'not available'}"
            ]
        )

    def _panel_d_projections(self):
        """Panel D: Projection patterns."""
        if not self.primary_region:
            self.think("Skipping Panel D - no primary region identified")
            return

        self.think(
            f"Panel D: Analyzing projection patterns from {self.primary_region}",
            {'region': self.primary_region}
        )

        result = self.executor.execute_with_reasoning(
            template_name='get_neuron_projections',
            params={'REGION': self.primary_region},
            reasoning=f"Getting neuron-level projections from {self.primary_region}",
            step_number=4,
            validate_fn=validate_projections
        )

        if result.success and not result.data.empty:
            self.projections = result.data

            # Add log10 transform
            self.projections['proj_log10'] = np.log10(1 + self.projections['proj_strength'])

            # Save raw projections
            path = self.save_dataframe(self.projections, "panel_d_raw_projections.csv")
            self.results['files']['panel_d_raw'] = path

            # Create pivot matrix
            pivot = self.projections.pivot_table(
                index='neuron_id',
                columns='target_acronym',
                values='proj_log10',
                fill_value=0,
                aggfunc='sum'
            )
            path = self.save_dataframe(pivot.reset_index(), "panel_d_projection_matrix.csv")
            self.results['files']['panel_d_matrix'] = path

            # Analyze findings
            n_neurons = self.projections['neuron_id'].nunique()
            n_targets = self.projections['target_acronym'].nunique()
            top_target = self.projections.groupby('target_acronym')['proj_strength'].sum().idxmax()

            # Reflect
            self.prov.log_reflect(
                step_number=4,
                validation_status='passed',
                confidence=1.0,
                should_replan=False,
                recommendations=[
                    f"Panel D: Projection matrix {n_neurons} neurons x {n_targets} targets",
                    f"Top projection target: {top_target}",
                    f"Applied log10(1+x) transform per canonical method"
                ]
            )

    def _panel_e_target_composition(self):
        """Panel E: Target molecular composition."""
        if self.projections.empty:
            self.think("Skipping Panel E - no projection data")
            return

        self.think("Panel E: Profiling molecular composition of projection targets")

        # Get top 10 targets by total projection strength
        target_totals = self.projections.groupby('target_acronym').agg(
            total=('proj_strength', 'sum'),
            eid=('target_eid', 'first')
        ).nlargest(10, 'total')

        all_compositions = []
        for target_acronym, row in target_totals.iterrows():
            target_eid = row['eid']

            result = self.executor.execute_with_reasoning(
                template_name='get_target_molecular_profile',
                params={'TARGET_EID': target_eid},
                reasoning=f"Getting molecular composition for target {target_acronym}",
                step_number=5
            )

            if result.success and not result.data.empty:
                for _, comp_row in result.data.iterrows():
                    all_compositions.append({
                        'target_name': target_acronym,
                        'subclass_name': comp_row['subclass_name'],
                        'markers': comp_row['markers'],
                        'pct_cells': comp_row['pct_cells'],
                        'rank': comp_row['rank']
                    })

        if all_compositions:
            self.target_composition = pd.DataFrame(all_compositions)
            path = self.save_dataframe(self.target_composition, "panel_e_target_full_composition.csv")
            self.results['files']['panel_e'] = path

            # Reflect
            n_targets = self.target_composition['target_name'].nunique()
            self.prov.log_reflect(
                step_number=5,
                validation_status='passed',
                confidence=1.0,
                should_replan=False,
                recommendations=[
                    f"Panel E: Profiled molecular composition for {n_targets} targets",
                    f"Total composition entries: {len(self.target_composition)}"
                ]
            )

    def _panel_f_subgraph(self):
        """Panel F: Export subgraph as Cypher."""
        self.think("Panel F: Generating subgraph export")

        lines = [
            f"// Panel F: {self.gene} Circuit Subgraph",
            f"// Generated by CircuitAgent",
            ""
        ]

        # Subclass node
        if not self.subclass_identity.empty:
            sc = self.subclass_identity.iloc[0]
            lines.append(f"MERGE (sc:Subclass {{name: '{sc['subclass_name']}', markers: '{sc['markers']}'}})")

        # Region nodes from enrichment
        for _, row in self.region_enrichment.head(10).iterrows():
            lines.append(f"MERGE (r_{row['region_name']}:Region {{acronym: '{row['region_name']}'}})")

        # Enrichment relationships
        if not self.subclass_identity.empty:
            sc_name = self.subclass_identity.iloc[0]['subclass_name']
            for _, row in self.region_enrichment.head(10).iterrows():
                lines.append(f"MERGE (r_{row['region_name']})-[:HAS_SUBCLASS {{pct_cells: {row['pct_cells']:.6f}}}]->(sc)")

        # Neuron nodes
        for _, row in self.neurons.head(20).iterrows():
            nid = row.get('neuron_id', '').replace("'", "\\'")
            lines.append(f"MERGE (n_{nid[:20]}:Neuron {{neuron_id: '{nid}'}})")

        # Projection relationships
        for _, row in self.projections.head(50).iterrows():
            nid = row.get('neuron_id', '').replace("'", "\\'")[:20]
            target = row.get('target_acronym', '')
            strength = row.get('proj_strength', 0)
            lines.append(f"MERGE (n_{nid})-[:PROJECT_TO {{strength: {strength:.2f}}}]->(t_{target}:Target {{acronym: '{target}'}})")

        # Save
        cypher_content = '\n'.join(lines)
        path = self.data_dir / "panel_f_subgraph.cypher"
        with open(path, 'w') as f:
            f.write(cypher_content)
        self.results['files']['panel_f'] = str(path)

        self.prov.log_reflect(
            step_number=6,
            validation_status='passed',
            confidence=1.0,
            should_replan=False,
            recommendations=[f"Panel F: Generated {len(lines)} lines of Cypher"]
        )

    def _generate_panel_b_figure(self):
        """Generate Panel B visualization (region enrichment bar chart)."""
        if self.region_enrichment.empty:
            return

        self.think("Generating Panel B figure")

        regions = self.region_enrichment['region_name'].tolist()
        pct_values = [p * 100 for p in self.region_enrichment['pct_cells'].tolist()]

        # Red to yellow gradient
        n = len(regions)
        red = np.array([231, 76, 60]) / 255
        yellow = np.array([241, 196, 15]) / 255
        colors = [red * (1 - i/(n-1)) + yellow * (i/(n-1)) for i in range(n)] if n > 1 else [red]

        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.set_facecolor('white')

        x_pos = np.arange(len(regions))
        bars = ax.bar(x_pos, pct_values, color=colors, edgecolor='#2C3E50', linewidth=0.8, width=0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Percentage of Cells (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
        ax.set_title(f'B    Region Enrichment of {self.gene} Subclass', fontsize=14, fontweight='bold', loc='left')

        # Add labels on bars
        for bar, pct in zip(bars, pct_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        path = self.data_dir / "region_enrichment.png"
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        self.results['files']['panel_b_figure'] = str(path)

    def _generate_report(self):
        """Generate integrated report with inline citations."""
        self.think("Generating integrated report")

        lines = [
            f"# Circuit Analysis: {self.gene}+ Neuron Circuit Analysis",
            "",
            f"**Gene Marker:** {self.gene}",
            f"**Primary Region:** {self.primary_region}",
            f"**Seed:** {self.seed}",
            "",
            "## Executive Summary",
            ""
        ]

        # Panel A summary
        if not self.subclass_identity.empty:
            sc = self.subclass_identity.iloc[0]
            lines.append(f"- **{self.gene} subclass:** \"{sc['subclass_name']}\" (markers: {sc['markers']})")

        # Panel B summary
        if not self.region_enrichment.empty:
            top = self.region_enrichment.iloc[0]
            n_regions = len(self.region_enrichment)
            lines.append(f"- **Top enriched region:** {top['region_name']} at {top['pct_cells']*100:.1f}%")
            lines.append(f"- **{n_regions} regions** contain {self.gene} subclass")

        # Panel C summary
        if not self.neurons.empty:
            lines.append(f"- **{len(self.neurons)} neurons** in {self.primary_region}")

        # Panel D summary
        if not self.projections.empty:
            n_neurons = self.projections['neuron_id'].nunique()
            n_targets = self.projections['target_acronym'].nunique()
            lines.append(f"- **Projection matrix:** {n_neurons} neurons x {n_targets} targets")

        # Panel E summary
        if not self.target_composition.empty:
            n_targets = self.target_composition['target_name'].nunique()
            lines.append(f"- **Target profiles:** {n_targets} targets analyzed")

        lines.append("")
        lines.append("## Agent Reasoning Trace")
        lines.append("")
        for finding in self.state.findings[:20]:
            lines.append(f"- {finding}")

        lines.append("")
        lines.append("## Evidence Summary")
        lines.append("")
        lines.append(self.evidence.to_markdown())

        # Save report
        report_content = '\n'.join(lines)
        path = Path(self.output_dir) / "report.md"
        with open(path, 'w') as f:
            f.write(report_content)
        self.results['files']['report'] = str(path)


def run_circuit_discovery(
    gene: str = "Car3",
    seed: int = 42,
    output_dir: str = "./outputs/circuit_discovery",
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
) -> Dict[str, Any]:
    """
    Convenience function to run Circuit with agent reasoning.
    """
    agent = CircuitAgent(
        gene=gene,
        seed=seed,
        output_dir=output_dir,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    return agent.run()


if __name__ == "__main__":
    result = run_circuit_discovery(gene="Car3", seed=42)
    print(f"Success: {result.get('success')}")
    print(f"Primary region: {result.get('primary_region')}")
    print(f"Files: {list(result.get('files', {}).keys())}")
