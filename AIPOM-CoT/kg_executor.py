import os
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

from query_templates import TEMPLATES, QueryTemplate, get_template
from provenance import ProvenanceLogger
from evidence_buffer import EvidenceBuffer


@dataclass
class QueryResult:
    """Result of a KG query execution."""
    template_name: str
    success: bool
    data: pd.DataFrame
    row_count: int
    query_hash: str
    reasoning: str
    confidence: float = 1.0
    findings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def is_empty(self) -> bool:
        return self.row_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'template_name': self.template_name,
            'success': self.success,
            'row_count': self.row_count,
            'query_hash': self.query_hash,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'findings': self.findings,
            'error': self.error
        }


class KGExecutor:
    """
    Neo4j query executor with agent-style reasoning and TPAR logging.

    Features:
    - Uses unified query templates
    - Logs reasoning for each query
    - Tracks evidence automatically
    - Supports validation callbacks
    - Deterministic with seed
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        provenance: Optional[ProvenanceLogger] = None,
        evidence: Optional[EvidenceBuffer] = None,
        seed: int = 42
    ):
        self.uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687")
        self.user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neuroxiv")
        self.prov = provenance
        self.eb = evidence
        self.seed = seed
        self._driver = None
        self._session = None
        self.query_count = 0
        self.total_rows = 0

        np.random.seed(seed)

    def connect(self) -> bool:
        """Establish Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self._driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                record = result.single()
                return record and record['ok'] == 1
        except Exception as e:
            if self.prov:
                self.prov.log_think(f"Neo4j connection failed: {e}", {'uri': self.uri})
            return False

    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _compute_query_hash(self, cypher: str) -> str:
        """Compute a hash for query deduplication and provenance."""
        return hashlib.md5(cypher.encode()).hexdigest()[:12]

    def execute_with_reasoning(
        self,
        template_name: str,
        params: Dict[str, Any] = None,
        reasoning: str = "",
        step_number: int = 0,
        expected_rows: int = None,
        validate_fn: Callable[[pd.DataFrame], Tuple[float, List[str]]] = None
    ) -> QueryResult:
        """
        Execute a query template with full reasoning and logging.

        Args:
            template_name: Name of the query template
            params: Parameters to pass to the query
            reasoning: Agent's reasoning for why this query is being run
            step_number: Step number in the analysis workflow
            expected_rows: Expected number of rows (for confidence calculation)
            validate_fn: Optional validation function (returns confidence, findings)

        Returns:
            QueryResult with data, confidence, and findings
        """
        params = params or {}
        template = get_template(template_name)

        # Build the full reasoning
        full_reasoning = reasoning or template.reasoning_hint
        if self.prov:
            self.prov.log_think(
                f"Step {step_number}: {full_reasoning}",
                {'template': template_name, 'params': params, 'purpose': template.purpose}
            )

        # Get the Cypher query
        cypher = template.cypher
        query_hash = self._compute_query_hash(cypher + str(params))

        # Log the action
        if self.prov:
            self.prov.log_act(
                step_number=step_number,
                action_type='kg_query',
                purpose=template.purpose,
                query=cypher[:100],
                params=params,
                result_summary={'template': template_name}
            )

        # Execute the query
        try:
            if not self._driver:
                self.connect()

            with self._driver.session() as session:
                result = session.run(cypher, **params)
                records = [dict(rec) for rec in result]

            df = pd.DataFrame(records) if records else pd.DataFrame()
            row_count = len(df)
            self.query_count += 1
            self.total_rows += row_count

            # Calculate confidence
            if validate_fn:
                confidence, findings = validate_fn(df)
            else:
                # Default confidence based on row count
                if expected_rows:
                    confidence = min(1.0, row_count / expected_rows)
                elif row_count == 0 and not template.can_return_empty:
                    confidence = 0.0
                elif row_count > 0:
                    confidence = 1.0
                else:
                    confidence = 0.5  # Empty but allowed

                findings = [f"Retrieved {row_count} rows"]
                if row_count > 0 and len(df.columns) > 0:
                    findings.append(f"Columns: {', '.join(df.columns[:5])}")

            # Add to evidence buffer
            if self.eb and row_count > 0:
                evidence_data = records[:10] if records else []
                key_fields = list(df.columns[:3]) if len(df.columns) > 0 else ['result']
                self.eb.add_evidence(
                    modality=template.modality,
                    source_step=step_number,
                    query=cypher[:80],
                    data=evidence_data,
                    key_fields=key_fields
                )

            # Log reflection
            if self.prov:
                status = 'passed' if confidence >= 0.7 else ('low_confidence' if confidence > 0 else 'failed')
                self.prov.log_reflect(
                    step_number=step_number,
                    validation_status=status,
                    confidence=confidence,
                    should_replan=confidence < 0.3,
                    recommendations=findings
                )

            return QueryResult(
                template_name=template_name,
                success=True,
                data=df,
                row_count=row_count,
                query_hash=query_hash,
                reasoning=full_reasoning,
                confidence=confidence,
                findings=findings
            )

        except Exception as e:
            error_msg = str(e)
            if self.prov:
                self.prov.log_reflect(
                    step_number=step_number,
                    validation_status='failed',
                    confidence=0.0,
                    should_replan=True,
                    recommendations=[f"Query failed: {error_msg}"]
                )

            return QueryResult(
                template_name=template_name,
                success=False,
                data=pd.DataFrame(),
                row_count=0,
                query_hash=query_hash,
                reasoning=full_reasoning,
                confidence=0.0,
                findings=[f"Error: {error_msg}"],
                error=error_msg
            )

    def execute_raw(self, cypher: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a raw Cypher query without reasoning (for simple lookups)."""
        params = params or {}
        if not self._driver:
            self.connect()

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            records = [dict(rec) for rec in result]

        self.query_count += 1
        return pd.DataFrame(records) if records else pd.DataFrame()

    def batch_execute(
        self,
        template_name: str,
        param_list: List[Dict[str, Any]],
        reasoning: str = "",
        step_number: int = 0,
        aggregate: bool = True
    ) -> QueryResult:
        """
        Execute a template multiple times with different parameters.

        Args:
            template_name: Name of the query template
            param_list: List of parameter dicts
            reasoning: Reasoning for the batch
            step_number: Step number
            aggregate: If True, combine results into single DataFrame

        Returns:
            Combined QueryResult
        """
        template = get_template(template_name)
        all_records = []
        total_confidence = 0.0

        if self.prov:
            self.prov.log_think(
                f"Step {step_number}: Batch execution of {template_name} ({len(param_list)} iterations)",
                {'template': template_name, 'batch_size': len(param_list)}
            )
            self.prov.log_act(
                step_number=step_number,
                action_type='kg_query_batch',
                purpose=f"Batch: {template.purpose}",
                query=template.cypher[:80],
                params={'batch_size': len(param_list)}
            )

        if not self._driver:
            self.connect()

        with self._driver.session() as session:
            for params in param_list:
                result = session.run(template.cypher, **params)
                records = [dict(rec) for rec in result]
                all_records.extend(records)
                self.query_count += 1

        df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
        row_count = len(df)
        self.total_rows += row_count

        confidence = min(1.0, row_count / (len(param_list) * 5)) if row_count > 0 else 0.0
        findings = [
            f"Batch executed {len(param_list)} queries",
            f"Total rows: {row_count}",
            f"Average rows per query: {row_count / len(param_list):.1f}" if param_list else "No queries"
        ]

        if self.eb and row_count > 0:
            self.eb.add_evidence(
                modality=template.modality,
                source_step=step_number,
                query=f"Batch: {template_name}",
                data=all_records[:10],
                key_fields=list(df.columns[:3]) if len(df.columns) > 0 else ['result']
            )

        if self.prov:
            self.prov.log_reflect(
                step_number=step_number,
                validation_status='passed' if confidence >= 0.5 else 'low_confidence',
                confidence=confidence,
                should_replan=confidence < 0.3,
                recommendations=findings
            )

        return QueryResult(
            template_name=template_name,
            success=True,
            data=df,
            row_count=row_count,
            query_hash=self._compute_query_hash(template.cypher + str(len(param_list))),
            reasoning=reasoning or f"Batch execution for {template.purpose}",
            confidence=confidence,
            findings=findings
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            'total_queries': self.query_count,
            'total_rows': self.total_rows,
            'neo4j_uri': self.uri,
            'seed': self.seed
        }


# Convenience functions for common patterns

def validate_enrichment(df: pd.DataFrame, min_pct: float = 0.05) -> Tuple[float, List[str]]:
    """Validate enrichment results (for gene/subclass enrichment queries)."""
    findings = []
    if df.empty:
        return 0.0, ["No enrichment data found"]

    if 'pct_cells' in df.columns:
        top_pct = df.iloc[0]['pct_cells']
        n_regions = len(df)
        findings.append(f"Top enrichment: {top_pct*100:.1f}%")
        findings.append(f"Found in {n_regions} regions")

        if top_pct >= min_pct:
            confidence = min(1.0, top_pct / 0.2)  # 20% = full confidence
            findings.append(f"Enrichment is {'significant' if top_pct > 0.15 else 'moderate'}")
        else:
            confidence = top_pct / min_pct
            findings.append(f"Enrichment below threshold ({min_pct*100:.0f}%)")

        return confidence, findings

    return 0.5, ["Missing pct_cells column"]


def validate_neurons(df: pd.DataFrame, min_count: int = 5) -> Tuple[float, List[str]]:
    """Validate neuron query results."""
    findings = []
    if df.empty:
        return 0.0, ["No neurons found"]

    n_neurons = len(df)
    findings.append(f"Found {n_neurons} neurons")

    if n_neurons >= min_count:
        confidence = 1.0
        findings.append("Sufficient neurons for analysis")
    else:
        confidence = n_neurons / min_count
        findings.append(f"Below minimum count ({min_count})")

    return confidence, findings


def validate_projections(df: pd.DataFrame, min_targets: int = 3) -> Tuple[float, List[str]]:
    """Validate projection query results."""
    findings = []
    if df.empty:
        return 0.0, ["No projections found"]

    if 'target_acronym' in df.columns or 'target_name' in df.columns:
        target_col = 'target_acronym' if 'target_acronym' in df.columns else 'target_name'
        n_targets = df[target_col].nunique()
        findings.append(f"Projections to {n_targets} unique targets")

        if n_targets >= min_targets:
            confidence = 1.0
            findings.append("Rich projection pattern")
        else:
            confidence = n_targets / min_targets
            findings.append("Limited projection targets")

        return confidence, findings

    return 0.5, [f"Found {len(df)} projection records"]
