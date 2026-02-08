import os
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

from provenance import ProvenanceLogger, create_provenance_logger
from evidence_buffer import EvidenceBuffer
from kg_executor import KGExecutor, QueryResult
from query_templates import TEMPLATES, get_template

# Optional LLM import
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class AnalysisDepth(Enum):
    """How deep should the analysis go?"""
    SHALLOW = "shallow"   # Quick overview, few queries
    MEDIUM = "medium"     # Standard analysis
    DEEP = "deep"         # Comprehensive, many queries


class Modality(Enum):
    """Data modalities for multi-modal analysis."""
    MOLECULAR = "molecular"
    MORPHOLOGICAL = "morphological"
    PROJECTION = "projection"


@dataclass
class AnalysisStep:
    """A single step in the agent's analysis plan."""
    step_number: int
    purpose: str
    template_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    modality: str = "molecular"
    reasoning: str = ""
    depends_on: List[int] = field(default_factory=list)
    expected_outcome: str = ""


@dataclass
class AnalysisState:
    """Current state of the analysis (for adaptive planning)."""
    discovered_entities: Dict[str, List[Any]] = field(default_factory=dict)
    executed_steps: List[int] = field(default_factory=list)
    modalities_covered: List[str] = field(default_factory=list)
    current_focus: str = ""
    confidence_scores: Dict[int, float] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    should_continue: bool = True


class NeuroscienceAgent(ABC):
    """
    Base class for agent-driven MS reproduction.

    Implements TPAR workflow:
    - THINK: Articulate goal and reasoning
    - PLAN: Generate candidate steps based on state
    - ACT: Execute queries with evidence logging
    - REFLECT: Evaluate results and decide next steps
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: str = "./outputs",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        depth: AnalysisDepth = AnalysisDepth.DEEP,
        use_llm: bool = True,
        llm_model: str = "gpt-4o-mini"
    ):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.depth = depth
        self.llm_model = llm_model

        # Set random seeds for determinism
        np.random.seed(seed)

        # Initialize LLM client if available and requested
        self.llm_client = None
        self.use_llm = use_llm and HAS_OPENAI
        if self.use_llm:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
            else:
                self.use_llm = False

        # Initialize components
        self.prov = create_provenance_logger(run_id=self.get_run_id(), seed=seed)
        self.evidence = EvidenceBuffer()
        self.executor = KGExecutor(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            provenance=self.prov,
            evidence=self.evidence,
            seed=seed
        )

        # Analysis state
        self.state = AnalysisState()
        self.results: Dict[str, Any] = {}
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_run_id(self) -> str:
        """Return unique run ID for this agent type."""
        pass

    @abstractmethod
    def get_analysis_goal(self) -> str:
        """Return the high-level goal of this analysis."""
        pass

    @abstractmethod
    def generate_plan(self, question: str) -> List[AnalysisStep]:
        """Generate analysis plan based on question and current state."""
        pass

    @abstractmethod
    def compile_results(self) -> Dict[str, Any]:
        """Compile final results after analysis."""
        pass

    # =========================================================================
    # LLM-DRIVEN REASONING METHODS
    # =========================================================================

    def llm_reason(self, prompt: str, system_prompt: str = None) -> str:
        """
        Use LLM to reason about the analysis.

        Args:
            prompt: The reasoning prompt
            system_prompt: Optional system prompt

        Returns:
            LLM's response text
        """
        if not self.llm_client:
            return ""

        system_prompt = system_prompt or """You are a neuroscience analysis agent.
You analyze brain connectivity data from a knowledge graph containing:
- Brain regions with molecular (transcriptomic) profiles
- Neuron morphology features (axonal/dendritic properties)
- Projection patterns between regions

Be concise and scientific in your reasoning."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for determinism
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.state.findings.append(f"[LLM_ERROR] {str(e)}")
            return ""

    def llm_select_templates(self, question: str, available_templates: List[str]) -> List[str]:
        """
        Use LLM to select appropriate query templates based on the question.

        Args:
            question: The analysis question
            available_templates: List of available template names

        Returns:
            List of selected template names
        """
        if not self.llm_client:
            return available_templates  # Return all if no LLM

        # Get template descriptions
        template_info = []
        for name in available_templates:
            try:
                t = get_template(name)
                template_info.append(f"- {name}: {t.purpose}")
            except:
                template_info.append(f"- {name}: (no description)")

        prompt = f"""Given this neuroscience analysis question:
"{question}"

Select the most relevant query templates from this list:
{chr(10).join(template_info)}

Return ONLY a JSON list of template names, e.g.: ["template1", "template2"]
Select templates that directly help answer the question."""

        response = self.llm_reason(prompt)

        try:
            # Parse JSON response
            selected = json.loads(response)
            if isinstance(selected, list):
                return [t for t in selected if t in available_templates]
        except:
            pass

        return available_templates  # Fallback to all templates

    def llm_analyze_results(self, step_purpose: str, result_summary: Dict) -> Tuple[float, List[str]]:
        """
        Use LLM to analyze query results and provide insights.

        Args:
            step_purpose: What this step was trying to accomplish
            result_summary: Summary of the query results

        Returns:
            (confidence_score, list_of_findings)
        """
        if not self.llm_client:
            # Default analysis without LLM
            row_count = result_summary.get('row_count', 0)
            if row_count == 0:
                return 0.0, ["No data found"]
            elif row_count < 5:
                return 0.5, [f"Limited data: {row_count} rows"]
            else:
                return 1.0, [f"Found {row_count} rows"]

        prompt = f"""Analyze these query results for a neuroscience analysis step.

Step purpose: {step_purpose}
Results: {json.dumps(result_summary, indent=2)}

Provide:
1. A confidence score (0.0-1.0) for how well this step achieved its purpose
2. Key findings (2-3 bullet points)

Return JSON: {{"confidence": 0.X, "findings": ["finding1", "finding2"]}}"""

        response = self.llm_reason(prompt)

        try:
            result = json.loads(response)
            confidence = float(result.get('confidence', 0.5))
            findings = result.get('findings', [])
            return confidence, findings
        except:
            row_count = result_summary.get('row_count', 0)
            return (1.0 if row_count > 0 else 0.0), [f"Retrieved {row_count} rows"]

    # =========================================================================
    # TPAR WORKFLOW METHODS
    # =========================================================================

    def think(self, thought: str, context: Dict[str, Any] = None) -> str:
        """
        THINK phase: Articulate reasoning about current goal.
        If LLM is available, uses LLM to expand on the thought.

        Args:
            thought: The agent's initial reasoning
            context: Additional context (parameters, state, etc.)

        Returns:
            Expanded thought (from LLM) or original thought
        """
        context = context or {}
        context['seed'] = self.seed
        context['depth'] = self.depth.value
        context['state'] = {
            'executed_steps': self.state.executed_steps,
            'modalities_covered': self.state.modalities_covered,
            'current_focus': self.state.current_focus
        }

        # If LLM available, expand the thought
        expanded_thought = thought
        if self.llm_client and context.get('use_llm', True):
            llm_expansion = self.llm_reason(
                f"Expand on this analysis thought: {thought}\nContext: {json.dumps(context, default=str)}"
            )
            if llm_expansion:
                expanded_thought = f"{thought}\n[LLM]: {llm_expansion}"

        self.prov.log_think(expanded_thought, context)
        self.state.findings.append(f"[THINK] {thought}")

        return expanded_thought

    def plan(self, steps: List[AnalysisStep], rationale: str = "") -> List[AnalysisStep]:
        """
        PLAN phase: Generate explicit plan with reasoning.

        Args:
            steps: List of AnalysisStep objects
            rationale: Why this plan was chosen

        Returns:
            The plan (for execution)
        """
        plan_data = []
        for step in steps:
            plan_data.append({
                'step': step.step_number,
                'purpose': step.purpose,
                'template': step.template_name,
                'modality': step.modality,
                'reasoning': step.reasoning,
                'depends_on': step.depends_on,
                'expected_outcome': step.expected_outcome
            })

        self.prov.log_plan(plan_data, planner_type=f'{self.get_run_id()}_agent')
        return steps

    def act(self, step: AnalysisStep) -> QueryResult:
        """
        ACT phase: Execute a step with evidence logging.

        Args:
            step: The analysis step to execute

        Returns:
            QueryResult with data and metadata
        """
        result = self.executor.execute_with_reasoning(
            template_name=step.template_name,
            params=step.params,
            reasoning=step.reasoning or step.purpose,
            step_number=step.step_number
        )

        # Update state
        self.state.executed_steps.append(step.step_number)
        if step.modality not in self.state.modalities_covered:
            self.state.modalities_covered.append(step.modality)
        self.state.confidence_scores[step.step_number] = result.confidence

        return result

    def reflect(self, step: AnalysisStep, result: QueryResult) -> Tuple[bool, List[str]]:
        """
        REFLECT phase: Evaluate results and decide next steps.
        Uses LLM for deeper analysis when available.

        Args:
            step: The step that was executed
            result: The result from act()

        Returns:
            (should_continue, recommendations)
        """
        findings = []

        # Analyze the result
        if result.success and result.row_count > 0:
            findings.append(f"Step {step.step_number} succeeded: {result.row_count} rows")
            findings.extend(result.findings)

            # Extract key insights based on modality
            if step.modality == 'molecular' and 'pct_cells' in result.data.columns:
                top_pct = result.data.iloc[0]['pct_cells']
                findings.append(f"Top enrichment: {top_pct*100:.1f}%")

            elif step.modality == 'morphological' and 'neuron_id' in result.data.columns:
                n_neurons = result.data['neuron_id'].nunique()
                findings.append(f"Found {n_neurons} unique neurons")

            elif step.modality == 'projection':
                if 'target_acronym' in result.data.columns:
                    n_targets = result.data['target_acronym'].nunique()
                    findings.append(f"Projections to {n_targets} targets")

            # Use LLM for deeper analysis if available
            if self.llm_client:
                result_summary = {
                    'row_count': result.row_count,
                    'columns': list(result.data.columns) if not result.data.empty else [],
                    'sample': result.data.head(3).to_dict() if not result.data.empty else {}
                }
                llm_confidence, llm_findings = self.llm_analyze_results(step.purpose, result_summary)
                findings.extend([f"[LLM] {f}" for f in llm_findings])

        else:
            findings.append(f"Step {step.step_number} returned empty or failed")
            if result.error:
                findings.append(f"Error: {result.error}")

        # Decide if replanning needed
        should_continue = result.confidence >= 0.3 or step.step_number < 3  # Continue for early steps

        # Update state findings
        self.state.findings.extend(findings)

        return should_continue, findings

    # =========================================================================
    # EXECUTION METHODS
    # =========================================================================

    def run(self, question: str = None) -> Dict[str, Any]:
        """
        Run the full TPAR analysis workflow.

        Args:
            question: The analysis question (uses default if None)

        Returns:
            Complete results dictionary
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
            f"Starting {self.get_run_id()} analysis",
            {'question': question, 'goal': self.get_analysis_goal()}
        )

        # Connect to KG
        if not self.executor.connect():
            self.results['success'] = False
            self.results['error'] = "Failed to connect to Neo4j"
            return self.results

        try:
            # PLAN: Generate analysis steps
            plan = self.generate_plan(question)
            plan = self.plan(plan, rationale=f"Agent-driven plan for: {question}")

            # ACT & REFLECT: Execute each step
            for step in plan:
                # Check dependencies
                if step.depends_on:
                    missing = [d for d in step.depends_on if d not in self.state.executed_steps]
                    if missing:
                        self.think(f"Skipping step {step.step_number} - dependencies not met: {missing}")
                        continue

                # Execute
                result = self.act(step)

                # Reflect
                should_continue, findings = self.reflect(step, result)

                # Store result
                self.results[f'step_{step.step_number}'] = result.to_dict()
                self.results[f'data_{step.step_number}'] = result.data

                # Check if we should stop
                if not should_continue and not self.state.should_continue:
                    self.think(f"Stopping early after step {step.step_number} due to low confidence")
                    break

            # Compile final results
            self.results = self.compile_results()
            self.results['success'] = True

        except Exception as e:
            self.results['success'] = False
            self.results['error'] = str(e)

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

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def save_dataframe(self, df: pd.DataFrame, filename: str, include_index: bool = True) -> str:
        """Save a DataFrame to the data directory.

        Args:
            df: DataFrame to save
            filename: Output filename
            include_index: Whether to include row labels (default True for matrices)
        """
        path = self.data_dir / filename
        df.to_csv(path, index=include_index)
        return str(path)

    def get_evidence_summary(self) -> Dict[str, Any]:
        """Get summary of collected evidence."""
        return {
            'kg_query_count': self.executor.query_count,
            'total_rows': self.executor.total_rows,
            'modalities_covered': self.state.modalities_covered,
            'coverage_rate': self.evidence.get_coverage_rate(),
            'evidence_markdown': self.evidence.to_markdown()
        }

    def reason_about_data(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Have the agent reason about a DataFrame's contents.

        Returns insights that can guide next steps.
        """
        insights = {
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'is_empty': df.empty
        }

        if df.empty:
            insights['reasoning'] = f"{name} is empty - may need alternative approach"
            return insights

        # Analyze based on columns
        if 'pct_cells' in df.columns:
            insights['type'] = 'enrichment'
            insights['top_value'] = df['pct_cells'].max()
            insights['mean_value'] = df['pct_cells'].mean()
            insights['reasoning'] = f"Enrichment data: max={insights['top_value']*100:.1f}%, mean={insights['mean_value']*100:.1f}%"

        elif 'neuron_id' in df.columns:
            insights['type'] = 'neurons'
            insights['n_unique'] = df['neuron_id'].nunique()
            insights['reasoning'] = f"Neuron data: {insights['n_unique']} unique neurons"

        elif 'proj_strength' in df.columns or 'total_projection_strength' in df.columns:
            col = 'proj_strength' if 'proj_strength' in df.columns else 'total_projection_strength'
            insights['type'] = 'projection'
            insights['total_strength'] = df[col].sum()
            insights['reasoning'] = f"Projection data: total strength = {insights['total_strength']:.1f}"

        else:
            insights['type'] = 'generic'
            insights['reasoning'] = f"Generic data with {len(df)} rows"

        # Log the reasoning
        self.think(f"Analyzed {name}: {insights['reasoning']}", insights)

        return insights
