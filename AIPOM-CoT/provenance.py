import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict


class EventType:
    """Provenance event types"""
    RUN_START = "RUN_START"
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    ATTRIBUTE_RESOLUTION = "ATTRIBUTE_RESOLUTION"
    THINK = "THINK"
    PLAN = "PLAN"
    ACT = "ACT"
    REFLECT = "REFLECT"
    RUN_END = "RUN_END"


@dataclass
class ProvenanceEvent:
    """Single provenance event"""
    timestamp: str
    event_type: str
    run_id: str
    seed: int
    data: Dict[str, Any]
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ProvenanceLogger:
    """
    JSONL-based provenance logger for AIPOM-CoT runs.

    Creates trace files in outputs/provenance/ directory.
    Each run gets its own trace file: {run_id}_trace.jsonl
    """

    def __init__(self, output_dir: str = "./outputs/provenance", run_id: Optional[str] = None, seed: int = 42):
        """
        Initialize provenance logger.

        Args:
            output_dir: Directory for trace files
            run_id: Unique identifier for this run (auto-generated if None)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or self._generate_run_id()
        self.seed = seed
        self.sequence_number = 0
        self.events: List[ProvenanceEvent] = []

        # Create trace file path
        self.trace_file = self.output_dir / f"{self.run_id}_trace.jsonl"

    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp and random suffix"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{timestamp}_{suffix}"

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp"""
        return datetime.now().isoformat()

    def log_event(self, event_type: str, data: Dict[str, Any]) -> ProvenanceEvent:
        """
        Log a provenance event.

        Args:
            event_type: One of EventType constants
            data: Event-specific data dictionary

        Returns:
            The created ProvenanceEvent
        """
        self.sequence_number += 1

        event = ProvenanceEvent(
            timestamp=self._get_timestamp(),
            event_type=event_type,
            run_id=self.run_id,
            seed=self.seed,
            data=data,
            sequence_number=self.sequence_number
        )

        self.events.append(event)

        # Append to trace file immediately (for crash recovery)
        with open(self.trace_file, 'a', encoding='utf-8') as f:
            f.write(event.to_json() + '\n')

        return event

    def log_run_start(self, mode: str, intent: str, snapshot_id: Optional[str] = None,
                      query: Optional[str] = None, budget: str = 'light') -> ProvenanceEvent:
        """Log run start event"""
        return self.log_event(EventType.RUN_START, {
            'mode': mode,
            'intent': intent,
            'snapshot_id': snapshot_id,
            'query': query,
            'budget': budget,
            'seed': self.seed
        })

    def log_entity_resolution(self, candidates: List[Dict], selected: List[Dict],
                              confidence: float) -> ProvenanceEvent:
        """Log entity resolution event"""
        return self.log_event(EventType.ENTITY_RESOLUTION, {
            'candidates': candidates,
            'selected': selected,
            'confidence': confidence,
            'num_candidates': len(candidates),
            'num_selected': len(selected)
        })

    def log_attribute_resolution(self, requested_fields: List[str],
                                 matched_fields: List[str],
                                 missing_fields: List[str]) -> ProvenanceEvent:
        """Log attribute resolution event"""
        return self.log_event(EventType.ATTRIBUTE_RESOLUTION, {
            'requested_fields': requested_fields,
            'matched_fields': matched_fields,
            'missing_fields': missing_fields,
            'match_rate': len(matched_fields) / max(len(requested_fields), 1)
        })

    def log_think(self, thought: str, context: Optional[Dict] = None) -> ProvenanceEvent:
        """Log thinking/reasoning event"""
        return self.log_event(EventType.THINK, {
            'thought': thought,
            'context': context or {}
        })

    def log_plan(self, plan_steps: List[Dict], planner_type: str = 'adaptive') -> ProvenanceEvent:
        """Log plan generation event"""
        return self.log_event(EventType.PLAN, {
            'plan_steps': plan_steps,
            'planner_type': planner_type,
            'num_steps': len(plan_steps)
        })

    def log_act(self, step_number: int, action_type: str, purpose: str,
                query: Optional[str] = None, params: Optional[Dict] = None,
                result_summary: Optional[Dict] = None) -> ProvenanceEvent:
        """Log action execution event"""
        # Create query hash for provenance pointer
        query_hash = None
        if query:
            query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

        return self.log_event(EventType.ACT, {
            'step_number': step_number,
            'action_type': action_type,
            'purpose': purpose,
            'query': query,
            'query_hash': query_hash,
            'params': params or {},
            'result_summary': result_summary or {}
        })

    def log_reflect(self, step_number: int, validation_status: str,
                    confidence: float, should_replan: bool,
                    recommendations: List[str] = None) -> ProvenanceEvent:
        """Log reflection event"""
        return self.log_event(EventType.REFLECT, {
            'step_number': step_number,
            'validation_status': validation_status,
            'confidence': confidence,
            'should_replan': should_replan,
            'recommendations': recommendations or []
        })

    def log_run_end(self, termination_reason: str, total_steps: int,
                    total_kg_queries: int, execution_time: float,
                    success: bool = True) -> ProvenanceEvent:
        """Log run end event"""
        return self.log_event(EventType.RUN_END, {
            'termination_reason': termination_reason,
            'total_steps': total_steps,
            'total_kg_queries': total_kg_queries,
            'execution_time': execution_time,
            'success': success
        })

    def get_trace_path(self) -> str:
        """Get the path to the trace file"""
        return str(self.trace_file)

    def get_events(self) -> List[ProvenanceEvent]:
        """Get all logged events"""
        return self.events.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the run"""
        act_events = [e for e in self.events if e.event_type == EventType.ACT]
        reflect_events = [e for e in self.events if e.event_type == EventType.REFLECT]

        return {
            'run_id': self.run_id,
            'seed': self.seed,
            'trace_file': str(self.trace_file),
            'total_events': len(self.events),
            'total_actions': len(act_events),
            'total_reflections': len(reflect_events),
            'event_types': list(set(e.event_type for e in self.events))
        }

    @staticmethod
    def load_trace(trace_path: str) -> List[ProvenanceEvent]:
        """
        Load events from a trace file.

        Args:
            trace_path: Path to the JSONL trace file

        Returns:
            List of ProvenanceEvent objects
        """
        events = []
        with open(trace_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(ProvenanceEvent(**data))
        return events


# Convenience function
def create_provenance_logger(run_id: str, seed: int = 42,
                             output_dir: str = "./outputs/provenance") -> ProvenanceLogger:
    """
    Create a provenance logger with the given run_id.

    Args:
        run_id: Unique identifier (e.g., "ms_result2", "ms_result4")
        seed: Random seed
        output_dir: Output directory

    Returns:
        Configured ProvenanceLogger
    """
    return ProvenanceLogger(output_dir=output_dir, run_id=run_id, seed=seed)


# Self-test
if __name__ == "__main__":
    import tempfile

    print("Provenance Logger Self-Test")
    print("=" * 50)

    # Create temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ProvenanceLogger(output_dir=tmpdir, run_id="test_run", seed=42)

        # Log events
        logger.log_run_start(mode='ms', intent='MS_REPRO', query='Test query')
        logger.log_entity_resolution(
            candidates=[{'text': 'Car3', 'type': 'Gene'}],
            selected=[{'text': 'Car3', 'type': 'Gene'}],
            confidence=0.95
        )
        logger.log_plan(
            plan_steps=[{'step': 1, 'purpose': 'Find neurons'}],
            planner_type='adaptive'
        )
        logger.log_act(
            step_number=1,
            action_type='kg_query',
            purpose='Find Car3+ neurons',
            query='MATCH (n:Neuron) RETURN n LIMIT 10'
        )
        logger.log_reflect(
            step_number=1,
            validation_status='passed',
            confidence=0.9,
            should_replan=False
        )
        logger.log_run_end(
            termination_reason='completed',
            total_steps=1,
            total_kg_queries=1,
            execution_time=1.5
        )

        # Verify
        print(f"Trace file: {logger.get_trace_path()}")
        print(f"Summary: {logger.get_summary()}")

        # Load and verify
        loaded = ProvenanceLogger.load_trace(logger.get_trace_path())
        print(f"Loaded {len(loaded)} events")

        print("=" * 50)
        print("Self-test passed!")
