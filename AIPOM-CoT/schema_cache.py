from dataclasses import dataclass, field
from typing import Dict, Any, List, Set
import re

try:
    from neo4j import Session  # type: ignore
except Exception:  # pragma: no cover
    Session = Any  # fallback for type hints


def _canon(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())


@dataclass
class SchemaCache:
    """
    In-memory schema snapshot for labels, properties and relationship types.
    Populated from Neo4j system procedures (db.schema.*).
    """
    node_props: Dict[str, Dict[str, str]] = field(default_factory=dict)   # label -> {prop: type}
    rel_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)    # rel -> {start: List[str], end: List[str], props: {prop: type}}
    synonyms: Dict[str, str] = field(default_factory=dict)                # cn/en -> canonical prop name

    def map_prop(self, text: str) -> str:
        """Map natural words to canonical property names via synonyms (lowercase, underscore)."""
        key = _canon(text)
        return self.synonyms.get(key, key)

    def has_label(self, label: str) -> bool:
        return label in self.node_props

    def has_prop(self, label: str, prop: str) -> bool:
        return self.has_label(label) and prop in self.node_props[label]

    def load_from_db(self, session: "Session"):
        """
        Populate schema cache using Neo4j procedures.
        Handles version differences by using `YIELD *` and probing keys dynamically.
        Fallback path (if procedures unavailable) scans sample data.
        """
        def _rk(rec, key, default=None):
            # robust get for neo4j.Record
            try:
                if hasattr(rec, "get"):
                    v = rec.get(key, default)
                else:
                    v = rec[key] if key in rec.keys() else default
            except Exception:
                try:
                    v = rec[key]
                except Exception:
                    v = default
            return v

        # -------- Nodes --------
        try:
            q1 = "CALL db.schema.nodeTypeProperties() YIELD * RETURN *"
            for rec in session.run(q1):
                # Possible variants: nodeLabels (List), nodeLabel (String), nodeType (String)
                node_labels = _rk(rec, "nodeLabels")
                if not node_labels:
                    nl = _rk(rec, "nodeLabel") or _rk(rec, "nodeType")
                    node_labels = [nl] if nl else []
                prop = _rk(rec, "propertyName")
                ptypes = _rk(rec, "propertyTypes") or _rk(rec, "propertyType")
                ptype = (ptypes or ["ANY"])
                ptype = ptype[0] if isinstance(ptype, list) else ptype
                for lab in (node_labels or []):
                    if lab:
                        self.node_props.setdefault(lab, {})[prop] = ptype
        except Exception:
            # Fallback: sample nodes and infer props
            qf = "MATCH (n) WITH labels(n) AS lbs, keys(n) AS ks UNWIND lbs AS l UNWIND ks AS k RETURN l AS label, collect(DISTINCT k)[0..200] AS ks LIMIT 200"
            for rec in session.run(qf):
                lab = _rk(rec, "label")
                ks = _rk(rec, "ks") or []
                self.node_props.setdefault(lab, {})
                for k in ks:
                    self.node_props[lab][k] = "ANY"

        # -------- Relationships --------
        tmp = {}
        try:
            q2 = "CALL db.schema.relTypeProperties() YIELD * RETURN *"
            for rec in session.run(q2):
                rt = _rk(rec, "relType") or _rk(rec, "relationshipType") or _rk(rec, "relationship")
                if not rt:
                    continue
                tmp.setdefault(rt, {"start": set(), "end": set(), "props": {}})
                src = _rk(rec, "sourceNodeLabels") or _rk(rec, "sourceNodeLabel")
                tgt = _rk(rec, "targetNodeLabels") or _rk(rec, "targetNodeLabel")
                if isinstance(src, list):
                    for s in src: tmp[rt]["start"].add(s)
                elif src:
                    tmp[rt]["start"].add(src)
                if isinstance(tgt, list):
                    for t in tgt: tmp[rt]["end"].add(t)
                elif tgt:
                    tmp[rt]["end"].add(tgt)
                pname = _rk(rec, "propertyName")
                ptypes = _rk(rec, "propertyTypes") or _rk(rec, "propertyType")
                ptype = (ptypes or ["ANY"])
                ptype = ptype[0] if isinstance(ptype, list) else ptype
                if pname:
                    tmp[rt]["props"][pname] = ptype
        except Exception:
            # Fallback: sample edges
            qf = """
            MATCH (a)-[r]->(b)
            WITH type(r) AS rt, labels(a) AS la, labels(b) AS lb, keys(r) AS kr
            RETURN rt, la[0..3] AS start, lb[0..3] AS end, kr[0..50] AS props
            LIMIT 500
            """
            for rec in session.run(qf):
                rt = _rk(rec, "rt")
                tmp.setdefault(rt, {"start": set(), "end": set(), "props": {}})
                for s in (_rk(rec, "start") or []):
                    tmp[rt]["start"].add(s)
                for t in (_rk(rec, "end") or []):
                    tmp[rt]["end"].add(t)
                for p in (_rk(rec, "props") or []):
                    tmp[rt]["props"][p] = "ANY"

        self.rel_types = {k: {"start": sorted(list(v["start"])), "end": sorted(list(v["end"])), "props": v["props"]}
                          for k, v in tmp.items()}

        # Initialize synonyms from common properties used in your KG builder
        morph = [
            "axonal_bifurcation_remote_angle", "axonal_branches", "axonal_length", "axonal_maximum_branch_order",
            "dendritic_bifurcation_remote_angle", "dendritic_branches", "dendritic_length", "dendritic_maximum_branch_order"
        ]
        stat = [
            "number_of_apical_dendritic_morphologies", "number_of_axonal_morphologies",
            "number_of_dendritic_morphologies", "number_of_neuron_morphologies", "number_of_transcriptomic_neurons"
        ]
        base_props = morph + stat + ["region_id", "acronym", "name", "full_name"]
        for k in base_props:
            self.synonyms[_canon(k)] = k
        # Chinese mappings
        self.synonyms.update({
            _canon("轴突长度"): "axonal_length",
            _canon("树突长度"): "dendritic_length",
            _canon("轴突分支数"): "axonal_branches",
            _canon("树突分支数"): "dendritic_branches",
            _canon("分叉阶"): "dendritic_maximum_branch_order",
            _canon("转录组数量"): "number_of_transcriptomic_neurons",
        })

    def summary_text(self, max_props: int = 16) -> str:
        """Produce a short, token-friendly schema summary for prompts."""
        lines = []
        lines.append(f"Labels: {len(self.node_props)}; Rels: {len(self.rel_types)}")
        for lab, props in sorted(self.node_props.items()):
            keys = list(props.keys())[:max_props]
            lines.append(f"- {lab} props: {', '.join(keys)}")
        for rt, spec in sorted(self.rel_types.items()):
            s = "/".join(spec["start"]) or "*"
            e = "/".join(spec["end"]) or "*"
            pr = ", ".join(list(spec["props"].keys())[:max_props])
            lines.append(f"- {rt}: {s} -> {e}; props: {pr}")
        return "\n".join(lines)
