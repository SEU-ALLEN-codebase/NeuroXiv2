import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import statistics

logger = logging.getLogger(__name__)


# ==================== Reflection Structures ====================

class ValidationStatus(Enum):
    """验证状态"""
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY_RESULT = "empty_result"
    UNEXPECTED = "unexpected"


@dataclass
class UncertaintyMetrics:
    """不确定性度量"""
    data_completeness: float  # 0-1: 数据完整性
    result_consistency: float  # 0-1: 结果一致性
    expectation_match: float  # 0-1: 与预期的匹配度
    overall_uncertainty: float  # 0-1: 综合不确定性


@dataclass
class AlternativeHypothesis:
    """替代假设"""
    hypothesis: str
    reasoning: str
    feasibility: float  # 0-1
    required_data: List[str]


@dataclass
class EvidenceQuality:
    """证据质量评估"""
    source_reliability: float  # 数据源可靠性
    sample_size_adequacy: float  # 样本量充分性
    measurement_precision: float  # 测量精度
    overall_quality: float  # 综合质量


@dataclass
class StructuredReflection:
    """结构化反思结果"""
    step_number: int

    # 1. Validation
    validation_status: ValidationStatus
    validation_details: Dict[str, Any]

    # 2. Uncertainty
    uncertainty: UncertaintyMetrics

    # 3. Alternatives
    alternative_hypotheses: List[AlternativeHypothesis]

    # 4. Evidence
    evidence_quality: EvidenceQuality

    # 5. Confidence
    confidence_score: float
    confidence_factors: Dict[str, float]

    # 6. Recommendations
    next_step_recommendations: List[str]
    should_replan: bool

    # Summary
    summary: str


# ==================== Structured Reflector ====================

class StructuredReflector:
    """
    结构化反思引擎

    替代简单的LLM文本生成,使用规则+统计方法
    """

    def __init__(self):
        pass

    def reflect(self,
                step_number: int,
                purpose: str,
                expected_result: str,
                actual_result: Dict[str, Any],
                question_context: str) -> StructuredReflection:
        """
        完整的结构化反思

        Args:
            step_number: 步骤编号
            purpose: 步骤目的
            expected_result: 预期结果
            actual_result: 实际执行结果
            question_context: 原始问题上下文

        Returns:
            StructuredReflection对象
        """

        # 1. Validation
        validation_status, validation_details = self._validate_result(
            expected_result, actual_result
        )

        # 2. Uncertainty quantification
        uncertainty = self._quantify_uncertainty(
            actual_result, expected_result, validation_status
        )

        # 3. Alternative hypotheses
        alternatives = self._generate_alternatives(
            purpose, actual_result, validation_status
        )

        # 4. Evidence quality
        evidence = self._assess_evidence_quality(actual_result)

        # 5. Confidence scoring
        confidence, factors = self._compute_confidence(
            validation_status, uncertainty, evidence
        )

        # 6. Recommendations
        recommendations, should_replan = self._generate_recommendations(
            validation_status, uncertainty, confidence
        )

        # Summary
        summary = self._generate_summary(
            step_number, purpose, validation_status, confidence
        )

        return StructuredReflection(
            step_number=step_number,
            validation_status=validation_status,
            validation_details=validation_details,
            uncertainty=uncertainty,
            alternative_hypotheses=alternatives,
            evidence_quality=evidence,
            confidence_score=confidence,
            confidence_factors=factors,
            next_step_recommendations=recommendations,
            should_replan=should_replan,
            summary=summary
        )

    # ==================== Validation ====================

    def _validate_result(self,
                         expected: str,
                         actual: Dict[str, Any]) -> tuple:
        """
        验证实际结果与预期

        检查:
        - 是否成功执行
        - 是否返回数据
        - 数据量是否合理
        - 数据类型是否符合预期
        """
        details = {}

        # Check execution success
        if not actual.get('success', False):
            return ValidationStatus.FAILED, {
                'reason': 'Query execution failed',
                'error': actual.get('error', 'Unknown error')
            }

        # Check data availability
        data = actual.get('data', [])
        details['row_count'] = len(data)

        if len(data) == 0:
            return ValidationStatus.EMPTY_RESULT, {
                'reason': 'No data returned',
                'row_count': 0
            }

        # Analyze expectation keywords
        expected_lower = expected.lower()

        # Expected patterns
        patterns = {
            'should_have_multiple': r'(multiple|several|many|list)',
            'should_have_few': r'(few|some|several)',
            'should_have_single': r'(single|one|specific)',
            'should_have_counts': r'(count|number|quantity)',
            'should_have_names': r'(name|label|identifier)'
        }

        detected_patterns = []
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, expected_lower):
                detected_patterns.append(pattern_name)

        details['detected_patterns'] = detected_patterns

        # Check if data matches expectations
        match_score = 0.0

        if 'should_have_multiple' in detected_patterns:
            if len(data) >= 5:
                match_score += 0.5

        if 'should_have_counts' in detected_patterns:
            # Check if data has count/number fields
            if data and any(k for k in data[0].keys() if 'count' in k.lower() or 'number' in k.lower()):
                match_score += 0.3

        if 'should_have_names' in detected_patterns:
            if data and any(k for k in data[0].keys() if 'name' in k.lower()):
                match_score += 0.2

        details['expectation_match_score'] = match_score

        # Determine status
        if match_score >= 0.7:
            status = ValidationStatus.PASSED
        elif match_score >= 0.3:
            status = ValidationStatus.PARTIAL
        elif len(data) > 0:
            status = ValidationStatus.UNEXPECTED
        else:
            status = ValidationStatus.EMPTY_RESULT

        return status, details

    # ==================== Uncertainty Quantification ====================

    def _quantify_uncertainty(self,
                              actual: Dict[str, Any],
                              expected: str,
                              status: ValidationStatus) -> UncertaintyMetrics:
        """
        量化不确定性

        考虑因素:
        - 数据完整性 (是否有NULL值)
        - 结果一致性 (分布是否合理)
        - 预期匹配度
        """
        data = actual.get('data', [])

        # Data completeness
        if not data:
            completeness = 0.0
        else:
            # Count non-null values
            total_values = sum(len(row) for row in data)
            non_null_values = sum(
                1 for row in data for v in row.values() if v is not None
            )
            completeness = non_null_values / total_values if total_values > 0 else 0.0

        # Result consistency (check variance if numeric)
        consistency = 0.8  # Default

        if data and len(data) > 2:
            # Check numeric fields
            numeric_fields = {}
            for row in data:
                for k, v in row.items():
                    if isinstance(v, (int, float)) and v is not None:
                        numeric_fields.setdefault(k, []).append(v)

            # Calculate coefficient of variation
            if numeric_fields:
                cvs = []
                for field, values in numeric_fields.items():
                    if len(values) > 1:
                        mean_val = statistics.mean(values)
                        if mean_val > 0:
                            std_val = statistics.stdev(values)
                            cv = std_val / mean_val
                            cvs.append(cv)

                if cvs:
                    avg_cv = statistics.mean(cvs)
                    # Lower CV = higher consistency
                    consistency = max(0.0, 1.0 - min(1.0, avg_cv))

        # Expectation match
        if status == ValidationStatus.PASSED:
            expectation_match = 0.9
        elif status == ValidationStatus.PARTIAL:
            expectation_match = 0.6
        elif status == ValidationStatus.UNEXPECTED:
            expectation_match = 0.3
        else:
            expectation_match = 0.0

        # Overall uncertainty (inverse of certainty)
        certainty = (completeness + consistency + expectation_match) / 3
        overall_uncertainty = 1.0 - certainty

        return UncertaintyMetrics(
            data_completeness=completeness,
            result_consistency=consistency,
            expectation_match=expectation_match,
            overall_uncertainty=overall_uncertainty
        )

    # ==================== Alternative Hypotheses ====================

    def _generate_alternatives(self,
                               purpose: str,
                               actual: Dict[str, Any],
                               status: ValidationStatus) -> List[AlternativeHypothesis]:
        """
        生成替代假设

        当结果不符合预期时,提出可能的原因和替代方案
        """
        alternatives = []

        data = actual.get('data', [])

        # If empty result
        if status == ValidationStatus.EMPTY_RESULT:
            alternatives.append(AlternativeHypothesis(
                hypothesis="The queried entity does not exist in the knowledge graph",
                reasoning="Zero results typically indicate missing data or incorrect entity names",
                feasibility=0.7,
                required_data=["Entity existence check", "Alternative entity names"]
            ))

            alternatives.append(AlternativeHypothesis(
                hypothesis="The relationship path does not exist",
                reasoning="Query may be traversing non-existent relationships",
                feasibility=0.6,
                required_data=["Schema path validation", "Alternative paths"]
            ))

            alternatives.append(AlternativeHypothesis(
                hypothesis="Query syntax error or filter too strict",
                reasoning="Query constraints may be overly restrictive",
                feasibility=0.5,
                required_data=["Query relaxation", "Filter analysis"]
            ))

        # If unexpected result
        elif status == ValidationStatus.UNEXPECTED:
            if len(data) > 0:
                alternatives.append(AlternativeHypothesis(
                    hypothesis="Data exists but does not match expected pattern",
                    reasoning="Results returned but structure differs from expectation",
                    feasibility=0.8,
                    required_data=["Pattern analysis", "Schema verification"]
                ))

        # If partial success
        elif status == ValidationStatus.PARTIAL:
            alternatives.append(AlternativeHypothesis(
                hypothesis="Additional query refinement needed",
                reasoning="Partial match suggests query is on right track but needs adjustment",
                feasibility=0.9,
                required_data=["Query parameter tuning"]
            ))

        return alternatives

    # ==================== Evidence Quality ====================

    def _assess_evidence_quality(self, actual: Dict[str, Any]) -> EvidenceQuality:
        """
        评估证据质量

        考虑:
        - 数据源 (KG是高质量的)
        - 样本量
        - 测量精度
        """
        data = actual.get('data', [])

        # Source reliability (KG is curated)
        source_reliability = 0.9  # High for curated KG

        # Sample size adequacy
        sample_size = len(data)

        if sample_size >= 50:
            sample_adequacy = 1.0
        elif sample_size >= 20:
            sample_adequacy = 0.8
        elif sample_size >= 10:
            sample_adequacy = 0.6
        elif sample_size >= 5:
            sample_adequacy = 0.4
        elif sample_size >= 1:
            sample_adequacy = 0.2
        else:
            sample_adequacy = 0.0

        # Measurement precision (check for numeric precision)
        precision = 0.8  # Default

        if data:
            # Check numeric field precision
            for row in data[:10]:
                for v in row.values():
                    if isinstance(v, float):
                        # Check decimal places
                        str_v = str(v)
                        if '.' in str_v:
                            decimals = len(str_v.split('.')[1])
                            if decimals >= 2:
                                precision = 0.9
                                break

        # Overall quality
        overall = (source_reliability + sample_adequacy + precision) / 3

        return EvidenceQuality(
            source_reliability=source_reliability,
            sample_size_adequacy=sample_adequacy,
            measurement_precision=precision,
            overall_quality=overall
        )

    # ==================== Confidence Scoring ====================

    def _compute_confidence(self,
                            status: ValidationStatus,
                            uncertainty: UncertaintyMetrics,
                            evidence: EvidenceQuality) -> tuple:
        """
        计算置信度分数

        综合考虑:
        - 验证状态
        - 不确定性
        - 证据质量
        """
        factors = {}

        # Factor 1: Validation status
        if status == ValidationStatus.PASSED:
            factors['validation'] = 0.9
        elif status == ValidationStatus.PARTIAL:
            factors['validation'] = 0.6
        elif status == ValidationStatus.UNEXPECTED:
            factors['validation'] = 0.4
        else:
            factors['validation'] = 0.1

        # Factor 2: Certainty (inverse of uncertainty)
        factors['certainty'] = 1.0 - uncertainty.overall_uncertainty

        # Factor 3: Evidence quality
        factors['evidence'] = evidence.overall_quality

        # Weighted average
        weights = {
            'validation': 0.4,
            'certainty': 0.3,
            'evidence': 0.3
        }

        confidence = sum(factors[k] * weights[k] for k in factors)

        return confidence, factors

    # ==================== Recommendations ====================

    def _generate_recommendations(self,
                                  status: ValidationStatus,
                                  uncertainty: UncertaintyMetrics,
                                  confidence: float) -> tuple:
        """
        生成下一步建议

        返回: (recommendations, should_replan)
        """
        recommendations = []
        should_replan = False

        # Based on validation status
        if status == ValidationStatus.EMPTY_RESULT:
            recommendations.append("Consider relaxing query constraints")
            recommendations.append("Verify entity names and relationships exist")
            recommendations.append("Try alternative schema paths")
            should_replan = True

        elif status == ValidationStatus.FAILED:
            recommendations.append("Check query syntax")
            recommendations.append("Verify schema compatibility")
            should_replan = True

        elif status == ValidationStatus.UNEXPECTED:
            recommendations.append("Analyze result structure")
            recommendations.append("Adjust expected result description")
            should_replan = False

        elif status == ValidationStatus.PARTIAL:
            recommendations.append("Refine query parameters")
            recommendations.append("Consider additional filtering")
            should_replan = False

        else:  # PASSED
            recommendations.append("Proceed to next step")
            recommendations.append("Use results for dependent queries")
            should_replan = False

        # Based on uncertainty
        if uncertainty.overall_uncertainty > 0.6:
            recommendations.append("High uncertainty detected - consider validation")
            should_replan = should_replan or (confidence < 0.4)

        # Based on confidence
        if confidence < 0.3:
            recommendations.append("Low confidence - replanning recommended")
            should_replan = True

        return recommendations, should_replan

    # ==================== Summary Generation ====================

    def _generate_summary(self,
                          step_number: int,
                          purpose: str,
                          status: ValidationStatus,
                          confidence: float) -> str:
        """生成简洁的反思摘要"""

        status_desc = {
            ValidationStatus.PASSED: "succeeded as expected",
            ValidationStatus.PARTIAL: "partially succeeded",
            ValidationStatus.UNEXPECTED: "returned unexpected results",
            ValidationStatus.EMPTY_RESULT: "returned no results",
            ValidationStatus.FAILED: "failed to execute"
        }

        desc = status_desc.get(status, "completed")

        conf_level = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"

        return (f"Step {step_number} ({purpose}) {desc}. "
                f"Confidence: {conf_level} ({confidence:.2f})")


# ==================== Test ====================

if __name__ == "__main__":
    reflector = StructuredReflector()

    # Test case 1: Successful query
    print("\n=== Test 1: Successful Query ===")

    reflection = reflector.reflect(
        step_number=1,
        purpose="Find Car3+ clusters",
        expected_result="Should return multiple clusters with Car3 in markers",
        actual_result={
            'success': True,
            'data': [
                {'cluster_name': 'L5 IT CTX', 'markers': 'Car3,Satb2', 'neuron_count': 1234},
                {'cluster_name': 'L6 CT CTX', 'markers': 'Car3,Fezf2', 'neuron_count': 987},
                {'cluster_name': 'L4 IT CTX', 'markers': 'Car3,Rorb', 'neuron_count': 2345}
            ],
            'rows': 3
        },
        question_context="Tell me about Car3+ neurons"
    )

    print(f"Status: {reflection.validation_status.value}")
    print(f"Confidence: {reflection.confidence_score:.3f}")
    print(f"Uncertainty: {reflection.uncertainty.overall_uncertainty:.3f}")
    print(f"Summary: {reflection.summary}")
    print(f"Recommendations: {reflection.next_step_recommendations}")

    # Test case 2: Empty result
    print("\n=== Test 2: Empty Result ===")

    reflection2 = reflector.reflect(
        step_number=2,
        purpose="Find non-existent gene",
        expected_result="Should find regions",
        actual_result={
            'success': True,
            'data': [],
            'rows': 0
        },
        question_context="Find XYZ123 gene"
    )

    print(f"Status: {reflection2.validation_status.value}")
    print(f"Confidence: {reflection2.confidence_score:.3f}")
    print(f"Should replan: {reflection2.should_replan}")
    print(f"Alternatives: {len(reflection2.alternative_hypotheses)}")
    for alt in reflection2.alternative_hypotheses:
        print(f"  - {alt.hypothesis} (feasibility: {alt.feasibility:.2f})")