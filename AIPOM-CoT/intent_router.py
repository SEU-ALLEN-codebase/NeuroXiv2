import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Set


class IntentType(Enum):
    """Intent classification types"""
    SMALLTALK = "smalltalk"
    FACT_LOOKUP = "fact_lookup"
    KG_TRAVERSAL = "kg_traversal"
    ANALYSIS = "analysis"
    MS_REPRO = "ms_repro"
    UNKNOWN = "unknown"


@dataclass
class BudgetLimits:
    """Runtime budget constraints"""
    max_kg_queries: int
    row_limit: int
    max_plan_steps: int


# Budget presets from spec
BUDGET_PRESETS = {
    'light': BudgetLimits(max_kg_queries=2, row_limit=50, max_plan_steps=2),
    'standard': BudgetLimits(max_kg_queries=8, row_limit=500, max_plan_steps=6),
    'heavy': BudgetLimits(max_kg_queries=30, row_limit=5000, max_plan_steps=12),
}

# Intent-specific caps (stricter than budget)
INTENT_CAPS = {
    IntentType.SMALLTALK: BudgetLimits(max_kg_queries=0, row_limit=0, max_plan_steps=0),
    IntentType.FACT_LOOKUP: BudgetLimits(max_kg_queries=2, row_limit=50, max_plan_steps=2),
    IntentType.KG_TRAVERSAL: BudgetLimits(max_kg_queries=3, row_limit=200, max_plan_steps=3),
    IntentType.ANALYSIS: BudgetLimits(max_kg_queries=30, row_limit=5000, max_plan_steps=12),
    IntentType.MS_REPRO: BudgetLimits(max_kg_queries=30, row_limit=5000, max_plan_steps=12),
    IntentType.UNKNOWN: BudgetLimits(max_kg_queries=1, row_limit=20, max_plan_steps=1),
}


class IntentRouter:
    """
    Intent classification and gating layer.

    Classifies user input into intent types and provides appropriate
    budget limits to prevent heavy pipelines for simple queries.
    """

    # Chinese smalltalk patterns
    CN_SMALLTALK = {
        '你好', '您好', 'hi', 'hello', '嗨', '哈喽', '早', '早上好',
        '晚上好', '下午好', '在吗', '在不在', '谢谢', '感谢', '多谢',
        '再见', '拜拜', '好的', '好', 'ok', 'thanks', 'thank you',
        'bye', 'goodbye', 'hey', 'yo', '嗯', '行', '可以'
    }

    # Question keywords indicating non-smalltalk
    CN_QUESTION_KEYWORDS = {'什么', '怎么', '为什么', '哪里', '哪些', '多少', '是否', '如何', '几'}
    EN_QUESTION_KEYWORDS = {'what', 'why', 'how', 'where', 'which', 'when', 'who', 'can', 'could', 'would', 'is', 'are', 'do', 'does'}

    # Phrases that indicate information request (treat as question)
    INFO_REQUEST_PATTERNS = ['tell me about', 'explain', 'describe', 'show me', '告诉我', '介绍', '说说', '讲讲']

    # Gene patterns (common gene naming conventions)
    # Use lookahead/lookbehind to handle Chinese text without word boundaries
    GENE_PATTERNS = [
        r'(?<![a-zA-Z])[A-Z][a-z]{2,}[0-9]*(?![a-zA-Z])',  # Car3, Pvalb, Sst
        r'(?<![a-zA-Z])[A-Z]{2,}[0-9]*[a-z]?(?![a-zA-Z])',  # GAD, SST, VIP
    ]

    # Brain region patterns
    REGION_PATTERNS = [
        r'(?<![a-zA-Z])[A-Z]{2,5}[0-9]?/?[0-9]?(?![a-zA-Z])',  # MOp, VISp, MOp2/3
        r'(?<![a-zA-Z])[A-Z][a-z]+[0-9]*/[0-9]+(?![a-zA-Z])',  # Layer patterns
    ]

    # Analysis trigger keywords
    ANALYSIS_KEYWORDS = {
        'comprehensive', 'analyze', 'analysis', 'compare', 'comparison',
        'fingerprint', 'mismatch', 'similarity', 'enrichment', 'ranking',
        'systematic', 'cross-modal', '分析', '比较', '综合', '特征',
        'neurons', 'neuron', '神经元'  # Gene+neurons often triggers analysis
    }

    # Traversal keywords
    TRAVERSAL_KEYWORDS = {
        'project', 'projection', 'connect', 'relate', 'path', 'neighbor',
        '投射', '连接', '关系', '相关', '路径'
    }

    def __init__(self):
        self._compiled_gene_patterns = [re.compile(p) for p in self.GENE_PATTERNS]
        self._compiled_region_patterns = [re.compile(p) for p in self.REGION_PATTERNS]

    def classify(self, user_input: str) -> IntentType:
        """
        Classify user input into an intent type.

        Args:
            user_input: The user's query string

        Returns:
            IntentType classification
        """
        if not user_input or not user_input.strip():
            return IntentType.UNKNOWN

        text = user_input.strip()
        text_lower = text.lower()

        # Rule 1: Check for SMALLTALK
        if self._is_smalltalk(text, text_lower):
            return IntentType.SMALLTALK

        # Rule 2: Check for explicit MS reproduction keywords
        if self._is_ms_repro(text_lower):
            return IntentType.MS_REPRO

        # Rule 3: Check for analysis intent
        if self._is_analysis(text_lower):
            return IntentType.ANALYSIS

        # Rule 4: Check for traversal/relation queries
        if self._is_traversal(text_lower):
            return IntentType.KG_TRAVERSAL

        # Rule 5: Check for fact lookup (has entity, simple question)
        if self._is_fact_lookup(text, text_lower):
            return IntentType.FACT_LOOKUP

        # Default to UNKNOWN for minimal probe
        return IntentType.UNKNOWN

    def _is_smalltalk(self, text: str, text_lower: str) -> bool:
        """
        Check if input is smalltalk.

        Criteria (from spec section 4.3):
        - Length <= 6 chars (Chinese) OR <= 10 chars (English)
        - AND no entity-like tokens detected
        - AND no question marks / interrogatives
        """
        # Check if text is in known smalltalk set
        if text_lower in self.CN_SMALLTALK or text in self.CN_SMALLTALK:
            return True

        # Length check
        is_short = self._is_short_text(text)
        if not is_short:
            return False

        # Check for question indicators
        if '?' in text or '？' in text:
            return False

        # Check for question keywords
        if self._has_question_keywords(text_lower):
            return False

        # Check for entity-like tokens
        if self._has_entities(text):
            return False

        return True

    def _is_short_text(self, text: str) -> bool:
        """Check if text is short enough for smalltalk consideration"""
        # Count Chinese characters
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # Count English characters (excluding spaces/punctuation)
        en_chars = len(re.findall(r'[a-zA-Z]', text))

        if cn_chars > 0 and cn_chars <= 6:
            return True
        if en_chars > 0 and cn_chars == 0 and len(text) <= 10:
            return True

        return False

    def _has_question_keywords(self, text_lower: str) -> bool:
        """Check for question keywords or info request patterns"""
        for kw in self.CN_QUESTION_KEYWORDS:
            if kw in text_lower:
                return True

        words = set(text_lower.split())
        if words & self.EN_QUESTION_KEYWORDS:
            return True

        # Check for info request patterns
        for pattern in self.INFO_REQUEST_PATTERNS:
            if pattern in text_lower:
                return True

        return False

    def _has_entities(self, text: str) -> bool:
        """Check for gene/region entity patterns"""
        for pattern in self._compiled_gene_patterns:
            if pattern.search(text):
                return True

        for pattern in self._compiled_region_patterns:
            if pattern.search(text):
                return True

        return False

    def _is_ms_repro(self, text_lower: str) -> bool:
        """Check for MS reproduction keywords"""
        ms_keywords = {'result2', 'result4', 'result6', 'ms-case', 'ms case', 'reproduce', 'aipom-cot推理'}
        return any(kw in text_lower for kw in ms_keywords)

    def _is_analysis(self, text_lower: str) -> bool:
        """Check for analysis intent"""
        return any(kw in text_lower for kw in self.ANALYSIS_KEYWORDS)

    def _is_traversal(self, text_lower: str) -> bool:
        """Check for traversal/relation queries"""
        return any(kw in text_lower for kw in self.TRAVERSAL_KEYWORDS)

    def _is_fact_lookup(self, text: str, text_lower: str) -> bool:
        """Check for simple fact lookup"""
        # Has question structure
        has_question = '?' in text or '？' in text or self._has_question_keywords(text_lower)

        # Has entity reference
        has_entity = self._has_entities(text)

        # Not complex (short-ish question with entity)
        is_simple = len(text) < 100

        return has_question and has_entity and is_simple


def get_budget_for_intent(intent: IntentType, budget_level: str = 'light') -> BudgetLimits:
    """
    Get effective budget limits for an intent, respecting both
    the budget level and intent-specific caps.

    Intent caps override budget only to be stricter, never looser.

    Args:
        intent: The classified intent type
        budget_level: One of 'light', 'standard', 'heavy'

    Returns:
        BudgetLimits with effective constraints
    """
    budget = BUDGET_PRESETS.get(budget_level, BUDGET_PRESETS['light'])
    intent_cap = INTENT_CAPS.get(intent, INTENT_CAPS[IntentType.UNKNOWN])

    # Take the minimum (stricter) of each limit
    return BudgetLimits(
        max_kg_queries=min(budget.max_kg_queries, intent_cap.max_kg_queries),
        row_limit=min(budget.row_limit, intent_cap.row_limit),
        max_plan_steps=min(budget.max_plan_steps, intent_cap.max_plan_steps)
    )


def get_smalltalk_response(user_input: str) -> str:
    """
    Generate a polite response for smalltalk without KG queries.

    Args:
        user_input: The smalltalk input

    Returns:
        Polite response string
    """
    text_lower = user_input.lower().strip()

    # Greeting responses
    greetings = {'hi', 'hello', 'hey', 'yo', '你好', '您好', '嗨', '哈喽'}
    if text_lower in greetings:
        return "Hello! I'm AIPOM-CoT, a neuroscience knowledge graph assistant. How can I help you explore the brain atlas today?"

    # Thanks responses
    thanks = {'谢谢', '感谢', '多谢', 'thanks', 'thank you'}
    if text_lower in thanks:
        return "You're welcome! Feel free to ask me about neurons, brain regions, gene markers, or their relationships."

    # Goodbye responses
    goodbyes = {'再见', '拜拜', 'bye', 'goodbye'}
    if text_lower in goodbyes:
        return "Goodbye! Come back anytime to explore more about the brain."

    # Check-in responses
    checkins = {'在吗', '在不在'}
    if text_lower in checkins:
        return "Yes, I'm here! What would you like to know about the brain atlas?"

    # Default friendly response
    return "Hello! I'm ready to help you explore the neuroscience knowledge graph. What would you like to know?"


# Self-test
if __name__ == "__main__":
    router = IntentRouter()

    test_cases = [
        # Smalltalk
        ("你好", IntentType.SMALLTALK),
        ("hi", IntentType.SMALLTALK),
        ("thanks", IntentType.SMALLTALK),
        ("在吗", IntentType.SMALLTALK),

        # Fact lookup
        ("What is HIP?", IntentType.FACT_LOOKUP),
        ("Car3是什么?", IntentType.FACT_LOOKUP),

        # Traversal
        ("MOp2/3 投射到哪些区域", IntentType.KG_TRAVERSAL),
        ("Where does VISp project to?", IntentType.KG_TRAVERSAL),

        # Analysis
        ("Give me a comprehensive analysis of Car3+ neurons", IntentType.ANALYSIS),
        ("Compare molecular and morphological fingerprints", IntentType.ANALYSIS),

        # MS Repro
        ("Run result4 case", IntentType.MS_REPRO),
    ]

    print("Intent Router Self-Test")
    print("=" * 50)

    all_passed = True
    for text, expected in test_cases:
        result = router.classify(text)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"{status}: '{text[:30]}...' -> {result.value} (expected: {expected.value})")

    print("=" * 50)
    print(f"Result: {'All tests passed!' if all_passed else 'Some tests failed.'}")
