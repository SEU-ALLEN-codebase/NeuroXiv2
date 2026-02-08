import re
import time
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging

from neo4j_exec import Neo4jExec
from aipom_cot_true_agent_v2 import RealSchemaCache

logger = logging.getLogger(__name__)


# ==================== Entity Data Structures ====================

@dataclass
class EntityMatch:
    """实体匹配结果"""
    text: str
    entity_type: str
    entity_id: str
    confidence: float
    match_type: str
    span: Tuple[int, int] = (0, 0)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """验证字段"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class EntityCluster:
    """相关实体的聚合"""
    primary_entity: EntityMatch
    related_entities: List[EntityMatch]
    cluster_type: str
    relevance_score: float


# ==================== Fixed Entity Recognizer ====================

class IntelligentEntityRecognizer:
    """
    智能实体识别器（修复版）

    🔧 关键修复：
    1. 超严格的停用词过滤
    2. KG验证层（只返回KG中存在的实体）
    3. 不从答案自动提取
    4. 改进的模糊匹配
    """

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema

        # 🔧 超严格停用词黑名单
        self.STOPWORDS = self._build_comprehensive_stopwords()

        # 缓存
        self._entity_cache = {}
        self._last_cache_time = time.time()
        self._cache_ttl = 3600

    def _build_comprehensive_stopwords(self) -> Set[str]:
        """构建超全面的停用词表"""
        stopwords = set()

        # 疑问词
        stopwords.update(['what', 'which', 'where', 'when', 'who', 'why', 'how'])

        # be动词
        stopwords.update(['are', 'is', 'was', 'were', 'be', 'been', 'being', 'am'])

        # 助动词
        stopwords.update([
            'do', 'does', 'did', 'done', 'doing',
            'have', 'has', 'had', 'having',
            'can', 'could', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must'
        ])

        # 介词
        stopwords.update([
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'into', 'onto', 'upon', 'off', 'out', 'over', 'under',
            'about', 'between', 'within', 'across', 'through'
        ])

        # 连词
        stopwords.update(['and', 'or', 'but', 'so', 'yet', 'nor'])

        # 冠词
        stopwords.update(['the', 'an', 'a'])

        # 代词
        stopwords.update([
            'it', 'its', 'they', 'their', 'them', 'this', 'that', 'these', 'those',
            'he', 'she', 'his', 'her', 'him', 'me', 'my', 'we', 'our', 'us',
            'you', 'your', 'i'
        ])

        # Context-sensitive exclusions: words that ARE valid KG entities but
        # should NOT be matched in common English phrases
        # These are 2-letter region acronyms that conflict with common words
        stopwords.update([
            'me',  # ME (Medial Entorhinal) conflicts with "tell me"
            'or',  # OR conflicts with conjunction "or"
            'an',  # AN conflicts with article "an"
            'as',  # AS conflicts with "as"
            'if',  # IF conflicts with "if"
            'so',  # SO conflicts with "so"
            'no',  # NO conflicts with "no"
        ])

        # 常见动词
        stopwords.update([
            'get', 'got', 'give', 'gave', 'given', 'show', 'tell', 'told',
            'make', 'made', 'take', 'took', 'taken', 'come', 'came',
            'find', 'found', 'see', 'saw', 'seen'
        ])

        # 常见形容词/副词
        stopwords.update([
            'not', 'all', 'some', 'any', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'such', 'no', 'nor', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'now', 'then', 'also', 'here', 'there',
            'well', 'even', 'still', 'already', 'yet'
        ])

        # 神经科学通用词（不是实体）
        stopwords.update([
            'cells', 'neurons', 'brain', 'regions', 'region', 'area', 'areas',
            'types', 'type', 'kind', 'kinds', 'group', 'groups',
            'part', 'parts', 'system', 'systems'
        ])

        return stopwords

    def recognize_entities(self, question: str) -> List[EntityMatch]:
        """
        智能实体识别（修复版）

        🔧 修复策略：
        1. 精确匹配 + KG验证
        2. 模糊匹配 + KG验证
        3. 正则Fallback + KG验证
        4. 超严格停用词过滤
        """
        logger.info(f"🔍 Recognizing entities in: {question}")

        matches = []

        # Step 1: 精确匹配
        exact_matches = self._exact_match_with_validation(question)
        matches.extend(exact_matches)

        if exact_matches:
            logger.info(f"   ✓ Exact match: {len(exact_matches)} entities")

        # Step 2: 模糊匹配（如果精确匹配太少）
        if len(matches) < 2:
            fuzzy_matches = self._fuzzy_match_with_validation(question)

            existing_texts = set([m.text.lower() for m in matches])
            for fm in fuzzy_matches:
                if fm.text.lower() not in existing_texts:
                    matches.append(fm)

            if fuzzy_matches:
                logger.info(f"   ✓ Fuzzy match: {len(fuzzy_matches)} entities")

        # Step 3: 正则Fallback（如果还是没有）
        if not matches:
            logger.warning("   ⚠️ Using regex fallback...")
            regex_matches = self._regex_fallback_with_validation(question)
            matches.extend(regex_matches)

            if regex_matches:
                logger.info(f"   ✓ Regex fallback: {len(regex_matches)} entities")
            else:
                logger.warning(f"   ⚠️ No entities found")

        # 去重和排序
        matches = self._deduplicate_and_rank(matches)

        if matches:
            logger.info(f"   📊 Final: {len(matches)} entities")
            for m in matches[:5]:
                logger.info(f"      • {m.text} ({m.entity_type}) [{m.confidence:.2f}]")

        return matches

    def _exact_match_with_validation(self, question: str) -> List[EntityMatch]:
        """
        精确匹配 + KG验证（修复版）

        🔧 关键修复：
        1. ✅ 先KG验证，再决定是否过滤
        2. ✅ KG中存在的实体，即使是停用词也保留
        3. ✅ KG中不存在的，自动过滤
        """
        matches = []

        entity_types = ['Region', 'GeneMarker']

        for entity_type in entity_types:
            entities = self._get_entities_of_type(entity_type)

            for entity in entities:
                names_to_check = []

                if 'acronym' in entity:
                    names_to_check.append(entity['acronym'])
                if 'name' in entity:
                    names_to_check.append(entity['name'])
                if 'gene' in entity:
                    names_to_check.append(entity['gene'])

                for name in names_to_check:
                    if not name or len(name) < 2:
                        continue

                    # 精确匹配
                    pattern = r'\b' + re.escape(name) + r'\b'

                    for match in re.finditer(pattern, question, re.IGNORECASE):
                        matched_text = match.group()

                        # 🔧 Critical fix: Filter short stopwords even if they exist in KG
                        # This prevents "me" in "tell me" from matching region "ME"
                        if matched_text.lower() in self.STOPWORDS and len(matched_text) <= 3:
                            continue

                        # 🔧 关键：先KG验证
                        validation = self._validate_entity_in_kg(entity_type, name)

                        if validation['exists']:
                            # ✅ KG中存在 → 保留
                            matches.append(EntityMatch(
                                text=matched_text,
                                entity_type=entity_type,
                                entity_id=validation.get('id', name),
                                confidence=1.0,
                                match_type='exact',
                                span=(match.start(), match.end()),
                                metadata=validation.get('data', {})
                            ))
                        # else: KG中不存在 → 自动过滤

        return matches

    def _fuzzy_match_with_validation(self, question: str) -> List[EntityMatch]:
        """
        模糊匹配 + KG验证（修复版）

        🔧 关键修复：
        1. ✅ 保留停用词过滤（防止模糊匹配太多噪音）
        2. ✅ 但在KG验证通过后再决定
        """
        matches = []

        words = re.findall(r'\b[A-Za-z]{2,8}\b', question)

        entity_types = ['Region']

        for entity_type in entity_types:
            entities = self._get_entities_of_type(entity_type)

            for word in words:
                word_lower = word.lower()

                # ✅ 模糊匹配仍然需要停用词过滤
                # 原因：避免 "are" 模糊匹配到 "area"
                if word_lower in self.STOPWORDS:
                    continue

                if len(word) < 3:
                    continue

                for entity in entities:
                    names_to_check = []

                    if 'acronym' in entity:
                        names_to_check.append(entity['acronym'])

                    for name in names_to_check:
                        if not name:
                            continue

                        name_lower = name.lower()

                        if word_lower == name_lower:
                            continue  # 已在exact match处理

                        # 🔧 Fix: Don't match if region name is a strict substring of the query word
                        # This prevents matching "IP" when user asks about "HIP"
                        if name_lower in word_lower and name_lower != word_lower:
                            # The region name is a substring of the query word
                            # Skip this match - the user likely meant the full word
                            continue

                        # 部分匹配 (only allow query word as substring of region name)
                        if word_lower in name_lower:
                            confidence = 0.8
                        else:
                            similarity = self._string_similarity(word_lower, name_lower)
                            if similarity < 0.7:
                                continue
                            confidence = similarity

                        # 🔧 KG验证
                        validation = self._validate_entity_in_kg(entity_type, name)

                        if validation['exists']:
                            span_match = re.search(r'\b' + re.escape(word) + r'\b', question, re.IGNORECASE)
                            if span_match:
                                matches.append(EntityMatch(
                                    text=span_match.group(),
                                    entity_type=entity_type,
                                    entity_id=validation.get('id', name),
                                    confidence=confidence,
                                    match_type='fuzzy',
                                    span=(span_match.start(), span_match.end()),
                                    metadata=validation.get('data', {})
                                ))

        return matches

    def _regex_fallback_with_validation(self, question: str) -> List[EntityMatch]:
        """
        正则Fallback + KG验证（修复版）

        🔧 关键修复：
        1. ✅ 保留停用词过滤（防止WHAT/WHERE等误报）
        2. ✅ 但KG验证是最终决策
        """
        matches = []

        # Pattern 1: 脑区缩写
        region_pattern = r'\b[A-Z]{2,5}\b'

        for match in re.finditer(region_pattern, question):
            text = match.group()

            # ✅ Regex fallback保留停用词过滤
            if text.lower() in self.STOPWORDS:
                continue

            # 🔧 KG验证
            validation = self._validate_entity_in_kg('Region', text)

            if validation['exists']:
                matches.append(EntityMatch(
                    text=text,
                    entity_type='Region',
                    entity_id=validation.get('id', text),
                    confidence=0.6,
                    match_type='regex_fallback',
                    span=(match.start(), match.end()),
                    metadata=validation.get('data', {})
                ))
                logger.info(f"      Regex validated: {text}")

        # Pattern 2: 基因名
        gene_pattern = r'\b[A-Z][a-z]{2,8}\d*\+?\b'

        for match in re.finditer(gene_pattern, question):
            text = match.group()
            gene_name = text.rstrip('+')

            # ✅ 基因也过滤常见单词
            gene_stopwords = [
                'what', 'which', 'where', 'when', 'cells', 'neurons',
                'brain', 'regions', 'does', 'have', 'show', 'tell',
                'about', 'between', 'compare', 'difference'
            ]
            if gene_name.lower() in gene_stopwords:
                continue

            # 🔧 KG验证
            validation = self._validate_entity_in_kg('GeneMarker', gene_name)

            if validation['exists']:
                matches.append(EntityMatch(
                    text=text,
                    entity_type='GeneMarker',
                    entity_id=validation.get('id', gene_name),
                    confidence=0.5,
                    match_type='regex_fallback',
                    span=(match.start(), match.end()),
                    metadata=validation.get('data', {})
                ))
                logger.info(f"      Regex validated: {text}")

        return matches

    def _validate_entity_in_kg(self, entity_type: str, entity_name: str) -> Dict:
        """在KG中验证实体是否存在"""

        if entity_type == 'Region':
            query = """
            MATCH (r:Region)
            WHERE r.acronym = $name OR r.name = $name
            RETURN r.acronym AS id, r.name AS name, r AS data
            LIMIT 1
            """
        elif entity_type == 'GeneMarker':
            query = """
            MATCH (c:Cluster)
            WHERE c.markers CONTAINS $name
            RETURN $name AS id, c AS data
            LIMIT 1
            """
        else:
            return {'exists': False}

        result = self.db.run(query, {'name': entity_name})

        if result['success'] and result['data']:
            row = result['data'][0]
            return {
                'exists': True,
                'id': row.get('id', entity_name),
                'data': row.get('data', {})
            }
        else:
            return {'exists': False}

    def _get_entities_of_type(self, entity_type: str) -> List[Dict]:
        """获取指定类型的所有实体"""
        cache_key = f"entities_{entity_type}"

        if cache_key in self._entity_cache:
            cache_time = self._entity_cache[cache_key].get('time', 0)
            if time.time() - cache_time < self._cache_ttl:
                return self._entity_cache[cache_key]['data']

        if entity_type == 'Region':
            query = """
            MATCH (r:Region)
            RETURN r.acronym AS acronym, r.name AS name
            LIMIT 500
            """
        elif entity_type == 'GeneMarker':
            query = """
            MATCH (c:Cluster)
            WHERE c.markers IS NOT NULL
            WITH split(c.markers, ',') AS marker_list
            UNWIND marker_list AS marker
            RETURN DISTINCT trim(marker) AS gene
            LIMIT 1000
            """
        else:
            return []

        result = self.db.run(query)

        if result['success'] and result['data']:
            entities = result['data']

            self._entity_cache[cache_key] = {
                'data': entities,
                'time': time.time()
            }

            return entities
        else:
            return []

    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if s1 == s2:
            return 1.0

        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Longest common substring ratio
        max_len = max(len(s1), len(s2))

        lcs_len = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                k = 0
                while (i + k < len(s1) and
                       j + k < len(s2) and
                       s1[i + k] == s2[j + k]):
                    k += 1
                lcs_len = max(lcs_len, k)

        return lcs_len / max_len

    def _deduplicate_and_rank(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """去重和排序"""
        seen = {}
        for match in matches:
            key = (match.text.lower(), match.entity_type)

            if key not in seen:
                seen[key] = match
            else:
                if match.confidence > seen[key].confidence:
                    seen[key] = match

        unique_matches = list(seen.values())
        unique_matches.sort(key=lambda m: m.confidence, reverse=True)

        return unique_matches


# ==================== Entity Clustering ====================

class EntityClusteringEngine:
    """实体聚类引擎"""

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema

    def cluster_entities(self,
                         matches: List[EntityMatch],
                         question: str) -> List[EntityCluster]:
        """聚类实体"""
        clusters = []

        # 按类型分组
        genes = [m for m in matches if m.entity_type == 'GeneMarker']
        regions = [m for m in matches if m.entity_type == 'Region']

        # 创建clusters
        if genes:
            cluster = self._create_gene_cluster(genes, regions, question)
            if cluster:
                clusters.append(cluster)

        if regions and not genes:
            cluster = self._create_region_cluster(regions, question)
            if cluster:
                clusters.append(cluster)

        clusters.sort(key=lambda c: c.relevance_score, reverse=True)

        return clusters

    def _create_gene_cluster(self,
                             genes: List[EntityMatch],
                             regions: List[EntityMatch],
                             question: str) -> Optional[EntityCluster]:
        """创建基因cluster"""
        if not genes:
            return None

        primary_gene = genes[0]

        related = list(regions)

        relevance = 0.9
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['gene', 'marker', 'express']):
            relevance *= 1.2

        return EntityCluster(
            primary_entity=primary_gene,
            related_entities=related,
            cluster_type='gene_marker',
            relevance_score=min(1.0, relevance)
        )

    def _create_region_cluster(self,
                               regions: List[EntityMatch],
                               question: str) -> Optional[EntityCluster]:
        """创建region cluster"""
        if not regions:
            return None

        primary_region = regions[0]

        relevance = 0.85
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['region', 'area', 'brain']):
            relevance *= 1.2

        return EntityCluster(
            primary_entity=primary_region,
            related_entities=regions[1:],
            cluster_type='region',
            relevance_score=min(1.0, relevance)
        )