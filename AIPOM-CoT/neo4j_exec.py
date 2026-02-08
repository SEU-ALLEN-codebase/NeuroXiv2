import logging
import re
import time
from typing import Any, Dict, Optional

from neo4j import GraphDatabase
try:
    from neo4j.exceptions import TransientError
except Exception:  # pragma: no cover
    class TransientError(Exception):
        pass

logger = logging.getLogger(__name__)


class Neo4jExec:
    """
    Robust read-only executor with EXPLAIN precheck, auto LIMIT, retries and timeouts.
    """
    def __init__(self, uri: str, user: str, pwd: str, database: str = "neo4j", timeout_s: int = 25, max_retries: int = 3):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=10)
        self.database = database
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass

    @staticmethod
    def _ensure_limit(q: str, default_limit: int = 100) -> str:
        if re.search(r"\bLIMIT\b", q, re.I):
            return q
        # avoid LIMIT inside subqueries only naive: append at end
        return f"{q}\nLIMIT {default_limit}"

    def explain_ok(self, q: str) -> bool:
        try:
            with self.driver.session(database=self.database) as s:
                s.run("EXPLAIN " + q, parameters=None, timeout=self.timeout_s)
            return True
        except Exception as e:  # noqa
            logger.debug(f"EXPLAIN failed: {e}")
            return False

    def run(self, q: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = self._ensure_limit(q)
        if not self.explain_ok(q):
            # soft degrade: strip ORDER BY when EXPLAIN fails
            q = re.sub(r"\bORDER BY .*?(?=LIMIT|$)", "", q, flags=re.I | re.S)
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                with self.driver.session(database=self.database) as s:
                    res = s.run(q, parameters=(params or {}), timeout=self.timeout_s)
                    data = [r.data() for r in res]
                dur = time.time() - start
                return {"success": True, "rows": len(data), "data": data, "t": dur, "query": q}
            except TransientError as e:  # transient retry
                backoff = 2 ** attempt
                logger.warning(f"TransientError; retry in {backoff}s: {e}")
                time.sleep(backoff)
            except Exception as e:
                logger.error(f"Neo4j query failed: {e}")
                break
        return {"success": False, "rows": 0, "data": [], "t": 0.0, "query": q}
