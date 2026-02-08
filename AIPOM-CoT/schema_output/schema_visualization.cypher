// NeuroXiv 2.0 Schema 可视化查询
// 在Neo4j Browser中运行以下查询

// 1. 显示每种节点类型的样例
MATCH (n:Class) RETURN n LIMIT 5;
MATCH (n:Cluster) RETURN n LIMIT 5;
MATCH (n:ME_Subregion) RETURN n LIMIT 5;
MATCH (n:Neuron) RETURN n LIMIT 5;
MATCH (n:Region) RETURN n LIMIT 5;
MATCH (n:Subclass) RETURN n LIMIT 5;
MATCH (n:Subregion) RETURN n LIMIT 5;
MATCH (n:Supertype) RETURN n LIMIT 5;

// 2. 显示所有关系模式
MATCH (a:Neuron)-[r:AXON_NEIGHBOURING]->(b:Neuron) RETURN a, r, b LIMIT 10;
MATCH (a:Cluster)-[r:BELONGS_TO]->(b:Supertype) RETURN a, r, b LIMIT 10;
MATCH (a:Neuron)-[r:DEN_NEIGHBOURING]->(b:Neuron) RETURN a, r, b LIMIT 10;
MATCH (a:Subregion)-[r:HAS_CLASS]->(b:Class) RETURN a, r, b LIMIT 10;
MATCH (a:Subregion)-[r:HAS_CLUSTER]->(b:Cluster) RETURN a, r, b LIMIT 10;
MATCH (a:Subregion)-[r:HAS_SUBCLASS]->(b:Subclass) RETURN a, r, b LIMIT 10;
MATCH (a:Subregion)-[r:HAS_SUPERTYPE]->(b:Supertype) RETURN a, r, b LIMIT 10;
MATCH (a:Neuron)-[r:LOCATE_AT]->(b:Region) RETURN a, r, b LIMIT 10;
MATCH (a:Neuron)-[r:LOCATE_AT_ME_SUBREGION]->(b:ME_Subregion) RETURN a, r, b LIMIT 10;
MATCH (a:Neuron)-[r:LOCATE_AT_SUBREGION]->(b:Subregion) RETURN a, r, b LIMIT 10;
MATCH (a:Neuron)-[r:NEIGHBOURING]->(b:Neuron) RETURN a, r, b LIMIT 10;
MATCH (a:Neuron)-[r:PROJECT_TO]->(b:Region) RETURN a, r, b LIMIT 10;

// 3. 统计查询
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (:`'+label+'`) RETURN count(*) as count', {}) YIELD value
RETURN label, value.count ORDER BY value.count DESC;