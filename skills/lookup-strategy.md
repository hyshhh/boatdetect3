# 查找策略 Skill

## 查找优先级
1. **精确查找优先**：有弦号时先调用 `lookup_by_hull_number`，命中即结束
2. **语义检索兜底**：lookup 未命中或无弦号时，用 description 调用 `retrieve_by_description`

## 精确查找规则
- 弦号直接查字典，O(1) 返回描述
- found=true → 匹配类型为 "exact"，返回库内确定信息
- found=false → 进入语义检索

## 语义检索规则
- 基于 FAISS 向量库对船描述做语义相似度匹配
- 返回 top_k 个候选结果（默认 3 个）
- 有结果 → 匹配类型为 "semantic"
- 无结果 → 匹配类型为 "none"

## 禁止
- 不要跳过任何步骤
- 不要同时调用多个工具（严格按顺序执行）
