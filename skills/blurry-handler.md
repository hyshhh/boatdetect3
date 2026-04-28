# 模糊弦号处理 Skill

## 触发条件
当 VLM 返回 `clarity="blurry"` 时触发二次识别流程。

## 处理流程
1. 调用 `re_examine_region`，传入 `hull_box` 的相对坐标（JSON 数组字符串，如 `"[0.1, 0.3, 0.5, 0.6]"`）
2. 二次识别会裁剪放大弦号区域，可能获得更清晰的读数
3. 用二次识别的弦号调用 `lookup_by_hull_number`
4. 如果二次识别仍然模糊，用原始弦号调用 `lookup_by_hull_number`
5. lookup 未命中时，调用 `retrieve_by_description` 语义检索兜底

## 关键原则
- 即使二次识别仍未改善，也要继续执行查找流程，**不要放弃**
- 弦号框坐标由 VLM 在 clarity="blurry" 时自动提供，清晰时无需坐标
- 二次识别使用更高 JPEG 质量（98%）以保留更多文字细节
