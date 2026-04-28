# 模糊弦号处理 Skill

## 触发条件
当 VLM 返回 `clarity="blurry"` 时触发。

## 处理流程
1. VLM 已返回 hull_box 坐标（弦号文字的相对位置）
2. 直接用原始弦号调用 `lookup_by_hull_number`
3. lookup 未命中时，调用 `retrieve_by_description` 语义检索兜底

## 关键原则
- 模糊弦号不做二次识别，直接走查找流程
- hull_box 坐标由 VLM 在 clarity="blurry" 时自动提供，用于 demo 虚线框显示
- 即使弦号模糊，也要尽力完成查找，**不要放弃**
