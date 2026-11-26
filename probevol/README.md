# probevol

## 题面（可直接粘贴）

光照探针体积（SH-L2 插值）。约束：8 角探针三线性权重和为 1（≤1e-12）且角点精确；每角为 SH-L2（9 系数）时先按系数插值再重建，与逐点暴力重建对照 ≤1e-9；边界 clamp 不得产生能量跳变（相邻格输出差有上界）；10 万次不超过 300 ms；非法分辨率可区分报错。 工作目录为仓库内的 probevol/。 验收：cd probevol && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd probevol && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
