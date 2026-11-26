# skinsparse

## 题面（可直接粘贴）

稀疏蒙皮压缩。约束：每顶点保留权重最大的 k 个骨骼并归一（≤1e-9），被丢弃权重的误差上界可报告（≤1%）；量化到 8 位后反量化误差 ≤1/255；索引表与权重表长度一致；1 万顶点不超过 400 ms；非法 k 可区分报错。 工作目录为仓库内的 skinsparse/。 验收：cd skinsparse && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd skinsparse && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
