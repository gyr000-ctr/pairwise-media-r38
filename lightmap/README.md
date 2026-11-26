# lightmap

## 题面（可直接粘贴）

光照图烘焙采样。约束：UV 到光照图双线性采样与解析式一致（≤1e-9）；越界按 clamp；必须实现从世界位置到 UV 的映射并与之互逆（≤1e-6）；接缝处不出现黑边（边界像素与内插值一致）；10 万次采样不超过 300 ms。 工作目录为仓库内的 lightmap/。 验收：cd lightmap && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd lightmap && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
