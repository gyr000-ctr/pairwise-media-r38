# ikfabrik

## 题面（可直接粘贴）

FABRIK 反向运动学。约束：末端到目标距离 ≤1e-6（可达时）；每段长度严格守恒（≤1e-12）；迭代收敛判据与最大迭代可配；不可达时伸直且方向正确；关节数 ≥2；1 万次求解不超过 500 ms；非法参数可区分报错。 工作目录为仓库内的 ikfabrik/。 验收：cd ikfabrik && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd ikfabrik && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
