# cloth

## 题面（可直接粘贴）

质点弹簧布料。约束：Verlet 积分固定步长且与解析单摆近似对照（≤5%）；结构/剪切/弯曲三类弹簧权重可配；约束迭代后伸长度 ≤1% 初始边长；碰撞球体穿透 ≤1e-9；1000 质点 1000 步不超过 900 ms；非法刚度可区分报错。 工作目录为仓库内的 cloth/。 验收：cd cloth && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd cloth && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
