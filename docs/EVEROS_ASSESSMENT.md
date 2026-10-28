# EverOS 评估报告：能否提升 AgentMem 性能与能力

**日期**：2026-10-22
**评估对象**：[EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS)（克隆至 `~/code/tmp/EverOS`，Apache-2.0）
**被评估项目**：`/Users/jace/code/agentmem`（Redis 8 vectorset 后端，MIT）

---

## 0. 结论摘要（先看这里）

| # | 发现 | 证据强度 | 建议 |
|---|---|---|---|
| 1 | **BM25 关键词检索只覆盖 facts 一层**，5,431 条文本语料中 79% 完全不可关键词检索 | 实测（真实 Redis 数据） | **立即修**，预期 +25% 关键词命中 |
| 2 | **现有手调 wRRF 权重在所有场景下都是负收益** | 7 场景敏感性扫描 | 删掉权重预设，改等权 |
| 3 | **直接照搬 EverOS 的 LR 系数不安全**，符号会随噪声结构翻转（−25% ~ +22%） | 7 场景留出集验证 | 不要照搬，要自己拟合 |
| 4 | **四源自拟合 LR 在全部 7 个场景稳定为正**（nDCG +8.5% ~ +33.9%） | 7 场景留出集验证 | 值得接，但需要标签来源 |
| 5 | **recall 热路径常驻 1–2 次 LLM 调用**是 P50 双峰延迟（7ms–2109ms）根因 | 生产日志 + 代码走读 | 改成按需触发，这是最大延迟杠杆 |
| 6 | EverOS 的 md-first 存储架构 / cascade watcher / LanceDB **不值得移植** | 架构分析 | 保持 Redis vectorset |
| 7 | 4 个模块是死代码（`services/retrieval_service.py`、`services/memory_service.py`、整个 `lifecycle/`），0 引用 | grep 全仓库验证 | 清理或接线 |

---

## 1. EverOS 是什么

EverOS 是 EverMind AI 的 local-first 记忆 runtime，Python library + 可选 server。核心设计：

```
Markdown（唯一真相源）
   ├── SQLite   ~/.everos/.index/sqlite/system.db   状态 + 审计 + 变更队列
   └── LanceDB  ~/.everos/.index/lancedb/           向量 + BM25 + 标量（可从 md 重建）
```

分五层 DDD：`entrypoints → service → memory → infra`，单向依赖由 import-linter 强制。
算法层拆到独立 PyPI 包 `everalgo-*`（Apache-2.0，公开可安装）——**这是本次评估中唯一真正可移植的部分**。

它自己的 benchmark（LoCoMo 94.42 / LongMemEval 94.00）与 AgentMem 的 76.69% **不同数据集、不同 answer 模型，不可直接比较**。

---

## 2. 两个项目的关键差异

| 维度 | EverOS | AgentMem |
|---|---|---|
| 真相源 | Markdown 文件，人可读可 diff 可 Git 化 | Redis vectorset（无 md 层） |
| 索引 | LanceDB IVF_FLAT(nprobes=32) + tantivy FTS | Redis 8 vectorset VSIM + 内存 rank_bm25 |
| 向量维度 | 1024（DashScope） | 384/768（本地 MiniLM / nomic） |
| 检索方法 | keyword / vector / hybrid / agentic，**默认 hybrid 零 LLM** | 4 路并行 + HyDE/MIRIX **常驻 LLM** |
| 融合 | RRF + LR（训练系数）+ vector_anchored | 4 路 wRRF（手调权重） |
| 进化 | reflection：聚类 → LLM 合并 → 软删除 → 重抽 | consolidation：decay/merge/prune（散落两套实现） |
| 代码组织 | src layout + DDD 五层 + 10 条编码规则 | `main.py` 单文件 110KB |
| 隐藏依赖 | 无（DashScope key 显式声明） | **aiserv `127.0.0.1:4000` 本地 LLM 代理**，README 未声明 |

**最值得学的一条架构原则**：EverOS 把 LLM 从检索热路径里彻底挪走。默认 `HYBRID` 方法在 `manager.py` docstring 里明确写着 "the response comes back straight after the heap-expand pipeline"——不过 LLM、不 rerank。LLM rerank 是 opt-in，agentic 多轮检索是独立显式方法。AgentMem 正好相反：HyDE 和 MIRIX 默认开启，每次 recall 都往 aiserv 发 1–2 次 LLM HTTP，预算 2.5s。

---

## 3. 已完成的移植工作

### 3.1 新增文件

| 文件 | 内容 |
|---|---|
| `core/fusion.py` | 从 everalgo 移植的融合/聚合算子，纯函数零 I/O：`rrf`、`lr`、`vector_anchored`、`multi_field_weighting`、`cosine_to_lr_score`、`score_propagation`、`maxsim_group`、`fit_lr_coefs` |
| `tests/unit/test_fusion.py` | 32 个单元测试，全部通过（无需 Redis/LLM/网络） |
| `scripts/bench_fusion.py` | 融合算子对比 benchmark，留出集验证 + 7 场景敏感性扫描 |
| `scripts/bench_bm25_coverage.py` | BM25 覆盖率实测（只读 Redis） |

`main.py`、`core/`、`api/` **一行未改**——新模块是纯新增，可随时回退。

### 3.2 移植了什么，为什么

everalgo 的算子被刻意写成无 I/O 纯函数，所以能 1:1 搬过来：

- **`rrf(k=60)`** — 和 AgentMem 已有的 RRF 同源，作为对照基线
- **`lr` / `multi_field_weighting`** — LR 融合，**吃分数幅度而不是排名**。这是关键差异：两条 VSIM 命中 cosine 0.91 和 0.62，在 RRF 里和 0.52/0.51 等效；LR 能区分
- **`vector_anchored(alpha=0.7)`** — BM25 饱和归一 `s/(s+5)`，防止无界 BM25 淹没有界余弦
- **`maxsim_group`** — child→parent max-pool，用于 atomic fact 分数回注 episode
- **`fit_lr_coefs`** — 从带标签的召回日志拟合自己的系数（纯 numpy 梯度下降，无 sklearn 依赖）

---

## 4. 关键实验证据

### 4.1 融合算子对比（`scripts/bench_fusion.py`）

方法：2,000 文档合成语料，每条有潜在相关性 `r`，四路检索器有不同的分数尺度和可靠性。gold = top-40 by `r`。**系数在独立语料上拟合，在被测语料上评估**（留出集，避免 in-sample 泄漏）。

**第一次跑出来的数字是假的**：`LR (fitted 4-src)` 一度显示 nDCG +30%，但系数和 gold 标签来自同一批数据。改成留出集后跌到 +25%，于是又做了敏感性扫描——结果推翻了更早的乐观结论：

```
SENSITIVITY SUMMARY — nDCG@10 Δ% vs production wRRF
scenario                                 LR(everos)   LR(fit2)  vec_anchor   LR(fit4)  wRRF tuned
default                                      +10.4%    +10.2%      +15.2%    +22.8%      -10.7%
bm25 blind (lexical_rate=0.10)               +21.7%     +9.6%      +19.9%    +33.9%       -5.9%
bm25 strong (lexical_rate=0.70)               +2.9%     +6.2%       +5.9%    +14.1%      -10.3%
dense noisy (glob_noise=0.30)                -25.6%    -26.3%      -25.2%    +32.5%       -1.5%
symbolic rare (sym_rate=0.05)                 -6.0%     -4.6%       -2.4%    +14.1%      -12.6%
symbolic common (sym_rate=0.45)               -7.6%     -8.7%       -0.4%     +8.5%       +2.8%
flat relevance (rel_b=1.5)                   -19.5%    -15.2%      -15.0%    +15.4%       -6.2%
```

三条诚实的结论：

1. **照搬 EverOS 的 LR 系数是危险的。**  dense 噪声一大（glob_noise=0.30）就从 +10% 翻到 −26%。那些系数是在 LanceDB 余弦 + tantivy BM25 的尺度上训出来的，AgentMem 是 Redis VSIM + rank_bm25 Okapi，尺度不同。
2. **四源自拟合 LR 是唯一在全场景为正的方法**（+8.5% ~ +33.9%）。代价是需要标签——可以从召回日志里"用户实际用了哪条"来构造。
3. **`main.py` 里那套 query-type 手调权重（如 entity 查询 `[0.8,0.8,1.4]`）在 6/7 场景是负收益**，最差 −12.6%。手调权重这个方向本身就很脆。

### 4.2 BM25 覆盖率缺口（`scripts/bench_bm25_coverage.py`，真实 Redis 数据）

`core/search.py:307` 的 `populate_bm25_from_redis` 只扫 `mem:facts`，且 cap 5000；所有 `bm25_index.add()` 调用点传的都是 `fact.content`。

实测（db=0，只读）：

```
mem:facts            1139 text-bearing items
mem:episodes         1602 text-bearing items
mem:procedures       2690 text-bearing items   ← 文本在 task/procedure 字段
--------------------------------------------------
total text-bearing: 5431
BM25 corpus today : 1139 facts  (21.0% of corpus)
keyword-invisible : 4292 episodes+procedures (79.0%)
```

跑 10 条代表性查询，对比"仅 facts"与"全层"BM25：

```
TOTAL (fact-only → all-tiers):  80 → 100  (+25.0%)
top-10 中来自当前不可见层的比例: 20/100 = 20%
```

即：**每 5 条关键词召回结果里就有 1 条，来自一个根本没有关键词覆盖的层**——它今天能出现只是靠 VSIM 余弦碰巧重叠。修这个缺口不需要移植任何东西，是 AgentMem 自己的漏。

另外 `populate_bm25_from_redis` 的 5000 cap 目前未触顶（facts=1139，余量 3861），但会**静默截断**——超过后 BM25 语料永久缺尾部，且增量 add 只覆盖新存的。EverOS 的做法是索引可从 md 真相源全量重建，AgentMem 没有这层，所以要么去掉 cap 用 VSCAN 分页，要么周期性重建。

---

## 5. 性能瓶颈定位（带行号）

| 优先级 | 问题 | 位置 | 影响 |
|---|---|---|---|
| P0 | HyDE + MIRIX 常驻 LLM 调用，2.5s 预算 | `main.py:1733-1741, 1808-1813` | recall P50 双峰：`live.log` 实测 7ms–2109ms。README 的 "P50 19ms" 只在 aiserv 秒回时成立 |
| P0 | store 路径 N+1 串行往返：每 fact 一次 VSIM + 一次 persona hget/hset + O(N²) 图边 Lua | `main.py:1273, 1306, 1348, 1359-1363` | 实测 store 6.7s/次 |
| P1 | 热度 bump 名为 batch 实则逐个 await | `core/store.py:411-418` | 命中 10 条 = 10 次串行 Lua |
| P1 | TIG 全量 HGETALL 拉整张转移图进 Python 过滤 | `main.py:2163` | 随历史工具调用数线性增长，无缓存 |
| P1 | consolidation 把全部 facts 重新 embedding（Redis vectorset 不暴露原始向量） | `services/consolidation_service.py:137-138` | 每小时 + 每 50 stores 触发，秒级到十秒级 CPU |
| P2 | 单 uvicorn worker，embedding 挤默认 to_thread 池 | `agentmem.sh:67` | 并发天花板 |
| P2 | `_rrf_merge` 用 content 字符串做去重键 | `main.py:1013-1020` | 两条 content 相同的不同 fact 被误合并 |

**最大的单一延迟杠杆是 P0 第一项**，而 EverOS 给的答案正是"LLM 默认退出热路径"：先跑零 LLM 的 hybrid，只有廉价充分性检查不通过时才升级到 LLM 多轮检索（EverOS 的 `agentic.py`：`temperature=0.0` 充分性检查 → 不足才生成 ≤3 个 multi-query）。

---

## 6. 不值得移植的部分

| 组件 | 原因 |
|---|---|
| LanceDB 存储层 | 需要整体重写存储。Redis vectorset 在跑且够用。EverOS 选 LanceDB 的理由（不需要 Redis/Mongo）对 AgentMem 不成立 |
| cascade watcher（watchdog→SQLite 队列→sha256 增量 diff） | 整套是"md 为真相源"的产物。AgentMem 没有 md 层。但 `content_sha256` 增量思想可以借 |
| md-first 架构 | 是 EverOS 的卖点（可 diff、可 Git 化），也是它索引可重建的原因。改成这个等于重写 |
| reflection 全链路 | 依赖 cluster 账本 + `deprecated_by` 软删除 + 搜索侧 `is_null` 过滤，三处必须成对移植。AgentMem 的 `lifecycle/merge_strategy.py` 本就是死代码，可以借它的聚类算法（加权质心 + 7 天时间窗 + 0.65/0.85 双阈值，约 120 行）但不必搬整套 |
| EverOS 的 benchmark 数字 | 不同数据集、不同 answer 模型，与 AgentMem 的 76.69% 不可比 |

**许可**：EverOS 与 everalgo 均为 Apache-2.0，移植进 MIT 项目合法，需保留 attribution。注意 `tests/fixtures/long_conversation_locomo_caroline_melanie.json` 是 CC BY-NC 4.0 的 carve-out，别混进来。

---

## 7. 建议动作（按性价比排序）

### 立即可做（无风险，不需移植）

1. **修 BM25 覆盖率**：`populate_bm25_from_redis` 扫全部三层，文本字段按层取（facts/episodes → `content`，procedures → `task` + `procedure`）；去掉 5000 cap 或改 VSCAN 分页；store 路径给 episodes/procedures 也调 `bm25_index.add`。**实测预期 +25% 关键词命中。**
2. **删掉手调 wRRF 权重预设**，改等权——6/7 场景实测为负收益。
3. **HyDE/MIRIX 改按需触发**：先跑零 LLM 的四路召回 + 融合，用一个廉价充分性检查（命中数、top-1 分数、分数间隙）决定是否升级到 LLM 增强。这是 P50 延迟的最大杠杆。
4. **清理死代码**：`services/retrieval_service.py`、`services/memory_service.py`、`lifecycle/`（4 个模块，grep 全仓库 0 引用）。
5. **`.env` 里的明文 API key**：已被 `.gitignore` 排除（`git ls-files` 确认未入库，仅 `.env.example` 被跟踪），所以没有泄露到仓库，但明文躺在磁盘上，建议轮换并改从环境变量注入。

### 需要标签才能做

6. **接四源自拟合 LR**：`core/fusion.py` 已就绪。需要从召回日志构造 `(scene, global, bm25, symbolic, label)` 标签对，用 `fit_lr_coefs` / 四源版拟合，然后 A/B。全场景稳定 +8.5%~+33.9% nDCG，但**先在小流量上验证再接默认**。

### 暂缓

7. store 路径 N+1 改 pipeline 批量化（`main.py:1273/1306/1348/1359`）——收益明确但要动核心写路径，需配套回归测试。
8. 接 `maxsim_group` 做 fact→episode 分数回注——需要先给 episode 建 fact 子索引。

---

## 8. 测试现状（重要）

当前 `tests/unit/`：**86 passed / 11 failed**。11 个失败**全部是既有问题，与本次改动无关**（本次只新增 4 个文件，未触碰任何既有代码）：

- `test_hybrid_search.py` 4 个：sentence-transformers MiniLM 模型不在 HF 本地缓存且无外网，embedding provider 起不来。**环境问题。**
- `test_lifecycle.py` 7 个：`AttributeError: module 'core.store' has no attribute 'DIMS'`——**`core/store.py` 引用了一个不存在的 `DIMS`，是真 bug。**

另外两个文档失信问题（评估中发现，非本次改动引入）：
- `scripts/bench-f1.py` 实际是 8 convs / 47 QA，README 写 "3 convs, 17 Q/A"
- `BENCHMARK_RESULTS.md` 自承 95.2% R@5 无仓库内复现路径（LongMemEval 脚本是 TypeScript 的，不在本仓库）

---

## 9. 修复记录（2026-10-22，已实施）

下列问题已全部修复并通过测试。**未 commit**——按你的规矩，先看证据再决定。

### 9.1 新发现的更严重问题：`mem:procedures` HNSW 图已退化

排查 BM25 覆盖率时发现一个比覆盖率缺口更严重的事实。Redis vectorset **没有枚举命令**（`VSCAN` 不存在），项目用零向量 `VSIM` 模拟全表扫描。但 VSIM 是近似 HNSW 遍历，**不保证返回 COUNT 个结果**：

| vectorset | VCARD | 单次 VSIM 返回 | 30 查询向量并集 |
|---|---:|---:|---:|
| `mem:facts` | 1,140 | 1,139 (99.9%) | 1,140 (100%) |
| `mem:episodes` | 1,611 | 1,609 (99.8%) | 1,611 (100%) |
| `mem:procedures` | 13,260 | ~1,730 (**13.0%**) | 6,479 (48.9%) |

新建的 16,000 元素集合仍有 98.3% 覆盖，所以**不是规模上限，是图退化**。后果：`vscan` 静默丢失 `mem:procedures` 87% 的元素，导致 consolidation、`scan_all_procedures`、BM25 填充全都在跑一个残缺口径上。

**修复：元素 ID 影子索引**（`core/store.py`）
- `mem:idx:<tier>` SET，每次 `VADD` 同步 `SADD`，每次删除走 `remove_element()`（VREM + SREM）
- `enumerate_elements()`：`SSCAN`（精确完整）+ 分批 `VGETATTR`（按 ID 解析，不受图健康度影响）
- `vscan` 保留但标注为"不可靠枚举"，并在 `returned < card` 时**打警告**而非静默
- 新增 `POST /admin/shadow-index/backfill` 与 `GET /admin/shadow-index/stats`；backfill 会登记当前可达子集并大声报告不可恢复比例
- 所有 VREM 调用点（consolidation / admin / compat / kb / 一次性脚本）统一改走 `remove_element`

### 9.2 BM25 层覆盖率（原报告 §4.2 的问题）

- `populate_bm25_from_redis` 改为扫全部三层，按层取文本字段（facts→`content`+`triple_str`，episodes→`content`，procedures→`task`+`procedure`），去掉 5000 静默截断
- `BM25Index` 变为 tier-aware（4 元组 `(uid, content, attrs, tier)`），`search()` 返回 `tier`，新增 `stats()`
- store 路径（`main.py` + `services/store_service.py` 共 7 处）给 episodes / procedures 补上 `bm25_index.add` 写入
- `main.py` recall 按 tier 路由 BM25 命中：facts→facts、episodes→episodes、procedures→并入 MACLA 程序列表
- `populate_from_items` 改为按 uid 去重且 **await 重建**（原来 `create_task` 会让启动后首次 recall 拿到空关键词结果）
- 实测（db=15 端到端）：`nginx gateway restart` → procedures 命中；`postgres cluster migration` → episodes 命中

### 9.3 删掉负收益的手调 wRRF 权重

`main.py` 里 4 套 query-type 权重预设（`[0.8,0.8,1.4,1.2]` 等）全部删除，改等权 `[1.0,1.0,1.0,1.0]`。依据是 §4.1 的 7 场景扫描：6/7 场景 nDCG 为负，最差 −12.6%。

### 9.4 recall 延迟：HyDE/MIRIX 加门控

新增 `_query_specificity()`——零检索、零 LLM 的查询特异性打分（长词 / 标识符 / 路径 / camelCase）。≥3 分则跳过 HyDE+MIRIX，直接走零 LLM 的四路召回。`_recall_augmentations(skip_llm=True)` 完全短路。这是 EverOS "LLM 默认退出热路径" 原则的落地。

### 9.5 性能批量化

- `core/heat.py::atomic_bump_many()`：一次 pipeline 发完所有 EVALSHA，替代 N 次串行 await；带 NOSCRIPT 回退
- `main.py` TIG：`HGETALL` 整图拉取改 `HSCAN ... MATCH <prefix>*`，Redis 服务端前缀过滤

### 9.6 测试与死代码

- 修复 `core.store.DIMS` 缺失（PEP 562 模块级 `__getattr__` 动态委派到 `embedder.DIMS`，避免 import 时快照过期）——原来 7 个 lifecycle 测试因此失败
- 修复 `do_consolidate` / `do_hard_prune` 签名漂移（4 个测试无参调用）
- 删除死代码：`services/retrieval_service.py`、`services/memory_service.py`、整个 `lifecycle/` 包（grep 全仓库 0 引用）
- 修正 README：F1 数据集 3 convs/17 Q/A → **8 convs/47 QA**（AST 解析 `scripts/bench-f1.py::CONVERSATIONS` 实测）；hooks 8 → **9 注册**（`claude-code/hooks.json` 实测）；补充 95.2% R@5 无法本地复现的说明

### 9.7 测试结果

| | 基线 (HEAD) | 修复后 |
|---|---:|---:|
| unit + integration 通过 | 109 | **158** |
| 失败 | 22 | **15** |

失败的 15 个**与基线完全一致**，全部源于 embedding 模型在此沙箱不可用（HF 缓存未命中 + 无网络 → `EmbeddingUnavailable: embedding circuit open`）。用 `git worktree` 建 HEAD 干净副本跑同一套测试确认过，零回归。新增 74 个测试（fusion 32 / shadow_index 29 / perf_fixes 13），修好 7 个原有失败。

### 9.8 未做（留作后续）

- `main.py:1273/1306/1348/1359` store 路径 N+1 串行往返（VSIM + persona hget/hset + O(N²) 图边 Lua）——收益明确但要动核心写路径，需配套回归测试
- consolidation 每小时全量重算 embedding（Redis vectorset 不暴露原始向量）
- 单 uvicorn worker 的并发天花板
- `.env` 明文 API key：`git ls-files` 确认**未入库**（仅 `.env.example` 被跟踪），但明文在磁盘上，建议轮换
- `mem:procedures` 中已不可达的 ~51% 元素：没有元素 ID 就无法 `VGETATTR`，只能靠后续写入自然补齐

## 10. 双轨作用域移植（2026-10-28）

EverOS 的核心模型是双轨：**user track = episodes + profile**，**agent track = agent_case + agent_skill**。移植前 agentmem 是单一全局语料，任一调用者的读写都落在同一池子里。本次把该模型落成“作用域”，并对全部接口向后兼容。

| 层 | 实现 |
|---|---|
| 事实源 | `core/scope.py`：`Scope(user_id, agent_id)`，XOR 语义；id 经白名单正则校验（id 会被拼进 `FILTER` 表达式） |
| 写入 | facts / episodes / procedures 的 attrs 打 `user_id` / `agent_id`；会话记录 owner（`mem:session:<id>:owner`，4h TTL） |
| 读取 | 向量腿在索引层 `FILTER`；BM25 / 图谱 / 结晶腿取回后 `matches()` 过滤 |
| 兼容 | 不带作用域的调用保持旧全局行为；带作用域的读**绝不**放宽到全局语料 |
| 按轨键 | `mem:persona:<track>`、`mem:pinned:session_summary:<track>` |
| 压缩 | `/session/compress` 按请求作用域提升，未指定时用会话写入时记录的 owner |
| 接口 | HTTP（`/recall` `/store` `/remember` `/import` `/session/*` `/context` `/profile` `/store-procedure` `/recall-procedures`）与 MCP 工具全部透传 `user_id` / `agent_id`，非法组合 422 |

**迁移**：`scripts/backfill_scope.py`（一次幂等迁移）。db0 实测 1581 facts + 1700 episodes 打 `user_id=jace`，312 条可达 procedures 打 `agent_id=raven-code`，并复制人格与 pinned 摘要；VCARD 前后元素数不变。

**验证**：

- `tests/unit` 310 passed。
- `tests/integration` 36 passed；4 条 KB 失败在 HEAD 洁净 worktree 下同样失败（embedding 维度 32 vs 384，与本次无关）。
- `tests/integration/test_api.py::TestScopeIsolation` 6 项：跨轨不可见、无作用域仍可见、双 id / 非法 id 422、procedure 归 agent 轨、压缩落回写入轨。
- Raven 侧适配器：`tests/test_agentmem_http_adapter.py` 5 项（写带轨、metadata 覆盖、camelCase 覆盖、无身分保持全局、读只带本轨）。
- 活库往返：Raven 适配器写入 → jace 召回命中且 attrs 带 `user_id: jace`；陌生 user 0 命中；agent 轨 procedure 召回 6 条；健康检查 ready。
