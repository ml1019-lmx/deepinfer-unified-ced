# DeepSeek 指令：Unified CED-PD 在改进版 DeepInfer 中的代码级可行性评估与改造计划

> **任务性质：代码审查与改造方案设计。先分析、后给计划；未经明确要求，不要直接修改代码。**
>
> 请完整阅读本指令，并实际检查当前工作区中的代码仓库。不要只根据 README、论文或概念描述给出泛化建议。

---

## 0. 你的角色与最终目标

你是一名资深大模型推理系统架构师和代码审查工程师，熟悉 DeepSeek-V4.1-Flash、DeepInfer、vLLM/vllm-ascend、Prefill/Decode disaggregation、KV Cache、SWA bounded replay、异步调度、ModelRunner、CUDA Graph 及 Ascend 执行路径。

本任务要评估：**如何在我们自己的改进版 DeepInfer 中，把 CED-PD 的 Prefill、Decode、Replay 执行路径整合成一套尽可能统一的调度和 ModelRunner 执行体系。**

目标不是重新实现一套推理引擎，也不是从零搭建环境，而是：

1. 以当前工作区中的**改进版 DeepInfer**为主要代码基线；
2. 参考 DeepSeek 官方 DeepInfer 上游实现，识别已有机制、接口和设计边界；
3. 参考 vllm-ascend 中可复用的调度、执行、KV Cache、KV Connector、ModelRunner 等实现；
4. 结合指定的 CED-PD 实测报告，区分已经验证的机制、尚未验证的假设和需要补充的实验；
5. 最终输出可落地的、落实到**具体文件、类、函数、调用链、数据结构和修改顺序**的代码改造计划。

不要把任务缩减成“增加一个 Replay 队列”或“增加一个 QueryKind 枚举”。重点是端到端执行链路能否闭环，以及同一个 batch / forward 中混合不同类型请求时，调度、metadata、Attention、KV 写入和采样是否正确。

## 1. 必须检查的代码来源

请先识别当前工作区目录、Git 仓库、分支和 commit SHA。若当前工作区中存在多个同类仓库，先列出路径和版本，不要擅自选择一个作为基线。

### 1.1 第一优先级：我们的改进版 DeepInfer

- 当前工作区中用户自己的 DeepInfer 改进版本。
- **这是主要实现基线，也是判断现有功能和代码改动量的唯一首要依据。**
- 先定位其相对官方上游的已有修改，尤其是 CED、KV Cache、PD Connector、Replay、Scheduler、ModelRunner、Attention 和 CUDA Graph 相关代码。
- 不要用官方上游代码覆盖、替代或否定本地已有改进。
- 如果无法访问本地改进版，明确指出缺少哪个目录/仓库，并停止对具体文件和函数的臆测；可以先完成参考架构梳理，但必须把结论标为待代码验证。

### 1.2 DeepSeek 官方 DeepInfer

- Repository: https://github.com/deepseek-ai/DeepInfer
- 参考上游架构、已有执行机制、数据结构和接口。
- 必须注意：官方公开 README 明确说明，完整推理服务所需的 KV Cache 系统和 VE/Prefill/Decode disaggregation 不包含在该公开版本中（尽管大部分 engine functionality 已实现）。
- 因此，不能假设官方上游已经具备完整 PD/CED 实现，也不能把 README 中的 feature list 当作代码实现证据。
- 对照上游时，请明确区分：上游已有、本地改进版已有、需要新增、需要重构。

### 1.3 vLLM Ascend

- Repository: https://github.com/vllm-project/vllm-ascend
- 检查当前工作区是否有本地 checkout 或项目实际使用的特定分支；如果有，以实际使用版本为主要参考，并记录 commit SHA。
- 官方仓库/分支用于对照调度、Worker/ModelRunner、Attention metadata、KV Cache、KV transfer connector、异步执行和 CUDA Graph 等设计。
- 不要仅凭 vLLM 上游接口推断 vllm-ascend 的真实行为；必须定位 Ascend 插件中实际的调用链和适配代码。
- 不要把 vLLM 的某个机制直接搬到 DeepInfer；需要说明接口差异和适配成本。

### 1.4 CED-PD 实测报告

请阅读完整报告及其必要的内部引用：

https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/docs/CED-PD-PERF-20260925.md

报告中的实验结论只能作为对应实验配置下的证据，不得外推为尚未测量的 Unified Scheduler 或 Replay/Decode 混批收益。

请特别保留以下实验口径：

- 已测试的 CED 路径：P 执行 Layer 0–19 + Layer-20 全局源投影；D 执行完整 40 层，并进行 128-token 有界 SWA Replay。
- 报告中的 Prefill 吞吐对照：
  - 32K：CED 约 1.74×；
  - 144K：CED 约 2.07×；
  - 1M：CED 约 2.78×。
- 同口径对照中，CED 与基线的 Decode ms/step 基本一致（差异小于约 1.1%），说明该实验的主要收益来自 P 侧 Prefill 层数减少，不能据此声称 Decode 本身加速。
- 144K / 1M 的 needle 正确性、多轮正确性和 prefix-cache 结果，应按报告中的具体配置和测试口径引用。
- 报告 §4.5 的并发测试只覆盖 CED 臂、并发扫描到 MAX_SEQS=4；它不是 Replay 注入实验，也没有证明 Replay 与正常 Decode 混批时的 TPOT 影响。
- 报告还记录了基线 D 侧 MULTISTREAM / DSA_OVERLAP 路径在特定长上下文配置下的错误。请把它作为报告中已定位的独立风险，不要误归因到 CED 或 KV 传输。
- 报告中非流式请求的 tpot_ms 可能包含 Prefill 时间。评估 Decode 性能时，应优先使用真实流式逐 token 计时和真实生成 token 数，不能混淆 TTFT、Prefill 吞吐和 Decode TPOT。

如报告与当前代码不一致，以当前代码和可复现实验为准，并说明差异。

## 2. 明确的范围边界

以下内容**不属于本次评估和改造计划范围**：

- 基础环境搭建、系统安装和环境变量初始化；
- 硬件适配或硬件选型；
- 依赖安装、依赖编译、镜像构建；
- CANN 安装、CANN/backend 初始化；
- 从零搭建 DeepInfer 或 vllm-ascend；
- 与本次 CED-PD 执行路径无关的通用性能优化。

我们已经有自己的改进版本和基础环境。请直接从代码执行逻辑出发。

不要输出环境安装教程、Dockerfile、编译命令清单或硬件适配建议来填充篇幅。

## 3. 需要评估的总体设计目标

希望评估一套统一执行模型，核心方向如下：

### 3.1 QueryKind / 请求执行语义

重点评估能否在请求或 sequence 维度区分：

- `PREFILL`：常规 Prefill；
- `DECODE`：正常自回归 Decode；
- `REPLAY`：为补齐缺失的 SWA KV 等状态而执行的历史 token 重算。

可以评估使用 QueryKind 枚举，也可以沿用或扩展现有请求类型/metadata；不要为了增加枚举而增加无意义抽象。

**关键要求：类型必须能在 request/sequence 粒度表达。** 不能只用一个 batch-global flag 判断整个 batch 是 Prefill、Decode 或 Replay，否则无法可靠表达混合 batch。

### 3.2 Replay 与 Decode 尽可能进入同一 Forward 执行路径

重点评估：

- Replay 请求与正常 Decode 请求能否进入同一个 ModelRunner forward / execution step；
- 同一批次内能否按 sequence/query metadata 区分 Replay 和 Decode；
- Attention metadata、position、sequence length、block table、query offsets 等是否支持混合语义；
- Replay 是否只写入需要补齐的 KV group，而不破坏已存在的 Global KV；
- Replay 对应的 token 是否只用于重算状态，是否应跳过正常 sampling、输出 token 提交、生成长度递增和用户可见输出；
- Decode 请求是否仍正常完成 logits、sampling、MTP/DSpark（若启用）及 token commit；
- 是否需要对不同 QueryKind 使用不同的 kernel/attention 子路径，但仍共享一次 ModelRunner 调度和 forward 调用。

不要把“同一个 Forward”误解为所有请求必须使用完全相同的 kernel 参数或完全相同的 Attention mask。需要从代码中判断可共享到哪一层。

如果同一 Forward 在当前实现中不可行，请给出最小阻碍、具体代码证据和替代方案，并量化额外调度/launch/同步成本。

### 3.3 KV Group 的差异化处理

请从代码确认 Global KV（GKV）与 SWA KV 等 group 的实际定义、布局、生命周期和读写规则。

需要评估：

- 不同 KV group 的存在性和完成状态如何表达；
- P 侧传输哪些 group，D 侧如何识别已加载和缺失的 group；
- Replay 如何根据缺失 group 计算重放范围；
- Replay 只补写缺失 group 时，如何防止重复写入或覆盖正确数据；
- block id、block table、slot mapping、page/block size、TP/DP/EP 映射及 group 顺序是否一致；
- 异步传输完成与 Replay 开始之间是否有正确的依赖和同步；
- prefix cache 命中、部分命中、请求取消、异常重试和 KV 回收是否安全。

不要假设 GKV/SWA 的数据结构、block ID 或映射关系；必须以实际代码为准。

## 4. 六个重点改造阶段

请按以下六个阶段分别审查，每个阶段都要给出当前实现、目标差距、具体代码位置、改造建议、依赖关系、风险和测试。

### 阶段一：CED Layer Execution

检查模型层执行和 P/D 分工：

- P 侧执行哪些 layer，哪些模块/权重不加载或不执行；
- D 侧如何接收 Prefill 结果并继续正常 Decode；
- Global Source / GKV 相关投影的计算位置和数据依赖；
- layer index、hidden state、embedding、position、RoPE、残差和模型中间状态如何传递；
- P 侧裁剪层或模块后，checkpoint loading、参数注册、forward 遍历和层编号是否一致；
- 当前 CED 路径与完整 40 层基线的真实差异。

请把“已存在的 CED 实现”和“拟议的统一化改造”分开描述。

如代码/工作区中存在更激进的候选路径（例如 P 侧只计算 Global Source/GKV，D 侧重算更多或全部必要 SWA 状态），请单独列为候选方案，分析额外 Replay 计算量、传输数据量、TTFT、D 侧资源占用和正确性依赖；不要把它与报告已经验证的 20 层 CED 路径混为一谈。

### 阶段二：KV Group Transfer

从 P 侧 KV 产生，到 Connector、网络传输、D 侧接收和 KV block 安装，完整梳理调用链。

重点检查：

- transfer descriptor / metadata 的定义；
- group ID、block ID、shape、dtype、stride、offset、layer mapping；
- 发送端和接收端的 group 对齐；
- 部分 group 缺失和部分传输完成时的状态；
- connector 的异步完成回调、事件、stream 和生命周期；
- D 侧何时可以安全地将请求交给 Replay 或 Decode；
- prefix cache 与远端 KV transfer 的交互。

输出一张“传输字段契约表”，逐字段列出生产端、消费端、定义位置、校验方式和潜在不一致风险。

### 阶段三：D-side Replay

详细检查现有 SWA bounded replay 的真实实现，不要只根据配置项或 README 推断。

重点回答：

1. Replay 的触发条件是什么？由哪个模块决定？
2. Replay 起止 token/position 如何计算？是否考虑 SWA window、最大 backtracking、保留 suffix 和已有 KV group？
3. Replay 的输入来自 token IDs、embedding、hidden states，还是其他中间状态？具体代码在哪里？
4. Replay 会经过哪些 layer 和 Attention group？
5. Replay 的 KV 写入如何与正常 Decode 区分？
6. Replay 完成条件如何通知 Scheduler？
7. Replay 是否会产生 logits、sampling 或一个用户可见 token？如果有，为什么；如果没有，如何确保输出计数和 request state 不被误推进？
8. 发生 prefix cache hit、部分 KV group 命中、transfer 延迟、请求取消时如何处理？
9. 当前 bounded replay 的计算量如何随保留 token 数和 layer 数变化？

请特别区分：
- “为了补齐 KV 状态而重算历史 token”；
- “正常 Decode 生成下一个 token”；
- “普通 Prefill”。

不要把三者仅仅当成不同长度的同一种请求。

### 阶段四：Unified Scheduler

评估如何将 Prefill、Decode、Replay 纳入统一调度。

重点检查：

- 当前各类请求分别从哪里进入调度；
- Admission Gate、batch builder、token budget、max-num-batched-tokens、sequence budget 和优先级逻辑；
- 是否存在将 Replay 错误识别为 Prefill 或 Decode 的分类逻辑；
- Replay 是否会独占一个 step，是否可以与 Decode 同步推进；
- 混合 batch 如何保证每条 sequence 的 QueryKind、token 数、position 和 KV 状态正确；
- Replay 的排队是否导致正常 Decode 的 head-of-line blocking；
- Decode 的 TPOT P50/P99、吞吐和公平性如何受到影响；
- preemption、abort、streaming、多轮请求、prefix cache 和多租户场景下的状态转换。

请给出建议的请求状态机和调度时序图。若建议 Replay 与 Decode 同步进入一个 step，必须明确说明 batch 形成条件、执行前置条件、完成回调和下一步状态变化。

### 阶段五：Unified ModelRunner

这是本次改造的核心，请深入到 forward 的数据结构和调用链。

重点拆解：

**A. Mixed Query Metadata**
- 如何在同一 batch 中表达不同请求的 QueryKind；
- query_start_loc、seq_lens、positions、block_tables、slot_mapping、context lengths、token counts 等字段是否需要扩展；
- batch 内不同 query 类型的 offsets 和 mask 如何组织；
- metadata 是 batch-level 还是 sequence-level；若当前设计不支持混合，需要改哪些结构。

**B. Attention**
- Attention backend 如何识别不同请求的 QueryKind；
- Replay 与 Decode 的 query length、KV length、causal mask、window mask、RoPE position 是否不同；
- 不同 KV group 的读写选择如何传入；
- 是否能在一个 forward 内处理不同请求的不同 Attention 语义；
- 如果不能，最小拆分点在哪里。

**C. KV Write**
- Prefill、Replay、Decode 分别写哪些 KV group；
- slot mapping 和 block table 是否能按请求正确映射；
- Replay 是否可能覆盖已有 GKV；
- KV 写入完成标志如何与 Scheduler 状态衔接；
- 对异步写入、并行 stream 和图模式有何影响。

**D. Selective Sampling**
- 只对需要生成 token 的 Decode query 执行 sampling；
- Replay query 不应意外增加 output token 数、推进生成状态或产生用户可见 token；
- 混合 batch 的 logits 行如何映射回原始 request；
- MTP/DSpark 等投机解码逻辑若存在，是否会与 Replay 混淆；
- 如果采样模块假设 batch 中所有请求都需要采样，应如何最小化改造。

请提供一张“Unified ModelRunner 数据流图”，从 Scheduler 输出 batch 开始，经过 metadata 构建、Attention、KV write、logits、selective sampling、request state update，标注每个 QueryKind 的分支点。

### 阶段六：CUDA Graph + Performance

不要一开始就把 CUDA Graph 纳入改造。建议遵循：

1. 先在 Eager 模式验证 Replay/Decode 混合执行的正确性；
2. 再确认 metadata、shape、KV 写入和 sampling 的稳定接口；
3. 最后评估 CUDA Graph capture/replay、graph key、padding/bucketing、动态 shape 和 fallback。

重点检查：

- 当前 Graph capture 的入口和 graph key；
- QueryKind 是否需要进入 graph key，还是可以作为 graph input metadata；
- 混合 batch 的动态 token 数和 offsets 是否会破坏 capture；
- 不同 QueryKind 是否可共享 graph；
- 哪些操作需要 eager fallback；
- Graph 内外 stream、event、KV transfer 和异步 scheduler 的依赖；
- 是否存在 Replay 完成后立即 Decode 的 race condition。

性能评估必须拆分：
- P 侧 Prefill：TTFT、Prefill token/s；
- D 侧 Decode：单流 TPOT P50/P95/P99、Decode token/s；
- Replay：Replay token 数、Replay wall time、Replay 占用的 step 数、Replay 对 Decode 的干扰；
- 系统整体：总吞吐、并发数、排队时间、活跃 Decode 数、KV 利用率；
- 混合调度：Replay 与 Decode 的 batch 组成、每类 token 数和 step 时间。

请设计以下对照实验，至少包含：
- 当前基线；
- CED + 当前 Replay 行为；
- Replay 独立执行；
- Replay 与 Decode 进入同一 Forward，但分别按 request metadata 执行；
- Eager 与 CUDA Graph；
- 有/无 prefix cache；
- 不同 Replay suffix 长度和并发数。

不能用非流式 wall time / completion_tokens 冒充 Decode TPOT。需要使用真实流式逐 token 计时，并明确首 token、最后 token、token 数和统计分位数的口径。

## 5. 必须完成的代码审查方法

请按以下顺序执行，不要跳过代码定位：

### Step 1：仓库与版本清点

列出：
- 仓库路径；
- remote URL；
- branch；
- commit SHA；
- 是否有未提交修改；
- 哪一个是用户改进版，哪一个是官方 upstream。

### Step 2：端到端调用链追踪

从请求进入开始，至少追踪：

`Request/API → Request/Sequence State → Scheduler/Admission → Batch Builder → Model Input/Metadata → ModelRunner → Model Forward → Attention → KV Write → Logits/Sampler → State Update`

CED-PD 另外追踪：

`P Prefill → KV Group Production → Connector/Transfer Descriptor → D-side KV Install → Replay Decision → Replay/Decode Execution`

每一段都要给出实际文件路径、类名、函数名和关键代码位置。

### Step 3：建立当前实现与目标设计差异表

每个功能点至少包含：

| 功能点 | 当前实现 | 所在文件/函数 | 目标行为 | 缺口 | 最小改造 |
|---|---|---|---|---|---|

所有“当前实现”都必须由代码证据支撑。

### Step 4：检查关键接口契约

至少输出以下契约：

1. Request/Sequence state；
2. QueryKind / Query metadata；
3. KV group descriptor；
4. Replay input/output；
5. ModelRunner input/output；
6. Selective sampling index mapping；
7. Scheduler completion callback。

逐字段列出类型、shape/维度、生产者、消费者、生命周期和校验方法。无法从代码确认的字段明确标注 UNKNOWN，不得自行编造。

### Step 5：评估最小可行改造路径

优先复用现有实现，避免无必要的大规模重构。

每个改造点说明：
- 要改的文件；
- 类/函数；
- 现有逻辑；
- 修改后的逻辑；
- 调用方/被调用方是否要同步改；
- 对现有 PREFILL/DECODE 路径的兼容性；
- 是否影响 KV Cache、prefix cache、MTP/DSpark 和 CUDA Graph；
- 单元测试与端到端测试。

如果建议新增抽象，说明它解决了哪个现有代码问题，以及为什么不能通过扩展现有接口实现。

## 6. 正确性验证矩阵

至少设计下列测试，并指出应该放在哪些现有测试目录/测试框架中：

| 场景 | 验证目标 |
|---|---|
| 纯 Prefill | 不影响现有 Prefill 正确性 |
| 纯 Decode | 不影响现有 Decode 输出和 TPOT |
| 纯 Replay | 补齐所需 KV，不能意外推进生成状态 |
| Replay + Decode 同一 batch | request 粒度 metadata、Attention、KV write、sampling 均正确 |
| GKV 已有、SWA 缺失 | Replay 只补缺失 group，不破坏 GKV |
| 部分 KV transfer 完成 | 不会过早执行 Replay/Decode |
| Prefix cache 全命中 | 不触发不必要 Replay |
| Prefix cache 部分命中 | 正确计算缺失状态和 Replay 范围 |
| 请求取消/超时 | transfer、Replay 和 KV 生命周期正确回收 |
| 多轮请求 | request state 和 KV 复用正确 |
| 长上下文 | 144K、1M 等目标长度下无静默错误 |
| Streaming | Replay 不泄漏为用户可见 token |
| MTP/DSpark 开关 | Replay 不污染投机解码状态 |
| Eager vs Graph | 输出、KV 状态和状态转换一致 |
| 混合并发 | 无死锁、无长期饥饿、无异常 TPOT 长尾 |

报告中的 144K/1M needle 和多轮测试可以作为回归基准，但需要新增专门的 Replay/Decode 混批测试。不能用已有 CED 测试替代混批正确性测试。

## 7. 风险分析与备选方案

至少比较以下方案：

- 方案 A：Replay 独立执行，Decode 独立执行；
- 方案 B：Replay 与 Decode 共享调度 step，但分开执行 forward；
- 方案 C：Replay 与 Decode 在同一个 ModelRunner forward 中，通过 request-level QueryKind / metadata 实现混合执行。

对每个方案分析：
- 代码改造范围；
- 正确性风险；
- Attention/KV write 复杂度；
- CUDA Graph 复杂度；
- kernel launch / 同步开销；
- Decode TPOT P50/P99 影响；
- 吞吐和公平性；
- 回滚难度。

不要预设方案 C 一定最好。如果当前 Attention backend 或 KV write 结构不支持混合执行，应以代码证据说明阻碍，并给出阶段性方案 B → C 的演进路径。

## 8. 最终输出格式（必须遵守）

请用中文输出，技术名词保留英文。不要只给概念图或伪代码。

### A. Executive Summary
- 总体可行性：可行 / 有条件可行 / 当前不可行；
- 最大的 3–5 个技术阻碍；
- 推荐方案及理由；
- 预计改造风险等级；
- 最小可行版本（MVP）范围。

### B. Repository & Version Inventory
列出仓库、路径、branch、commit SHA 和本地改进版相对 upstream 的关键差异。

### C. Current Code Architecture
给出当前执行链路图、CED-PD 数据流图，并标注真实文件/函数。

### D. Six-Stage Gap Analysis
严格按六个阶段逐项分析，给出当前行为、代码证据、目标行为、差距、改造建议。

### E. File / Class / Function Change List
必须是可执行的文件级清单，至少包含：

| 优先级 | 文件 | 类/函数 | 修改内容 | 调用链影响 | 风险 | 测试 |
|---|---|---|---|---|---|---|

不要编造不存在的文件名、类名、函数名。无法确认时标记 TODO / UNKNOWN，并说明如何在仓库中进一步定位。

### F. Data Contract & State Machine
给出 QueryKind、Replay metadata、KV group transfer descriptor、ModelRunner input、sampling mapping 的建议契约，以及请求状态机。

### G. Implementation Roadmap
按依赖关系拆分 PR/commit：
- PR-0：代码审查和接口契约确认；
- 后续 PR：按最小依赖顺序安排；
- 每个 PR 写明修改文件、核心逻辑、测试和验收标准；
- Eager correctness 通过后，才能进入 CUDA Graph 阶段。

### H. Test & Benchmark Plan
给出正确性测试矩阵、混合 batch 测试、长上下文测试、Eager/Graph 对照和性能统计口径。

### I. Risks / Alternatives
对 A/B/C 三种方案作出对照，列出不可行点、风险和 fallback。

### J. Open Questions
只列真正被代码或资料阻塞、必须由开发者确认的问题。不要把能从仓库中查到的问题推给用户。

## 9. 证据和回答纪律

以下要求是硬约束：

1. **每个关键判断都要有代码证据**：仓库路径 + 文件 + 类/函数 + 行号或可定位代码片段。
2. 区分四种信息：
   - CODE VERIFIED：代码直接证实；
   - REPORT MEASURED：指定报告中实测；
   - INFERRED：基于代码/数据的推断；
   - UNKNOWN：目前无法确认。
3. 不得虚构函数名、文件名、配置项、性能数字、接口字段或测试结果。
4. 如果引用上游代码，注明仓库和 commit/branch；不要把上游实现说成本地改进版已有实现。
5. 不要把 CED 的 Prefill 加速比说成 Decode 加速比。
6. 不要把普通 Decode、Prefill 和 Replay 混为一谈。
7. 不要声称已有测试证明 Replay/Decode 混批正确或有性能收益，除非找到对应测试或实测数据。
8. 如果仓库代码与报告结论不一致，明确指出，不要自行抹平差异。
9. 不要输出环境安装、依赖编译、CANN/backend 初始化或硬件适配教程。
10. 先完成代码审查和方案评估；未经明确要求，不要自动改写仓库文件。

**现在开始：先清点仓库和 commit，再追踪真实调用链，最后按上述格式输出代码级评估与改造计划。不要跳过代码阅读直接给方案。**
