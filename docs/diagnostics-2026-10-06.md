# 认证、权限、上游与超时扩展：交付记录

日期：2026-10-06。本次按用户要求继续实现和验证，暂不展开原理教学。业务数据与规则全部虚构，DeepSeek 只提供真实模型能力。

## 已实现且验证的行为

| 请求编号 | 实际核对的证据 | 预期行为 | 本轮真实模型验证 |
| --- | --- | --- | --- |
| req_auth_revoked | 401、CREDENTIAL_REVOKED、同请求凭据状态 revoked、对应规则 | 历史凭据撤销诊断 | 通过一次 |
| req_auth_expired | 401、TOKEN_EXPIRED、expired、过期时间不晚于请求、规则 | 历史凭据过期诊断 | 未运行 |
| req_permission_missing | 403、SCOPE_MISSING、active、所需权限不在授权列表、规则 | 历史权限缺失诊断 | 未运行 |
| req_upstream_incident | 503、UPSTREAM_INCIDENT_CONFIRMED、同上游服务、事件编号与覆盖请求的时间区间、规则 | 请求关联故障事件；不承诺未执行或立即恢复 | 未运行 |
| req_timeout_unknown | 504、UPSTREAM_TIMEOUT_UNKNOWN、执行未知、幂等支持标记、规则 | 带来源的转人工说明；不自动重试 | 通过一次 |

保留原来的两种 429，当前共六种可确认诊断及一种执行未知的受控转人工路径。未登记错误仍安全停止，不能处理所有 401/403/5xx 根因。

仍是三个只读工具；账号查询工具按请求返回用量快照或诊断快照。新增快照保存在独立表中，查询仍受可信账号、请求、时间、版本限制。只接受限定字段；不保存 API 密钥。导入时拒绝覆盖无关请求，重复导入不产生重复记录。基础和扩展导入分别是一个事务；扩展失败不撤销此前已完成的基础导入。

原始 requests.json、account_states.json 和 demo-api-v1.md 字节未改变。默认数据库现有 8 条请求、3 份用量快照和 5 份诊断快照；知识库 4 篇、32 个片段。新增固定规则文件为 demo-diagnostics-v1.md，SHA-256 为 `b13a16f7fe346f4486995f1528d2a73a01ccf73643929353d9e7a6081e074a58`；五条规则各自核对原文行哈希，改文档后须重新审核程序规则。

## 验证与实际失败

- 最终完整测试：224/224，33.065 秒；比此前增加 20 项测试方法，含多组边界子案例。覆盖凭据过期边界/时区、权限已存在、事件不匹配/已结束、快照缺失/陈旧/错误版本、跨账号、错误 JSON、伪造引用、提前声称执行未知、显示偏好与连续对话。
- 新的 diagnostic-dev-v1：20 个构造开发场景、10 个家族，固定取证与动态调用两种离线适配路径合计 40/40 通过。离线回复由规则构造，实际 API 为 0；验证程序与约束，不代表真实模型质量。
- 原 conversation-dev-v1：增加文档后由 20/20 回退到 18/20。天气问题通过“今天”误命中，发票问题通过中文相邻字片段“个服”误命中。对通用问句先分隔再切词，修复后 20/20。前后报告保留；当前词面过滤仍不等于通用语义检索。
- 实现期间还修复了新文档元信息与登记契约不符、日志假定所有快照都有配额字段、停止引用加入时的缩进错误，以及新增测试漏建会话的问题；未将这些中间运行算作通过。
- 所有本轮场景都用于开发，可见并可能参与调优；不称作独立保留集，人工业务真值及引用核验仍为 pending。

| 真实场景 | API 次数 | 工具次数 | 输入 tokens | 输出 tokens | 合计 tokens | 耗时 ms | 格式纠正 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 凭据撤销 | 4 | 3 | 9647 | 351 | 9998 | 10350.695 | 1 次后通过 |
| 超时执行未知 | 3 | 3 | 6204 | 194 | 6398 | 7516.271 | 0 |
| 合计 | 7 | 6 | 15851 | 545 | 16396 | 不作性能对照 | 1 |

撤销案例第 3 次模型返回发生 JSON 语法错误：408 字符、非代码围栏开头、解析位置第 1 行第 1 列。第 4 次为同预算内格式纠正，随后通过证据检查。原始错误回复未保存，因此不推断具体错法。没有追加调用、自动网络重试或计算未经核实的费用。

两例独立冒烟验证，不是固定流程/Agent 的真实对照，也不是重复稳定性测量。每例最多 4 次 API、4 次工具、8 轮决策，单请求 25 秒、全流程 110 秒，输出上限 768 tokens；SDK 自动重试为 0。没有大批量付费评估。

最终代码指纹：`a722d1af3211d408a4d7ce1578ce00c7e727b489daabbd0c9e7623aed9684345`（supportlab/*.py 内容与路径哈希，算法见 evaluation.fingerprint；仓库尚无提交）。版本为 tool-loop-v4.3、deepseek-chat-v4.3、grounded-diagnostics-v1、grounded-stop-v2、fts-keyword-v3。依赖未变化。

报告：[新增离线场景](../reports/diagnostics-offline-2026-10-06.json)、[真实撤销](../reports/diagnostics-live-revoked-2026-10-06.json)、[真实超时](../reports/diagnostics-live-timeout-2026-10-06.json)、[检索回归修复前](../reports/quality-extension-before-2026-10-06.json)、[修复后](../reports/quality-extension-after-2026-10-06.json)。

## 复现

在项目根目录运行。前五条命令不会调用模型。

```powershell
.\.venv\Scripts\python.exe -m supportlab init-demo --extended
.\.venv\Scripts\python.exe -m supportlab kb-build
.\.venv\Scripts\python.exe -m supportlab eval-diagnostics --output reports/my-diagnostics.json
.\.venv\Scripts\python.exe -m supportlab eval-app
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
# 以下每条会调用真实模型，最多 4 次 API；不要把密钥贴到聊天
.\.venv\Scripts\python.exe -m supportlab eval-diagnostics --case revoked --live --output reports/my-revoked.json
.\.venv\Scripts\python.exe -m supportlab eval-diagnostics --case timeout --live --output reports/my-timeout.json
```

## 收尾验收与学习状态

功能原型已经覆盖最初计划的三类故障，但还不能宣称成熟交付。下一步应冻结当前规则和接口，按场景家族建立独立保留测试并先完成离线验收；更广泛的真实评估需明确调用预算。之后补失败分析、可复现演示与求职描述。简单网页和模拟工单是可选展示扩展，向量检索、多 Agent 不是强制前提。

凭据过期、权限缺失、上游事件三条新增路径只验证过离线程序；独立人工复核、未参与调优的测试集及真实重复运行均未完成。自动转人工只输出说明，没有联系人员或创建工单。用户理解进度保持原记录，不能把 AI 完成代码和测试等同于用户已掌握；集中讲解与小练习继续暂缓。
