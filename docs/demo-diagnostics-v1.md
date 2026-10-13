# Demo API v1：认证、权限与上游故障规则

- 文档 ID：`demo-diagnostics-v1`
- 服务版本：`demo-api-v1`
- 资料性质：虚构服务的模拟规则

本页只定义 SupportLab 虚构服务的历史诊断规则，不代表 DeepSeek 或其他供应商的业务政策。状态码或错误码单独出现都不足以确认原因；必须结合同一账号、请求、时间和版本的诊断快照。

| HTTP 状态 | 错误码 | 可确认的结论 | 必要证据与处理边界 |
| --- | --- | --- | --- |
| 401 | `CREDENTIAL_REVOKED` | 该次请求使用的凭据已被撤销 | 快照 credential_status 为 revoked；不要索要或展示密钥；请有权限的维护者核实当前凭据状态。 |
| 401 | `TOKEN_EXPIRED` | 该次请求使用的凭据已过期 | 快照 credential_status 为 expired，credential_expires_at 不晚于请求时间；不能仅凭 401 断言密钥失效。 |
| 403 | `SCOPE_MISSING` | 该次请求缺少所需权限 | credential_status 为 active；required_scope 是非空权限名且不在 granted_scopes 中；不能把 403 一律归因于账号无效。 |
| 503 | `UPSTREAM_INCIDENT_CONFIRMED` | 该次请求关联上游故障事件 | upstream_service 与 incident_service 相同；非空 incident_id；请求处于事件开始时间（含）至结束时间（不含）内，结束时间可未知；事件关联不证明请求没有执行，不自动重试。 |
| 504 | `UPSTREAM_TIMEOUT_UNKNOWN` | 超时后无法确认是否执行成功 | execution_state 为 unknown；idempotency_supported 只能为 true、false 或 null；收集请求与快照并核对本行后转人工；先由维护者核实执行结果，不自动重试，也不将未知当作未执行。 |

所有快照都只描述当时状态。撤销、过期、缺权限的记录不能证明今天仍然如此；历史事件结束不证明整个服务已经恢复。只能给出补证据建议，本项目不修改密钥、权限或真实服务配置。
