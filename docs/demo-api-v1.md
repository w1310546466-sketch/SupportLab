# 虚构服务 Demo API v1：最小排错资料

- 文档 ID：`demo-errors-v1`
- 服务版本：`demo-api-v1`
- 数据集版本：`requests-v1`
- 性质：为 SupportLab 构造的模拟规则，不代表任何真实供应商。

| HTTP 状态 | 服务错误码 | 本虚构服务中的含义 | 建议进一步核查 |
| --- | --- | --- | --- |
| 429 | `RATE_LIMIT_EXCEEDED` | 请求触发频率限制 | 当前频率额度、请求窗口、`retry_after_seconds` |
| 429 | `MONTHLY_QUOTA_EXHAUSTED` | 月度用量配额耗尽 | 当前月度配额、已用量及重置时间 |
| 403 | `PERMISSION_DENIED` | 请求所需权限不足 | 当前账号权限与目标资源 |

`retry_after_seconds` 是该次请求记录的建议等待秒数；`null` 表示未提供等待时长，不能解释为零秒。
同为 429，服务错误码可以提供不同线索。只看 HTTP 状态码不能断定余额不足。
当前工具只返回事实，不生成完整诊断。账号状态工具已提供请求时刻快照，字段与规则见
[账号快照说明](demo-account-state-v1.md)。

当前演示账号为 `acc_demo`。`req_other_001` 属于另一个模拟账号，用于检查访问范围；
在演示账号下，它与不存在的编号均返回 `not_found`，不披露其他账号的数据或记录是否存在。
