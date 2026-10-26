# API 使用说明

## 认证

- 所有接口使用 Bearer Token：`Authorization: Bearer <token>`。
- Token 在控制台「开发者设置」生成，有效期 90 天。

## 限流

- 默认每个账号 100 次/分钟，超出返回 429。
- 需要更高配额走工单申请。

## 示例

```bash
curl -H "Authorization: Bearer $TOKEN" https://api.example.com/v1/docs
```

## 错误处理

- 401：Token 无效或已过期，重新生成。
- 429：触发限流，按响应头 `Retry-After` 指定的秒数重试。
