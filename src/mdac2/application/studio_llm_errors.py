"""Safe service diagnostics: never echo credentials, URLs or vendor response bodies."""
import httpx


class ServiceError(ValueError):
    pass


def service_error(exc, label):
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        reason = {400:'请求格式或模型参数不被支持',401:'API密钥无效或已过期',
                  403:'没有调用此模型的权限',404:'接口路径或模型不存在',
                  408:'服务端等待超时',413:'请求内容过大',
                  429:'服务端限流或额度不足'}.get(status,
                  '服务端暂时异常，请稍后重试' if status >= 500 else '接口拒绝请求')
        return ServiceError(f'{label}失败（HTTP {status}）：{reason}')
    if isinstance(exc, httpx.ReadTimeout):
        return ServiceError(f'{label}等待响应超时；服务端可能仍在处理，请稍后手动重试')
    if isinstance(exc, httpx.ConnectTimeout):
        return ServiceError(f'{label}连接超时，请检查网络与代理')
    if isinstance(exc, httpx.ProxyError):
        return ServiceError(f'{label}代理连接失败，请检查代理地址与代理服务')
    if isinstance(exc, httpx.ConnectError):
        return ServiceError(f'{label}无法连接服务，请检查网络、接口地址与代理')
    return ServiceError(f'{label}传输中断，请稍后重试或检查网络与代理')
