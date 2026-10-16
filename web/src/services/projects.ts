export interface Chart {
  revision_id: string
  timing_revision: string
  events: Record<string, unknown>[]
  onset_revision: string | null
  arrangement_policy_hash: string | null
  semantics_policy_hash: string | null
  user_patch_ids: string[]
  schema_version: number
}
export interface Project { project_id: string; title: string; head_revision: string | null; archived: boolean; chart: Chart | null }
export class ApiError extends Error {
  status: number
  constructor(status: number) {
    super(({401: '安装令牌无效，请重新连接。', 403: '请从受信任的本机地址打开。', 404: '工程不存在。', 409: '工程已被其他窗口修改，请查看差异。', 422: '数据格式不合法。'} as Record<number, string>)[status] ?? '请求失败，请检查本地服务后重试。')
    this.status = status
  }
}
const identity = (id: string) => { if (!/^[A-Za-z0-9_-]{1,128}$/.test(id)) throw new Error('工程 ID 不合法。'); return id }
export class ProjectsApi {
  private token: () => string
  private fetcher: typeof fetch
  constructor(token: () => string, fetcher: typeof fetch = fetch) { this.token = token; this.fetcher = fetcher.bind(globalThis) }
  private async request(path: string, body?: unknown): Promise<any> {
    let response: Response
    try {
      response = await this.fetcher('/api/v1/projects' + path, { method: body === undefined ? 'GET' : 'POST',
        headers: { Authorization: `Bearer ${this.token()}`, 'Content-Type': 'application/json' },
        cache: 'no-store', redirect: 'error', ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
    } catch { throw new ApiError(0) }
    if (!response.ok) throw new ApiError(response.status)
    try { return await response.json() } catch { throw new ApiError(502) }
  }
  async create(title: string): Promise<Project> {
    if (!title.trim() || title.length > 256) throw new Error('请输入 1–256 字的工程名称。')
    return { ...await this.request('', { title }), chart: null }
  }
  async get(id: string): Promise<Project> { return this.request('/' + identity(id)) }
  async save(project: Project, chart: Chart): Promise<{ revision_id: string }> {
    return this.request('/' + identity(project.project_id) + '/commands', {
      command_id: crypto.randomUUID(), expected_revision: project.head_revision, type: 'replace_chart', payload: chart,
    })
  }
}

