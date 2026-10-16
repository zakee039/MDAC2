import { ApiError, ProjectsApi } from '../../services/projects.ts'
import type { Chart, Project } from '../../services/projects.ts'

// The draft survives network errors and conflicts. No automatic retry/rebase of writes.
export class ProjectSession {
  project: Project | null = null
  draft: Chart | null = null
  remote: Project | null = null
  busy = false
  error = ''
  conflict = false
  dirty = false
  private api: ProjectsApi
  constructor(api: ProjectsApi) { this.api = api }
  private async run(action: () => Promise<void>) {
    if (this.busy) return
    this.busy = true; this.error = ''
    try { await action() } catch (error) { this.error = error instanceof Error ? error.message : '操作失败。' }
    finally { this.busy = false }
  }
  private install(project: Project) {
    this.project = structuredClone(project); this.draft = structuredClone(project.chart)
    this.remote = null; this.conflict = false; this.dirty = false
  }
  async create(title: string) { await this.run(async () => { if (this.dirty) throw new Error('请先保存或明确放弃草稿。'); this.install(await this.api.create(title)) }) }
  async open(id: string) { await this.run(async () => { if (this.dirty) throw new Error('请先保存或明确放弃草稿。'); this.install(await this.api.get(id)) }) }
  edit(chart: Chart) {
    if (this.busy || !this.project || this.conflict) throw new Error('当前不能编辑，请先处理冲突。')
    if (!chart || chart.schema_version !== 1 || !Array.isArray(chart.events) || !chart.timing_revision) throw new Error('无效谱面草稿。')
    this.draft = structuredClone(chart); this.dirty = true
  }
  discard() { if (!this.busy && this.project) this.install(this.project) }
  async save() {
    await this.run(async () => {
      if (!this.project || !this.draft || !this.dirty || this.conflict) return
      const chart = { ...structuredClone(this.draft), revision_id: crypto.randomUUID() }
      try {
        const result = await this.api.save(this.project, chart)
        this.install({ ...this.project, head_revision: result.revision_id, chart })
      } catch (error) {
        if (error instanceof ApiError && error.status === 409) this.conflict = true
        throw error
      }
    })
  }
  async compare() { await this.run(async () => { if (this.project && this.conflict) this.remote = await this.api.get(this.project.project_id) }) }
  useRemote() { if (this.remote && !this.busy) this.install(this.remote) }
  merge(chart: Chart) {
    if (this.busy || !this.remote) throw new Error('请先获取远端版本并审阅差异。')
    if (!chart || chart.schema_version !== 1 || !Array.isArray(chart.events) || !chart.timing_revision) throw new Error('无效谱面草稿。')
    // Explicit user-authored merge becomes a draft against the fetched head; CAS still applies.
    this.project = structuredClone(this.remote); this.remote = null; this.conflict = false
    this.edit(chart)
  }
}
