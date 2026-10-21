"""Persistent automatic projects, immutable attempts and one bounded worker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from uuid import uuid4

from mdac2.application.studio_metadata import clean,SCENES,MetadataLLM,STRATEGY_LEVELS,package_filename,cover_prompt
from mdac2.data_preparation.audio_features import sha
from mdac2.infrastructure.project_repository import RevisionConflict
from mdac2.application.studio_models import (STRATEGIES,
    pipeline_for, strategy_manifest_for, weight_manifest_for)
from mdac2.application.studio_scenes import AVAILABLE_SCENES


class StudioWorkflow:
    def __init__(self,root,runtime,repository):
        self.root=Path(root);self.runtime=Path(runtime);self.repository=repository
        self.directory=self.runtime/'studio';self.directory.mkdir(parents=True,exist_ok=True)
        self.llm=MetadataLLM(self.runtime/'llm-settings.json')
        self.stop=threading.Event();self.thread=None;self.process=None;self.lease=None
        self.preferences_lock=threading.RLock();self.folder_dialog_lock=threading.Lock()
        self.timing_lock=threading.Lock();self.timing_project=None
        with repository._connection() as c:
            c.execute('CREATE TABLE IF NOT EXISTS studio_projects(project_id TEXT PRIMARY KEY REFERENCES projects(project_id), data TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS studio_attempts(attempt_id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(project_id),data TEXT NOT NULL)')
            c.commit()
    def _read(self,c,pid):
        row=c.execute('SELECT data FROM studio_projects WHERE project_id=?',(pid,)).fetchone()
        if row is None:raise KeyError(pid)
        data=self._project_data(row[0])
        return data
    @staticmethod
    def _project_data(raw):
        data=json.loads(raw)
        # Older releases called the strategy 'model'. Adapt the read view;
        # loading a project must not rewrite its stored settings or selection.
        for key in ('metadata','draft'):
            fields=data.get(key)
            if fields and 'strategy_id' not in fields and fields.get('model') in STRATEGIES:
                fields['strategy_id']=fields.pop('model')
        return data
    def _write(self,c,data):
        c.execute('UPDATE studio_projects SET data=? WHERE project_id=?',(json.dumps(data),data['project_id']))
    def list(self):
        with self.repository._connection() as c:
            data=[self._project_data(r[0]) for r in c.execute('SELECT data FROM studio_projects')]
            return sorted(data,key=lambda d:d['updated'],reverse=True)
    def _project_folder(self,pid):
        from uuid import UUID
        from mdac2.infrastructure.artifact_store import _plain
        if str(UUID(pid))!=pid:raise ValueError('Invalid project identity')
        _plain(self.directory)
        base=self.directory.resolve();folder=self.directory/pid
        if folder.resolve().parent!=base:raise ValueError('Project path escapes studio')
        if folder.exists():
            _plain(folder)
            for item in folder.rglob('*'):_plain(item)
        return folder
    def delete(self,pid,version):
        if self.timing_project==pid:raise RevisionConflict('Timing analysis is running')
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
            folder=self._project_folder(pid)
            # Verify all database constraints before touching project-owned files.
            if any(c.execute(f'SELECT 1 FROM {table} WHERE project_id=?',(pid,)).fetchone() for table in ('jobs','revisions')):
                raise ValueError('此项目存在其他工作流引用，不能删除')
            c.execute('DELETE FROM studio_attempts WHERE project_id=?',(pid,))
            c.execute('DELETE FROM studio_projects WHERE project_id=?',(pid,))
            c.execute('DELETE FROM projects WHERE project_id=?',(pid,))
            if folder.exists():shutil.rmtree(folder)
            c.commit()
        return dict(ok=True,project_id=pid)
    def save_draft(self,pid,version,metadata,user_prompt=None,timing=None,trim=None):
        from mdac2.application.studio_timing import settings
        if set(metadata)!={'title','author','creator','scene','strategy_id'}:raise ValueError('歌曲信息字段不完整')
        draft={k:clean(v,empty=True) for k,v in metadata.items()}
        if draft['scene'] not in SCENES or draft['strategy_id'] not in STRATEGIES:raise ValueError('无效场景或制谱策略')
        if user_prompt is not None:user_prompt=cover_prompt(user_prompt)
        if timing is not None:timing=settings(timing)
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
            d.update(draft=draft,version=version+1,updated=time.time())
            if user_prompt is not None:d['cover_prompt']=user_prompt
            if timing is not None:d['timing_draft']=timing
            if trim is not None:
                from mdac2.application.studio_trim import settings as trim_settings
                interval=trim_settings(trim,d['prepared_audio']['duration'])
                if timing is None:
                    from mdac2.application.studio_timing import rebase
                    old=trim_settings(d.get('trim_draft') or d.get('trim'),d['prepared_audio']['duration'])
                    d['timing_draft']=rebase(d.get('timing_draft') or d.get('timing'),interval['start_ms']-old['start_ms'])
                d['trim_draft']=interval
            self._write(c,d);c.commit()
        return self.get(pid)
    def preferences(self,language=None,delivery_folder=None):
        path=self.runtime/'preferences.json'
        with self.preferences_lock:
            data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(language='zh')
            data.setdefault('delivery_folder','')
            if language is not None:
                if language not in ('zh','en','ja'):raise ValueError('Unsupported language')
                data['language']=language
            if delivery_folder is not None:
                folder=Path(delivery_folder)
                if not folder.is_absolute() or not folder.is_dir():raise ValueError('请选择已存在的游戏目录')
                data['delivery_folder']=str(folder.resolve())
            if language is not None or delivery_folder is not None:
                temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data),encoding='utf-8');os.replace(temp,path)
            return data
    def select_delivery_folder(self):
        from mdac2.application.studio_folder import choose_folder
        if not self.folder_dialog_lock.acquire(blocking=False):raise ValueError('目录选择窗口已打开')
        try:
            chosen=choose_folder()
            return dict(cancelled=chosen is None,preferences=self.preferences(delivery_folder=chosen))
        finally:self.folder_dialog_lock.release()
    def get(self,pid):
        with self.repository._connection() as c:
            d=self._read(c,pid)
            d['attempts']=[json.loads(r[0]) for r in c.execute('SELECT data FROM studio_attempts WHERE project_id=?',(pid,))]
        for a in d['attempts']:
            p=self.directory/pid/a['attempt_id']/'progress.json'
            if p.exists():
                try:a['progress']=json.loads(p.read_text(encoding='utf-8'))
                except (OSError,ValueError):pass
        if d.get('generation'):
            try:d['difficulty_prediction']=self._difficulty(d)
            except (OSError,ValueError,KeyError):d['difficulty_prediction']=None
        return d
    def _difficulty(self,d):
        from mdac2.application.studio_difficulty import estimate
        folder=self.directory/d['project_id']/d['generation']
        if not (folder/'result.json').is_file() or not (folder/'chart.json').is_file():
            raise ValueError('无法读取完整谱面，请重新制谱')
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        digest=sha(folder/'chart.json')
        if result.get('status')!='succeeded' or digest!=result['chart_sha256']:
            raise ValueError('谱面校验失败，请重新制谱')
        prediction=estimate(json.loads((folder/'chart.json').read_text(encoding='utf-8')))
        return dict(prediction,generation=d['generation'],chart_sha256=digest)
    def confirm_difficulty(self,pid,version,value):
        from mdac2.application.studio_difficulty import level
        value=level(value)
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
            if not d.get('generation') or d.get('draft'):raise ValueError('请先完成制谱')
            prediction=self._difficulty(d)
            d.update(difficulty=dict(prediction,value=value,predicted_value=prediction['value']),export=None,
                version=version+1,updated=time.time())
            self._write(c,d);c.commit()
        return self.get(pid)
    def models(self):
        return [dict(strategy_id=sid,label=spec['label'],difficulty=STRATEGY_LEVELS[sid],
                     weight_model=spec['weight_model'],
                     available=(self.root/weight_manifest_for(spec['weight_model'])).is_file() and
                               (self.root/strategy_manifest_for(sid)).is_file(),
                     default=sid=='v4') for sid,spec in STRATEGIES.items()]
    def import_audio(self,path,filename):
        from mdac2.application.studio_media import EXTENSIONS,prepare
        filename=clean(filename);suffix=Path(filename).suffix.lower()
        if '/' in filename or chr(92) in filename or suffix not in EXTENSIONS:raise ValueError('不支持的音频或视频格式')
        pid=str(uuid4());dest=self.directory/pid;dest.mkdir()
        source=dest/('source'+suffix)
        try:
            shutil.copyfile(path,source)
            prepared=prepare(self.root,source,dest/'audio')
        except Exception:
            shutil.rmtree(self._project_folder(pid));raise
        title=Path(filename).stem
        d=dict(project_id=pid,version=1,filename=filename,source=str(source.resolve()),source_sha256=sha(source),
            prepared_audio=prepared,metadata=dict(title=title,author='',creator='mikee',scene='scene_01',strategy_id='v4'),
            confirmed=False,metadata_source='filename',suggestion=None,active_attempt=None,generation=None,export=None,
            created=time.time(),updated=time.time())
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');c.execute('INSERT INTO projects(project_id,title) VALUES (?,?)',(pid,title))
            c.execute('INSERT INTO studio_projects VALUES (?,?)',(pid,json.dumps(d)));c.commit()
        return d
    def generate_cover(self,pid,version,metadata,force=False,mode='llm',user_prompt=''):
        from mdac2.application.studio_imagegen import generate,fallback
        from mdac2.application.studio_cover import normalize
        d=self.get(pid)
        if mode not in ('local','llm'):raise ValueError('请选择本地或LLM生成')
        user_prompt=cover_prompt(user_prompt)
        title=clean(metadata.get('title'));artist=clean(metadata.get('author',''),empty=True)
        song=dict(title=title,author=artist);old=d.get('cover') or {}
        if not force and old and mode=='llm' and (old['origin'] not in ('imagegen','local_text') or old.get('song')==song and old.get('user_prompt','')==user_prompt):return d
        profile=self.llm.load('cover');detail=dict(song=song,shape='circle',user_prompt=user_prompt)
        if mode=='local':return self.save_cover(pid,version,fallback(title,artist),origin='local_text',detail=detail,
                                                 allow_project_change=True)
        try:
            data,provenance=generate(self.root,profile,title,artist,user_prompt)
            data=normalize(self.root,data);detail.update(provenance);origin='imagegen'
            self._llm_diagnostic(pid,'cover',True)
        except (ValueError,OSError) as exc:
            from mdac2.application.studio_llm_errors import ServiceError
            data=fallback(title,artist);origin='local_text'
            detail['fallback_reason']='unconfigured' if not profile['id'] else 'generation_failed'
            detail['fallback_message']=('请在设置中为封面生成选择生图模型' if not profile['id'] else
                str(exc) if isinstance(exc,ServiceError) else '生图返回结果无法用作封面，请检查接口协议或重试')
            self._llm_diagnostic(pid,'cover',False,detail['fallback_message'])
        # Cover generation is independent of the project draft.  Persist against
        # the latest project revision so edits made while the image service runs
        # are retained; a deleted project still fails at the final database read.
        return self.save_cover(pid,version,data,origin=origin,detail=detail,allow_project_change=True)
    def _llm_diagnostic(self,pid,role,success,message=''):
        # Only fixed application messages; never provider bodies, prompts or keys.
        path=self.directory/pid/('llm-'+role+'-status.json')
        with self.preferences_lock:
            if not path.parent.is_dir():return
            temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(time=time.time(),success=success,message=message),ensure_ascii=False),encoding='utf-8')
            os.replace(temp,path)
    @staticmethod
    def _cover_history(d):
        history=list(d.get('cover_history',[]))
        old=d.get('cover')
        if old and not any(c.get('id')==old.get('id') for c in history):
            old=dict(old,id=old.get('id') or 'legacy-'+old['original_sha256'])
            if not any(c['id']==old['id'] for c in history):history.append(old)
            d['cover']=old
        number=0
        for item in history:
            if item['origin']=='imagegen':
                number+=1;item.setdefault('number',number)
                if d.get('cover',{}).get('id')==item['id']:d['cover']=item
        return history
    def prepare_covers(self,pid,version):
        """Discover local embedded artwork once, without calling an image service."""
        from mdac2.application.studio_cover import embedded
        d=self.get(pid)
        if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
        if d.get('embedded_checked'):return d
        source=Path(d['source'])
        if sha(source)!=d['source_sha256']:raise ValueError('原始媒体校验失败')
        data=embedded(self.root,source)
        if data:
            return self.save_cover(pid,version,data,origin='embedded',detail=dict(shape='circle'),select=False)
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
            d.update(cover_history=self._cover_history(d),embedded_checked=True,version=version+1,updated=time.time())
            self._write(c,d);c.commit()
        return self.get(pid)
    def select_cover(self,pid,version,cover_id):
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
            history=self._cover_history(d)
            cover=next((item for item in history if item['id']==cover_id),None)
            if cover is None:raise ValueError('此封面不存在，请重新载入项目')
            for key,digest in [('path','sha256'),('original_path','original_sha256')]:
                if sha(Path(cover[key]))!=cover[digest]:raise ValueError('封面文件校验失败')
            d.update(cover=dict(cover),cover_history=history,export=None,version=version+1,updated=time.time())
            self._write(c,d);c.commit()
        return self.get(pid)
    def save_cover(self,pid,version,data=None,*,origin='upload',crop=None,detail=None,select=True,allow_project_change=False):
        from mdac2.application.studio_cover import normalize
        import hashlib
        d=self.get(pid)
        if not allow_project_change and (d['version']!=version or d['active_attempt']):raise RevisionConflict('Project changed or busy')
        folder=self.directory/pid/'covers';folder.mkdir(exist_ok=True)
        new_image=data is not None
        if new_image:original=normalize(self.root,data)
        else:
            if not d.get('cover'):raise ValueError('请先选择封面')
            original=Path(d['cover']['original_path']).read_bytes()
            if hashlib.sha256(original).hexdigest()!=d['cover']['original_sha256']:raise ValueError('封面校验失败')
            origin=d['cover']['origin']
            detail={k:d['cover'][k] for k in ('song','shape','profile_id','model','style','prompt_sha256','fallback_reason','fallback_message','user_prompt') if k in d['cover']}
        crop=crop or dict(x=.5,y=.5,zoom=1)
        rendered=normalize(self.root,original,crop)
        # Every newly saved cover has a transparent circular render. Keep the
        # original intact for future crops, including legacy uploaded artwork.
        from mdac2.application.studio_imagegen import circle
        detail=dict(detail or {},shape='circle')
        rendered=circle(rendered)
        def publish(content):
            digest=hashlib.sha256(content).hexdigest();p=folder/(digest+'.png')
            if not p.exists():p.write_bytes(content)
            return str(p.resolve()),digest
        original_path,original_sha=publish(original);path,digest=publish(rendered)
        history=self._cover_history(d)
        cover=dict(id=str(uuid4()) if new_image else d['cover']['id'],created=time.time() if new_image else d['cover'].get('created',time.time()),
            path=path,sha256=digest,original_path=original_path,original_sha256=original_sha,origin=origin,crop=crop)| (detail or {})
        if origin=='imagegen':cover['number']=sum(x['origin']=='imagegen' for x in history)+1 if new_image else d['cover']['number']
        if new_image:history.append(cover)
        else:history=[cover if item['id']==cover['id'] else item for item in history]
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if not allow_project_change and (d['version']!=version or d['active_attempt']):raise RevisionConflict('Project changed during cover update')
            d.update(cover_history=history,version=version+1,updated=time.time())
            if select:d.update(cover=cover,export=None)
            if origin=='embedded':d['embedded_checked']=True
            self._write(c,d);c.commit()
        return self.get(pid)
    def cover_path(self,pid,original=False):
        d=self.get(pid);cover=d.get('cover')
        if not cover:raise KeyError('cover')
        path=Path(cover['original_path' if original else 'path'])
        if sha(path)!=cover['original_sha256' if original else 'sha256']:raise ValueError('封面校验失败')
        return path
    def audio_path(self,pid):
        d=self.get(pid);prepared=d.get('prepared_audio')
        if not prepared:raise ValueError('此旧项目没有导入音轨，请重新导入')
        folder=Path(prepared['path']);record=folder/'audio.json'
        if sha(record)!=prepared['record_sha256']:raise ValueError('音频记录校验失败')
        info=json.loads(record.read_text(encoding='utf-8'));path=folder/'mix.wav'
        if sha(path)!=info['wav_sha256']:raise ValueError('音频校验失败')
        return path,info['wav_sha256']
    def detect_trim(self,pid):
        if not self.timing_lock.acquire(blocking=False):raise ValueError('已有音频分析正在运行')
        self.timing_project=pid
        try:
            path,digest=self.audio_path(pid)
            folder=self.directory/pid/'trim-analysis'/str(uuid4());folder.mkdir(parents=True)
            import sys
            try:
                with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
                    subprocess.run([sys.executable,str(self.root/'tools/studio_detect_trim.py'),'--audio',str(path),'--sha',digest,'--output',str(folder/'result.json')],
                        cwd=self.root,stdout=stdout,stderr=stderr,timeout=120,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except (OSError,subprocess.SubprocessError):raise ValueError('自动裁剪分析未完成，诊断已保留；可手动设置出点') from None
            result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if sha(path)!=digest:raise ValueError('音轨身份已变化')
            return result
        finally:self.timing_project=None;self.timing_lock.release()
    def detect_timing(self,pid,version):
        from mdac2.infrastructure.timing_models import TimingModels
        d=self.get(pid)
        if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
        if not self.timing_lock.acquire(blocking=False):raise ValueError('已有测拍任务正在运行')
        self.timing_project=pid
        try:
            try:python,_,_=TimingModels(self.root).configuration('final0')
            except (ValueError,OSError,KeyError,StopIteration):raise ValueError('本地测拍模型不可用，请手打测拍或手动填写') from None
            audio,digest=self.audio_path(pid)
            folder=self.directory/pid/'timing'/str(uuid4());folder.mkdir(parents=True)
            output=folder/'result.json'
            with (folder/'worker.log').open('w',encoding='utf-8') as log:
                try:
                    subprocess.run([str(python),str(self.root/'tools/studio_detect_timing.py'),
                        '--audio',str(audio),'--sha',digest,'--output',str(output)],cwd=self.root,stdout=log,stderr=log,
                        timeout=180,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUTF8':'1'})
                except subprocess.TimeoutExpired:raise ValueError('自动测拍超时，请使用手打测拍或手动填写') from None
                except subprocess.CalledProcessError:raise ValueError('自动测拍失败或节拍不足，请使用手打测拍；诊断日志已保留') from None
            result=json.loads(output.read_text(encoding='utf-8'))
            with self.repository._connection() as c:
                c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
                if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed during timing analysis')
                d.update(timing_analysis=result,version=version+1,updated=time.time());self._write(c,d);c.commit()
            return self.get(pid)
        finally:self.timing_project=None;self.timing_lock.release()
    def confirm(self,pid,version,metadata,timing=None,trim=None):
        from mdac2.application.studio_timing import settings
        if set(metadata)!={'title','author','creator','scene','strategy_id'}:raise ValueError('歌曲信息字段不完整')
        meta={k:clean(v,empty=k=='author') for k,v in metadata.items()}
        if meta['scene'] not in AVAILABLE_SCENES or meta['strategy_id'] not in STRATEGIES:raise ValueError('无效场景或制谱策略，特殊玩法场景暂不支持')
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
            supplied_timing=timing is not None
            timing=settings(timing if supplied_timing else d.get('timing'))
            timing['subdivisions']=12
            from mdac2.application.studio_trim import settings as trim_settings
            duration=d.get('prepared_audio',{}).get('duration',600)
            interval=trim_settings(trim if trim is not None else d.get('trim'),duration)
            if not supplied_timing:
                from mdac2.application.studio_timing import rebase
                timing=rebase(timing,interval['start_ms']-trim_settings(d.get('trim'),duration)['start_ms'])
            if interval!=trim_settings(d.get('trim'),duration) or meta!=d['metadata'] or timing!=settings(d.get('timing')):d.update(generation=None,export=None,difficulty=None)
            d.update(metadata=meta,timing=timing,timing_draft=None,trim=interval,trim_draft=None,draft=None,confirmed=True,version=version+1,updated=time.time())
            self._write(c,d);c.execute('UPDATE projects SET title=? WHERE project_id=?',(meta['title'],pid));c.commit()
        return d
    def suggest(self,pid,version):
        d=self.get(pid)
        if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed or busy')
        try:suggestion=self.llm.suggest(Path(d['filename']).stem)
        except ValueError as exc:
            self._llm_diagnostic(pid,'scene',False,str(exc))
            raise
        self._llm_diagnostic(pid,'scene',True)
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if d['version']!=version or d['active_attempt']:raise RevisionConflict('Project changed during LLM call')
            d.update(suggestion=suggestion,version=version+1,updated=time.time())
            # Suggestions are persisted separately, never overwrite confirmed edits.
            self._write(c,d);c.commit()
        return d
    def enqueue(self,pid,version,operation):
        if operation not in ('generate','export'):raise ValueError('不支持的操作')
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,pid)
            if version!=d['version'] or d['active_attempt']:raise RevisionConflict('Project changed or already running')
            if not d['confirmed'] or d.get('draft'):raise ValueError('请先确认歌曲信息')
            if operation=='export' and not d['generation']:raise ValueError('请先完成制谱')
            if operation=='export':
                prediction=self._difficulty(d);rating=d.get('difficulty') or {}
                if any(rating.get(k)!=prediction[k] for k in ('generation','chart_sha256','algorithm')):
                    raise ValueError('请先确认当前谱面的定级')
            from mdac2.application.studio_timing import settings
            timing=settings(d.get('timing'))
            if operation=='generate' and timing['enabled'] and timing['subdivisions']!=12:
                raise ValueError('请重新确认BPM设置，新的制谱使用每拍12等分')
            attempt=str(uuid4());out=self.directory/pid/attempt;out.mkdir()
            request=dict(operation=operation,metadata=d['metadata'],source=d['source'],source_sha256=d['source_sha256'],
                cache=str((self.directory/'cache'/(d['source_sha256'] + ('-cpu' if os.environ.get('MDAC2_DEVICE','cuda')=='cpu' else '') )).resolve()),
                gpu_lock=str((self.root/'runtime/studio-gpu.lock').resolve()),
                strategy_id=d['metadata']['strategy_id'],
                weight_model=STRATEGIES[d['metadata']['strategy_id']]['weight_model'],
                weight_manifest_sha256=sha(self.root/weight_manifest_for(STRATEGIES[d['metadata']['strategy_id']]['weight_model'])),
                strategy_pipeline=pipeline_for(d['metadata']['strategy_id']),
                strategy_manifest_sha256=sha(self.root/strategy_manifest_for(d['metadata']['strategy_id'])),
                device=os.environ.get('MDAC2_DEVICE','cuda'))
            request['timing']=timing
            request['trim']=d.get('trim')
            if d.get('trim') and d.get('prepared_audio'):
                from mdac2.application.studio_trim import settings as trim_settings
                if d['trim']!=trim_settings(None,d['prepared_audio']['duration']):
                    import hashlib
                    identity=hashlib.sha256(json.dumps(d['trim'],sort_keys=True).encode()).hexdigest()[:20]
                    request['cache']+='-trim-'+identity
            if d.get('prepared_audio'):request['prepared_audio']=d['prepared_audio']
            if operation=='export':
                request['generation']=str((self.directory/pid/d['generation']).resolve())
                request['cover']=d.get('cover')
                request['difficulty']=d['difficulty']
            (out/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
            a=dict(attempt_id=attempt,project_id=pid,operation=operation,state='queued',created=time.time(),message='排队等待',pid=None)
            c.execute('INSERT INTO studio_attempts VALUES (?,?,?)',(attempt,pid,json.dumps(a)))
            d.update(active_attempt=attempt,version=version+1,updated=time.time())
            self._write(c,d);c.commit()
        return a
    def finish(self,a,state,message):
        with self.repository._connection() as c:
            c.execute('BEGIN IMMEDIATE');d=self._read(c,a['project_id'])
            a.update(state=state,message=message,finished=time.time())
            c.execute('UPDATE studio_attempts SET data=? WHERE attempt_id=?',(json.dumps(a),a['attempt_id']))
            if d['active_attempt']==a['attempt_id']:
                d.update(active_attempt=None,version=d['version']+1,updated=time.time())
                if state=='succeeded':
                    if a['operation']=='generate':d.update(generation=a['attempt_id'],export=None,difficulty=None)
                    else:d['export']=a['attempt_id']
                self._write(c,d)
            c.commit()
    def cancel(self,pid):
        d=self.get(pid)
        if d['active_attempt']:(self.directory/pid/d['active_attempt']/'CANCEL').touch()
        return dict(ok=True)
    def package(self,pid):
        d=self.get(pid)
        if not d['export']:raise ValueError('尚无可导出的资源包')
        path=self.directory/pid/d['export']/'chart.mdm';report=json.loads(path.with_name('result.json').read_text(encoding='utf-8'))
        if sha(path)!=report['package_sha256']:raise ValueError('资源包校验失败，请重新导出')
        return path,d
    @staticmethod
    def package_difficulty(path):
        import zipfile
        with zipfile.ZipFile(path) as z:return json.loads(z.read('info.json').decode('utf-8'))['difficulty2']
    def package_name(self,path,d,revision=1):
        return package_filename(d['metadata'],self.package_difficulty(path),revision)
    def deliver(self,pid):
        path,d=self.package(pid)
        selected=self.preferences()['delivery_folder']
        if not selected:return dict(folder_required=True)
        folder=Path(selected)
        if not folder.is_dir():raise ValueError('已保存的游戏目录不可用，请在设置中重新选择')
        digest=sha(path)
        import tempfile
        temp=None
        try:
            for revision in range(1,1000):
                dest=folder/self.package_name(path,d,revision)
                if dest.exists():
                    # Same chart already delivered: reuse. Different chart: next vN.
                    if sha(dest)==digest:break
                    continue
                # Publish only a fully copied, verified file; never replace another chart.
                if temp is None:
                    with tempfile.NamedTemporaryFile(dir=folder,prefix='.mdac-',suffix='.tmp',delete=False) as f,path.open('rb') as src:
                        temp=Path(f.name);shutil.copyfileobj(src,f)
                    if sha(temp)!=digest:raise ValueError('交付校验失败')
                try:os.link(temp,dest);break
                except FileExistsError:
                    if sha(dest)==digest:break
            else:raise ValueError('同名导出版本过多，请清理游戏目录')
        finally:
            if temp is not None:temp.unlink(missing_ok=True)
        if sha(dest)!=digest:raise ValueError('交付校验失败')
        return dict(path=str(dest),sha256=sha(dest))
    def start(self):
        if os.name=='nt':
            import msvcrt
            self.lease=(self.directory/'runtime.lock').open('a+b')
            if self.lease.seek(0,2)==0:self.lease.write(b'0');self.lease.flush()
            self.lease.seek(0)
            try:msvcrt.locking(self.lease.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                self.lease.close();self.lease=None
                raise RuntimeError('此项目目录已有工作台服务运行') from None
        # Interrupted attempts retain artifacts and require an explicit retry.
        with self.repository._connection() as c:
            pending=[json.loads(r[0]) for r in c.execute('SELECT data FROM studio_attempts')]
        for a in pending:
            if a['state']=='running':self.finish(a,'interrupted','服务中断；保留原结果，可重试。旧进程如仍运行会由GPU锁阻止重复执行。')
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    @staticmethod
    def terminate(process):
        if process.poll() is not None:return
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
        else:process.terminate()
        process.wait(timeout=15)
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=20)
        if self.lease:self.lease.close();self.lease=None
    def loop(self):
        while not self.stop.is_set():
            with self.repository._connection() as c:
                queued=[json.loads(r[0]) for r in c.execute('SELECT data FROM studio_attempts') if json.loads(r[0])['state']=='queued']
            if not queued:self.stop.wait(.5);continue
            a=min(queued,key=lambda x:x['created']);out=self.directory/a['project_id']/a['attempt_id']
            if (out/'CANCEL').exists():self.finish(a,'cancelled','已取消');continue
            try:
                with (out/'worker.log').open('w',encoding='utf-8') as log:
                    worker_python=Path(os.environ.get('MDAC2_WORKER_PYTHON',self.root/'tools/.deps/pretraining-py313/Scripts/python.exe'))
                    process=subprocess.Popen([str(worker_python),
                        str(self.root/'tools/studio_generate.py'),'--request',str((out/'request.json').resolve()),'--output',str(out.resolve())],
                        cwd=self.root,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                        env={**os.environ,'PYTHONUTF8':'1'})
                    self.process=process;a.update(state='running',pid=process.pid,message='正在处理')
                    with self.repository._connection() as c:
                        c.execute('UPDATE studio_attempts SET data=? WHERE attempt_id=?',(json.dumps(a),a['attempt_id']));c.commit()
                    deadline=time.monotonic()+10800
                    while process.poll() is None:
                        if self.stop.is_set() or (out/'CANCEL').exists() or time.monotonic()>deadline:
                            self.terminate(process);break
                        self.stop.wait(.3)
                    if self.stop.is_set():self.finish(a,'interrupted','服务已关闭，可重试')
                    elif (out/'CANCEL').exists():self.finish(a,'cancelled','已取消，可重试')
                    elif time.monotonic()>deadline:self.finish(a,'failed','处理超时，可重试')
                    else:
                        r=json.loads((out/'result.json').read_text(encoding='utf-8')) if (out/'result.json').exists() else {}
                        ok=process.returncode==0 and r.get('status')=='succeeded'
                        self.finish(a,'succeeded' if ok else 'failed','已完成' if ok else r.get('message','处理失败，请检查本机模型环境后重试'))
            except Exception:
                if self.process and self.process.poll() is None:self.terminate(self.process)
                self.finish(a,'failed','后台任务启动或执行失败；详细诊断保留在本机任务目录')
            finally:self.process=None
