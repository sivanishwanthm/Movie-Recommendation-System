"""FastAPI serving with anonymous private profiles and real interaction logging."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import time
from typing import Optional, List
from uuid import UUID, uuid4
from fastapi import FastAPI, Depends, HTTPException, Request, Response, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import Session
from .database import connect, check_schema, now, Profile, Movie, Rating, Batch, Event, Experiment, Assignment, ExperimentBatch, ExperimentAttempt
from .model import LiveRecommender
from ..experimentation.service import read_spec, enroll, expose, analyze

log=logging.getLogger('recsyslab.api')
COOKIE='recsys_session'

@dataclass
class Settings:
    database_url: str
    model_run: Path
    reports: Path
    origin: str = 'http://localhost:3000'
    secure_cookie: bool = False
    active_experiment: Optional[str] = None
    experiment_admin_token: Optional[str] = None
    release_root: Optional[Path] = None
    metrics_token: Optional[str] = None

    @classmethod
    def environment(cls):
        return cls(os.environ['DATABASE_URL'], Path(os.getenv('MODEL_RUN','artifacts/phase2-1m')),
                   Path(os.getenv('REPORTS_DIR','reports')), os.getenv('APP_ORIGIN','http://localhost:3000'),
                   os.getenv('COOKIE_SECURE','false').lower()=='true', os.getenv('ACTIVE_EXPERIMENT') or None,
                   os.getenv('EXPERIMENT_ADMIN_TOKEN') or None,
                   Path(os.environ['RELEASE_ROOT']) if os.getenv('RELEASE_ROOT') else None,
                   (Path(os.environ['METRICS_TOKEN_FILE']).read_text().strip() if os.getenv('METRICS_TOKEN_FILE')
                    else os.getenv('METRICS_TOKEN')) or None)

class NameInput(BaseModel):
    name: str = Field(min_length=1,max_length=60)
    @field_validator('name')
    @classmethod
    def clean_name(cls,value):
        if not value.strip(): raise ValueError('enter a profile name')
        return value.strip()

class RecommendationInput(BaseModel):
    k: int = Field(default=12,ge=1,le=50,strict=True)

class RatingInput(BaseModel):
    event_id: UUID
    value: float = Field(ge=.5,le=5,allow_inf_nan=False)
    @field_validator('value')
    @classmethod
    def half_stars(cls,value):
        if value*2 != int(value*2):raise ValueError('use half-star increments')
        return value

class ImpressionInput(BaseModel):
    batch_id: UUID
    movie_ids: List[int] = Field(min_length=1,max_length=50)

class ClickInput(BaseModel):
    event_id: UUID
    batch_id: UUID
    movie_id: int = Field(gt=0)


def create_app(settings: Optional[Settings]=None) -> FastAPI:
    settings=settings or Settings.environment()
    engine=connect(settings.database_url)

    @asynccontextmanager
    async def lifespan(app):
        check_schema(engine)
        from ..operations.releases import selected_path
        model_path = selected_path(settings.release_root) if settings.release_root else settings.model_run
        app.state.recommender=LiveRecommender(model_path)
        app.state.release_id=model_path.name if settings.release_root else None
        with Session(engine) as db:
            ids=set(db.scalars(select(Movie.id)).all())
            if not app.state.recommender.available.issubset(ids):
                raise ValueError('movie metadata is missing model catalog IDs; run catalog import')
        app.state.experiment=None
        if settings.active_experiment:
            with Session(engine) as db:
                spec=read_spec(db,settings.active_experiment)
                if spec.model_version!=app.state.recommender.version or spec.catalog_size!=len(app.state.recommender.available):
                    raise ValueError('experiment model/catalog does not match the loaded artifact')
                app.state.experiment=spec
        app.state.benchmarks={}
        for phase,relative in [('phase1','metrics.json'),('phase2','phase2/metrics.json'),('phase3','phase3/metrics.json')]:
            path=settings.reports/relative
            if path.exists():app.state.benchmarks[phase]=json.loads(path.read_text())
        yield
        engine.dispose()

    app=FastAPI(title='RecSysLab API',version='0.6.0',lifespan=lifespan)
    app.state.engine=engine

    @app.middleware('http')
    async def request_guard(request,call_next):
        if request.method in {'POST','PUT','PATCH','DELETE'}:
            origin=request.headers.get('origin')
            if (origin and origin != settings.origin) or request.headers.get('x-recsys-client') != 'web':
                return JSONResponse(status_code=403,content={'detail':'untrusted request origin or missing client header'})
        start=time.perf_counter()
        response=await call_next(request)
        attempt_id=getattr(request.state,'experiment_attempt_id',None)
        if attempt_id:
            elapsed=(time.perf_counter()-start)*1000
            try:
                with Session(engine) as telemetry:
                    attempt=telemetry.get(ExperimentAttempt,attempt_id)
                    attempt.status_code=response.status_code
                    attempt.api_latency_ms=elapsed
                    telemetry.commit()
            except SQLAlchemyError:
                log.error('experiment telemetry incomplete; analysis must not clear guardrails')
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        return response

    from ..operations.monitoring import RequestMonitor
    monitor = RequestMonitor()
    app.state.monitor = monitor
    app.middleware('http')(monitor)

    @app.get('/api/live')
    def live():
        return {'status': 'alive'}

    @app.get('/api/metrics', include_in_schema=False)
    def metrics(request: Request):
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        supplied = request.headers.get('authorization', '')
        if not settings.metrics_token:
            raise HTTPException(503, 'metrics disabled')
        if not secrets.compare_digest(supplied, 'Bearer ' + settings.metrics_token):
            raise HTTPException(403, 'metrics access denied')
        return Response(generate_latest(monitor.registry), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(IntegrityError)
    async def integrity_error(request,exc):
        return JSONResponse(status_code=409,content={'detail':'interaction conflicts with an existing record; retry with the original event ID'})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request,exc):
        log.error('database operation failed')
        return JSONResponse(status_code=503,content={'detail':'database unavailable; retry shortly'})

    def session():
        with Session(engine) as db:yield db

    def profile(request: Request,db: Session=Depends(session)):
        token=request.cookies.get(COOKIE)
        if not token:raise HTTPException(401,'create a profile to continue')
        value=hashlib.sha256(token.encode()).hexdigest()
        row=db.scalar(select(Profile).where(Profile.token_hash==value,Profile.expires_at>now()).with_for_update())
        if row is None:raise HTTPException(401,'profile session expired; create a new profile')
        return row

    def movie_dict(movie):
        return {'id':movie.id,'title':movie.title,'genres':movie.genres,
                'recommendable':movie.id in app.state.recommender.available}

    def owned_batch(db,p,batch_id):
        batch=db.get(Batch,str(batch_id))
        if batch is None or batch.profile_id!=p.id:raise HTTPException(404,'recommendation batch not found')
        return batch

    @app.get('/api/health')
    def health(db: Session=Depends(session)):
        db.execute(select(1))
        return {'status':'ready','model_version':app.state.recommender.version,'schema_version':2,'release_id':app.state.release_id}

    @app.post('/api/profiles',status_code=201)
    def create_profile(body: NameInput,response: Response,request: Request,db: Session=Depends(session)):
        token=request.cookies.get(COOKIE)
        if token and db.scalar(select(Profile.id).where(Profile.token_hash==hashlib.sha256(token.encode()).hexdigest(),Profile.expires_at>now())):
            raise HTTPException(409,'profile already exists; update its name instead')
        raw=secrets.token_urlsafe(32)
        row=Profile(id=str(uuid4()),name=body.name,token_hash=hashlib.sha256(raw.encode()).hexdigest(),expires_at=now()+timedelta(days=30))
        db.add(row);db.commit()
        response.set_cookie(COOKIE,raw,httponly=True,secure=settings.secure_cookie,samesite='strict',max_age=30*86400,path='/')
        return {'id':row.id,'name':row.name}

    @app.get('/api/me')
    def me(p=Depends(profile)):
        return {'id':p.id,'name':p.name,'expires_at':p.expires_at}

    @app.patch('/api/me')
    def rename(body: NameInput,p=Depends(profile),db: Session=Depends(session)):
        p.name=body.name;db.commit()
        return {'id':p.id,'name':p.name}

    @app.get('/api/movies')
    def movies(q: str=Query(default='',max_length=100),offset: int=Query(default=0,ge=0),
               limit: int=Query(default=24,ge=1,le=100),db: Session=Depends(session)):
        statement=select(Movie)
        if q.strip():statement=statement.where(Movie.title.ilike('%'+q.strip().replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%',escape='\\'))
        total=db.scalar(select(func.count()).select_from(statement.subquery()))
        records=db.scalars(statement.order_by(Movie.title,Movie.id).offset(offset).limit(limit)).all()
        return {'items':[movie_dict(m) for m in records],'total':total,'offset':offset,'limit':limit}

    @app.get('/api/movies/{movie_id}')
    def movie(movie_id: int,db: Session=Depends(session)):
        row=db.get(Movie,movie_id)
        if row is None:raise HTTPException(404,'movie not found')
        return movie_dict(row)

    @app.get('/api/ratings')
    def ratings(p=Depends(profile),db: Session=Depends(session)):
        rows=db.execute(select(Rating,Movie).join(Movie,Movie.id==Rating.movie_id).where(Rating.profile_id==p.id).order_by(Rating.updated_at.desc())).all()
        return [{'movie':movie_dict(m),'value':r.value,'updated_at':r.updated_at} for r,m in rows]

    @app.put('/api/ratings/{movie_id}')
    def rate(movie_id: int,body: RatingInput,p=Depends(profile),db: Session=Depends(session)):
        if db.get(Movie,movie_id) is None:raise HTTPException(404,'movie not found')
        old=db.get(Event,str(body.event_id))
        if old:
            if (old.profile_id,old.kind,old.movie_id,old.value)!=(p.id,'rating',movie_id,body.value):raise HTTPException(409,'event ID was already used for another interaction')
            return {'event_id':old.id,'value':old.value,'replayed':True}
        row=db.get(Rating,(p.id,movie_id))
        if row is None:db.add(Rating(profile_id=p.id,movie_id=movie_id,value=body.value))
        else:row.value=body.value;row.updated_at=now()
        db.add(Event(id=str(body.event_id),profile_id=p.id,movie_id=movie_id,kind='rating',value=body.value))
        db.commit()
        return {'event_id':str(body.event_id),'value':body.value,'replayed':False}

    @app.post('/api/recommendations')
    def recommendations(body: RecommendationInput,request: Request,p=Depends(profile),db: Session=Depends(session)):
        ratings=db.scalars(select(Rating).where(Rating.profile_id==p.id)).all()
        recommender=app.state.recommender
        positive_count=sum(r.value>=recommender.model.threshold and r.movie_id in recommender.available for r in ratings)
        assignment=None
        attempt=None
        if app.state.experiment:
            try:
                assignment=enroll(db,p.id,app.state.experiment,positive_count,now())
            except ValueError as exc:
                raise HTTPException(409,str(exc))
            if assignment:
                attempt=ExperimentAttempt(id=str(uuid4()),assignment_id=assignment.id)
                db.add(attempt);db.commit()
                # Durable enrollment/request denominator, even if variant execution fails.
                request.state.experiment_attempt_id=attempt.id
        start=time.perf_counter()
        try:
            if assignment and assignment.variant=='control':
                ids=recommender.model.baseline.recommend({r.movie_id for r in ratings},body.k)
                strategy='experiment_popularity_control'
            else:
                ids,strategy,positive_count=recommender.recommend(ratings,body.k)
        except Exception:
            log.error('recommendation scoring failed')
            raise HTTPException(503,'recommendation model unavailable')
        latency=(time.perf_counter()-start)*1000
        records={m.id:m for m in db.scalars(select(Movie).where(Movie.id.in_(ids))).all()}
        batch=Batch(id=str(uuid4()),profile_id=p.id,movie_ids=ids,model_version=recommender.version,strategy=strategy,latency_ms=latency)
        db.add(batch);db.flush()
        if assignment:
            db.add(ExperimentBatch(batch_id=batch.id,assignment_id=assignment.id))
            attempt.batch_id=batch.id
        db.commit()
        return {'batch_id':batch.id,'items':[movie_dict(records[i]) for i in ids], 'strategy':strategy,
                'positive_count':positive_count,'model_version':batch.model_version,'generation_ms':latency,
                'experiment':{'key':assignment.experiment_key,'variant':assignment.variant} if assignment else None}

    @app.post('/api/impressions')
    def impressions(body: ImpressionInput,p=Depends(profile),db: Session=Depends(session)):
        batch=owned_batch(db,p,body.batch_id)
        ids=set(body.movie_ids)
        if not ids.issubset(set(batch.movie_ids)):raise HTTPException(422,'impression contains an item outside the batch')
        added=0
        for movie_id in sorted(ids):
            key=f'impression:{batch.id}:{movie_id}'
            if not db.scalar(select(Event.id).where(Event.profile_id==p.id,Event.dedupe_key==key)):
                db.add(Event(id=str(uuid4()),profile_id=p.id,movie_id=movie_id,batch_id=batch.id,kind='impression',dedupe_key=key));added+=1
        expose(db,batch.id,now())
        db.commit()
        return {'recorded':added}

    @app.post('/api/clicks')
    def click(body: ClickInput,p=Depends(profile),db: Session=Depends(session)):
        batch=owned_batch(db,p,body.batch_id)
        if body.movie_id not in batch.movie_ids:raise HTTPException(422,'clicked item is outside the batch')
        old=db.get(Event,str(body.event_id))
        if old and (old.profile_id,old.kind,old.movie_id,old.batch_id)!=(p.id,'click',body.movie_id,batch.id):raise HTTPException(409,'event ID already used')
        impression=f'impression:{batch.id}:{body.movie_id}'
        if not db.scalar(select(Event.id).where(Event.profile_id==p.id,Event.dedupe_key==impression)):
            raise HTTPException(409,'record a visible impression before its click')
        key=f'click:{batch.id}:{body.movie_id}'
        existing=db.scalar(select(Event.id).where(Event.profile_id==p.id,Event.dedupe_key==key))
        if not existing:db.add(Event(id=str(body.event_id),profile_id=p.id,movie_id=body.movie_id,batch_id=batch.id,kind='click',dedupe_key=key))
        db.commit()
        return {'recorded':not bool(existing)}

    @app.get('/api/analytics')
    def analytics(p=Depends(profile),db: Session=Depends(session)):
        counts=dict(db.execute(select(Event.kind,func.count()).where(Event.profile_id==p.id).group_by(Event.kind)).all())
        batch_count,latency=db.execute(select(func.count(Batch.id),func.avg(Batch.latency_ms)).where(Batch.profile_id==p.id)).one()
        ratings_count=db.scalar(select(func.count()).select_from(Rating).where(Rating.profile_id==p.id))
        impressions=counts.get('impression',0);clicks=counts.get('click',0)
        return {'scope':'current_profile','source':'real_app_interactions','impressions':impressions,'clicks':clicks,
                'card_click_through_rate':clicks/impressions if impressions else None,'rating_events':counts.get('rating',0),
                'rated_movies':ratings_count,'recommendation_batches':batch_count,'mean_generation_ms':latency,
                'experiment_status':'active' if app.state.experiment else 'available_not_active','note':'Descriptive interaction counts, not randomized experiment results.'}

    @app.get('/api/benchmarks')
    def benchmarks():
        return {'source':'observational_offline_movielens_1m','phases':app.state.benchmarks,
                'live_model':app.state.recommender.version,'note':'Frozen historical profiles; these scores do not measure live-profile quality or online lift.'}

    @app.get('/api/experiments')
    def experiments(db: Session=Depends(session)):
        rows=db.scalars(select(Experiment).order_by(Experiment.created_at.desc())).all()
        return {'active_key':settings.active_experiment,'experiments':[
            {'key':row.key,'specification':read_spec(db,row.key).canonical()} for row in rows],
            'source':'live_definitions','analysis_requires_admin':True}

    @app.get('/api/experiments/assignment')
    def my_assignment(p=Depends(profile),db: Session=Depends(session)):
        if not settings.active_experiment:return {'assignment':None,'active_key':None}
        row=db.scalar(select(Assignment).where(Assignment.profile_id==p.id,Assignment.experiment_key==settings.active_experiment))
        return {'active_key':settings.active_experiment,'assignment':{
            'variant':row.variant,'assigned_at':row.assigned_at,'window_ends_at':row.window_ends_at} if row else None}

    @app.get('/api/experiments/synthetic')
    def synthetic_report():
        path=settings.reports/'phase5/simulation.json'
        if not path.exists():return {'source':'synthetic_simulation','available':False}
        result=json.loads(path.read_text())
        if result.get('source')!='synthetic_simulation':raise HTTPException(503,'synthetic report source mismatch')
        return dict(result,available=True)

    @app.get('/api/experiments/{key}/analysis')
    def experiment_analysis(key: str,request: Request,db: Session=Depends(session)):
        if not settings.experiment_admin_token:raise HTTPException(503,'operator analysis access is not configured')
        supplied=request.headers.get('x-experiment-admin','')
        if not secrets.compare_digest(supplied,settings.experiment_admin_token):raise HTTPException(403,'operator token required')
        try:spec=read_spec(db,key)
        except ValueError as exc:raise HTTPException(404,str(exc))
        return analyze(db,spec,now())

    return app
