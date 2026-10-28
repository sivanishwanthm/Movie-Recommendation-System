"""Database-backed enrollment, attribution and profile-level analysis."""
from datetime import datetime, timedelta
from uuid import uuid4
import numpy as np
from sqlalchemy import select
from ..serving.database import Experiment, Assignment, ExperimentBatch, Exposure, ExperimentAttempt, Batch, Event
from .design import ExperimentSpec, assign
from .statistics import conversion_test, sample_size, srm, mean_difference


def register(db, spec: ExperimentSpec):
    existing=db.get(Experiment,spec.key)
    if existing:
        if existing.fingerprint != spec.fingerprint() or existing.specification != spec.canonical():
            raise ValueError('experiment versions are immutable; register a new version')
        return existing
    row=Experiment(key=spec.key,specification=spec.canonical(),fingerprint=spec.fingerprint())
    db.add(row);db.flush()
    return row


def read_spec(db, key: str) -> ExperimentSpec:
    row=db.get(Experiment,key)
    if row is None:raise ValueError('experiment definition not found')
    spec=ExperimentSpec.model_validate(row.specification)
    if spec.key!=key or spec.fingerprint()!=row.fingerprint:
        raise ValueError('experiment definition integrity mismatch')
    return spec


def enroll(db, profile_id: str, spec: ExperimentSpec, positives: int, at: datetime):
    """Caller holds the profile row lock. Assignment occurs before variant execution."""
    existing=db.scalar(select(Assignment).where(Assignment.profile_id==profile_id,Assignment.experiment_key==spec.key))
    if existing:
        return existing if at < existing.window_ends_at else None
    if not spec.starts_at <= at < spec.ends_at or positives < spec.min_positives:
        return None
    # Prevent a second experiment enrolling this profile during another active follow-up.
    overlap=db.scalar(select(Assignment.id).where(Assignment.profile_id==profile_id,Assignment.window_ends_at>at))
    if overlap:raise ValueError('profile has another experiment observation window in progress')
    row=Assignment(id=str(uuid4()),profile_id=profile_id,experiment_key=spec.key,variant=assign(spec,profile_id),
                   assigned_at=at,window_ends_at=at+timedelta(hours=spec.window_hours))
    db.add(row);db.flush()
    return row


def expose(db, batch_id: str, at: datetime):
    link=db.get(ExperimentBatch,batch_id)
    if link:
        assignment=db.get(Assignment,link.assignment_id)
        if assignment.assigned_at <= at < assignment.window_ends_at and db.get(Exposure,assignment.id) is None:
            db.add(Exposure(assignment_id=assignment.id,batch_id=batch_id,exposed_at=at))


def summarize(spec: ExperimentSpec, records: list, as_of: datetime, source: str='live_randomized', samples: int=1000):
    """One row per assigned profile; all mature assignments retained (ITT)."""
    if source not in {'live_randomized','synthetic_simulation'}:raise ValueError('invalid analysis source')
    rows=[r for r in records if r['assigned_at']<=as_of]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate analysis unit')
    arms={arm:[r for r in rows if r['variant']==arm] for arm in ['control','treatment']}
    if sum(map(len,arms.values()))!=len(rows):raise ValueError('invalid variant')
    mature={arm:[r for r in group if r['window_ends_at']<=as_of] for arm,group in arms.items()}
    counts={arm:sum(bool(r['converted']) for r in group) for arm,group in mature.items()}
    primary=conversion_test(counts['control'],len(mature['control']),counts['treatment'],len(mature['treatment']),spec.alpha)
    allocation=srm(len(arms['control']),len(arms['treatment']),spec.treatment_probability,spec.srm_alpha)
    plan=sample_size(spec.baseline_rate,spec.minimum_detectable_effect,spec.power,spec.alpha,spec.treatment_probability)
    telemetry={};vectors={};coverage={}
    for arm,group in mature.items():
        latencies=[];errors=[];items=set();pending=0;attempts=0;exposed=0;missing_latency=0;users_without_attempts=0
        for row in group:
            observed=row['attempts'];attempts+=len(observed);exposed+=int(row['exposed'])
            if not observed:users_without_attempts+=1
            complete=[a for a in observed if a['status_code'] is not None]
            pending+=len(observed)-len(complete)
            measured=[a['latency_ms'] for a in complete if a['latency_ms'] is not None]
            missing_latency+=len(complete)-len(measured)
            if measured:latencies.append(float(np.mean(measured)))
            if complete:errors.append(sum(a['status_code']>=400 for a in complete)/len(complete))
            for a in complete:items.update(a['items'])
        coverage[arm]=len(items)/spec.catalog_size
        vectors[arm]={'latency':latencies,'error':errors}
        telemetry[arm]={'enrolled_users':len(arms[arm]),'mature_users':len(group),'pending_users':len(arms[arm])-len(group),
                        'exposed_mature_users':exposed,'requests':attempts,'unfinished_requests':pending,
                        'missing_latency_requests':missing_latency,'users_without_requests':users_without_attempts,
                        'catalog_coverage':coverage[arm], 'unique_recommended_items':len(items),
                        'mean_user_api_latency_ms':float(np.mean(latencies)) if latencies else None,
                        'mean_user_request_error_rate':float(np.mean(errors)) if errors else None}
    latency=mean_difference(vectors['control']['latency'],vectors['treatment']['latency'],samples=samples)
    error_test=conversion_test(sum(v>0 for v in vectors['control']['error']),len(vectors['control']['error']),
                               sum(v>0 for v in vectors['treatment']['error']),len(vectors['treatment']['error']))
    error={'difference':error_test['absolute_difference'],'ci95':error_test['difference_ci'],
           'method':'Newcombe interval for probability that a profile has any failed recommendation request'}
    coverage_delta=coverage['treatment']-coverage['control'] if all(mature.values()) else None
    guardrails={'api_latency_ms':dict(latency,maximum_delta=spec.max_latency_delta_ms),
                'user_any_request_error_probability':dict(error,maximum_delta=spec.max_error_rate_delta),
                'catalog_coverage':{'difference':coverage_delta,'minimum_delta':spec.min_catalog_coverage_delta,
                                    'method':'descriptive catalog-wide fraction; no significance test'}}
    telemetry_complete=all(not (t['unfinished_requests'] or t['missing_latency_requests'] or t['users_without_requests']) for t in telemetry.values())
    final_time=as_of>=spec.ends_at+timedelta(hours=spec.window_hours)
    powered=all(len(mature[a])>=plan[a+'_users'] for a in arms)
    reasons=[]
    if not rows:reasons.append('no_enrolled_profiles')
    if not final_time:reasons.append('fixed_horizon_not_complete')
    if not powered:reasons.append('planned_sample_size_not_reached')
    if allocation['mismatch']:reasons.append('sample_ratio_mismatch')
    if not telemetry_complete:reasons.append('incomplete_request_telemetry')
    if latency['ci95'] is None or latency['ci95'][1]>spec.max_latency_delta_ms:reasons.append('latency_guardrail_not_cleared')
    if error['ci95'] is None or error['ci95'][1]>spec.max_error_rate_delta:reasons.append('error_guardrail_not_cleared')
    if coverage_delta is None or coverage_delta<spec.min_catalog_coverage_delta:reasons.append('coverage_guardrail_not_cleared')
    if source=='synthetic_simulation':reasons.append('synthetic_data_cannot_support_online_decisions')
    return {'experiment_key':spec.key,'source':source,'as_of':as_of.isoformat(),
            'primary_metric':f'any attributed recommendation click within {spec.window_hours} hours of enrollment',
            'analysis_unit':'browser profile; mature enrolled profiles (intention-to-treat)',
            'primary':primary,'srm':allocation,'plan':plan,'variants':telemetry,'guardrails':guardrails,
            'review_eligible':not reasons,'review_blockers':reasons,
            'interpretation':'No automatic winner. Interim p-values are descriptive; fixed-horizon review and guardrail checks are required.'}


def analyze(db, spec: ExperimentSpec, as_of: datetime):
    assignments=db.scalars(select(Assignment).where(Assignment.experiment_key==spec.key,Assignment.assigned_at<=as_of)).all()
    ids=[a.id for a in assignments]
    if not ids:return summarize(spec,[],as_of)
    attempts=db.scalars(select(ExperimentAttempt).where(ExperimentAttempt.assignment_id.in_(ids),ExperimentAttempt.created_at<=as_of)).all()
    by_assignment={a.id:[] for a in assignments};assignment_map={a.id:a for a in assignments}
    batch_ids=[a.batch_id for a in attempts if a.batch_id]
    batches={b.id:b for b in db.scalars(select(Batch).where(Batch.id.in_(batch_ids))).all()} if batch_ids else {}
    for attempt in attempts:
        assignment=assignment_map[attempt.assignment_id]
        if assignment.assigned_at<=attempt.created_at<assignment.window_ends_at:
            by_assignment[assignment.id].append({'status_code':attempt.status_code,'latency_ms':attempt.api_latency_ms,
                                                'items':batches[attempt.batch_id].movie_ids if attempt.batch_id in batches else []})
    clicks=db.execute(select(ExperimentBatch.assignment_id,Event.created_at).join(Event,Event.batch_id==ExperimentBatch.batch_id)
                      .where(ExperimentBatch.assignment_id.in_(ids),Event.kind=='click',Event.created_at<=as_of)).all()
    converted={aid for aid,at in clicks if assignment_map[aid].assigned_at<=at<assignment_map[aid].window_ends_at}
    exposures={e.assignment_id for e in db.scalars(select(Exposure).where(Exposure.assignment_id.in_(ids),Exposure.exposed_at<=as_of)).all()}
    records=[{'id':a.id,'variant':a.variant,'assigned_at':a.assigned_at,'window_ends_at':a.window_ends_at,
              'converted':a.id in converted,'exposed':a.id in exposures,'attempts':by_assignment[a.id]} for a in assignments]
    return summarize(spec,records,as_of)
