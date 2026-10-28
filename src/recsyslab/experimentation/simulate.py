"""File-only synthetic A/B scenarios and repeated null calibration. No DB writes."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
from .design import ExperimentSpec, assign
from .service import summarize
from .statistics import conversion_test, wilson


def simulate(spec: ExperimentSpec, users: int=10000, seed: int=42, treatment_effect: float=0., allocation_fault: bool=False):
    if users<2 or not 0<=spec.baseline_rate+treatment_effect<=1:
        raise ValueError('invalid simulation parameters')
    rng=np.random.default_rng(seed)
    assigned_at=spec.starts_at+timedelta(seconds=1)
    records=[]
    for i in range(users):
        identifier=f'synthetic:{seed}:{i}'
        variant=assign(spec,identifier)
        if allocation_fault:variant='treatment' if rng.random()<.8 else 'control'
        rate=spec.baseline_rate+(treatment_effect if variant=='treatment' else 0)
        converted=bool(rng.random()<rate)
        items=rng.choice(spec.catalog_size,size=min(12,spec.catalog_size),replace=False).tolist()
        records.append({'id':identifier,'variant':variant,'assigned_at':assigned_at,
                        'window_ends_at':assigned_at+timedelta(hours=spec.window_hours),
                        'converted':converted,'exposed':converted or bool(rng.random()<.8),
                        'attempts':[{'status_code':200,'latency_ms':max(.1,float(rng.normal(15,3))),'items':items}]})
    result=summarize(spec,records,spec.ends_at+timedelta(hours=spec.window_hours+1),source='synthetic_simulation')
    result['simulation_inputs']={'users':users,'seed':seed,'control_conversion_probability':spec.baseline_rate,
                                 'treatment_conversion_probability':spec.baseline_rate+treatment_effect,
                                 'deliberate_assignment_corruption':allocation_fault,
                                 'latency_model':'independent Normal(15,3) ms, clipped at 0.1; one successful request per simulated profile'}
    return result


def null_calibration(repetitions=500, per_arm=1000, baseline=.1, seed=31415):
    if repetitions<1 or per_arm<1 or not 0<baseline<1:raise ValueError('invalid null simulation')
    rng=np.random.default_rng(seed)
    rejected=0
    for _ in range(repetitions):
        c,t=map(int,rng.binomial(per_arm,baseline,size=2))
        rejected+=conversion_test(c,per_arm,t,per_arm)['p_value']<.05
    return {'source':'synthetic_simulation','repetitions':repetitions,'users_per_arm':per_arm,
            'null_conversion_rate':baseline,'seed':seed,'nominal_alpha':.05,'rejections':rejected,
            'false_positive_fraction':rejected/repetitions,'monte_carlo_ci95':wilson(rejected,repetitions),
            'method':'independent equal-rate Bernoulli experiments; two-sided Fisher exact at 0.05',
            'note':'Monte Carlo diagnostic, not evidence of live experiment validity or sequential-testing safety.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--users',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    spec=ExperimentSpec(name='synthetic-calibration',version=1,salt='synthetic-only-fixed-salt',
                        starts_at=datetime(2026,1,1,tzinfo=timezone.utc),ends_at=datetime(2026,1,15,tzinfo=timezone.utc),
                        model_version='synthetic-no-real-model',catalog_size=1000)
    output={'source':'synthetic_simulation','note':'All outcomes, latencies and candidate IDs are generated. No MovieLens clicks or live engagement are claimed.',
            'scenarios':{'null':simulate(spec,args.users,args.seed),
                         'injected_effect':simulate(spec,args.users,args.seed,treatment_effect=.03),
                         'assignment_fault':simulate(spec,args.users,args.seed,allocation_fault=True)},
            'null_calibration':null_calibration()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(output,stream,indent=2,allow_nan=False)
    print(f'Wrote explicitly synthetic diagnostics to {args.output}; no database was accessed.')

if __name__=='__main__':main()
