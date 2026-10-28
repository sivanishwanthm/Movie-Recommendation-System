"""Operator-only immutable definition registration and live aggregate export."""
import argparse
import json
import os
from pathlib import Path
from sqlalchemy.orm import Session
from ..serving.database import connect, check_schema, now
from .design import ExperimentSpec
from .service import register, read_spec, analyze
from .statistics import sample_size


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    define=sub.add_parser('register');define.add_argument('--config',type=Path,required=True)
    report=sub.add_parser('analyze');report.add_argument('--key',required=True);report.add_argument('--output',type=Path,required=True)
    plan=sub.add_parser('plan');plan.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        spec=ExperimentSpec.model_validate_json(args.config.read_text())
        print(json.dumps(sample_size(spec.baseline_rate,spec.minimum_detectable_effect,spec.power,spec.alpha,spec.treatment_probability),indent=2))
        return
    engine=connect(os.environ['DATABASE_URL']);check_schema(engine)
    with Session(engine) as db:
        if args.command=='register':
            spec=ExperimentSpec.model_validate_json(args.config.read_text());register(db,spec);db.commit()
            print(f'Registered {spec.key}; not activated. Set ACTIVE_EXPERIMENT explicitly to enroll live profiles.')
        else:
            result=analyze(db,read_spec(db,args.key),now())
            with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
            print(f'Exported labeled live aggregates to {args.output}')
    engine.dispose()

if __name__=='__main__':main()
