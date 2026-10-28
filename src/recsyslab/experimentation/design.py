"""Immutable pre-registration contract and deterministic profile-level assignment."""
from datetime import datetime
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str = Field(pattern=r'^[a-z][a-z0-9_-]{2,50}$')
    version: int = Field(ge=1, strict=True)
    salt: str = Field(min_length=16, max_length=128)
    starts_at: datetime
    ends_at: datetime
    control: Literal['popularity'] = 'popularity'
    treatment: Literal['item_cf'] = 'item_cf'
    model_version: str = Field(min_length=5, max_length=80)
    catalog_size: int = Field(ge=1, strict=True)
    min_positives: int = Field(default=3, ge=3, strict=True)
    window_hours: int = Field(default=24, ge=1, le=168, strict=True)
    treatment_probability: float = Field(default=.5, ge=.1, le=.9, allow_inf_nan=False)
    baseline_rate: float = Field(default=.1, gt=0, lt=1, allow_inf_nan=False)
    minimum_detectable_effect: float = Field(default=.02, gt=0, lt=1, allow_inf_nan=False)
    alpha: float = Field(default=.05, gt=0, lt=.5, allow_inf_nan=False)
    power: float = Field(default=.8, gt=.5, lt=1, allow_inf_nan=False)
    srm_alpha: float = Field(default=.001, gt=0, lt=.05, allow_inf_nan=False)
    max_latency_delta_ms: float = Field(default=20., ge=0, allow_inf_nan=False)
    max_error_rate_delta: float = Field(default=.01, ge=0, le=1, allow_inf_nan=False)
    min_catalog_coverage_delta: float = Field(default=-.05, ge=-1, le=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def check_design(self):
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError('experiment timestamps must include UTC offsets')
        if self.ends_at <= self.starts_at:
            raise ValueError('enrollment end must follow enrollment start')
        if self.baseline_rate + self.minimum_detectable_effect >= 1:
            raise ValueError('baseline plus absolute effect must be below one')
        return self

    @property
    def key(self) -> str:
        return f'{self.name}:v{self.version}'

    def canonical(self) -> dict:
        return self.model_dump(mode='json')

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True).encode()).hexdigest()


def assign(spec: ExperimentSpec, profile_id: str) -> str:
    material = json.dumps([spec.key, spec.salt, str(profile_id)], separators=(',', ':')).encode()
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') / 2**64
    return 'treatment' if bucket < spec.treatment_probability else 'control'
