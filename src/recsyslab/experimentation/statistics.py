"""User-level fixed-horizon inference, exact SRM and approximate design power."""
import math
import numpy as np
from scipy.stats import binomtest, fisher_exact, norm


def wilson(successes: int, total: int, alpha: float = .05):
    if not 0 <= successes <= total or total < 0 or not 0 < alpha < 1:
        raise ValueError('invalid binomial counts or alpha')
    if total == 0:
        return None
    z = float(norm.ppf(1-alpha/2))
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0.,center-radius),min(1.,center+radius)]


def conversion_test(control_success: int, control_n: int, treatment_success: int, treatment_n: int, alpha=.05):
    c_ci, t_ci = wilson(control_success,control_n,alpha), wilson(treatment_success,treatment_n,alpha)
    result = {'control': {'users':control_n,'converted':control_success,'rate':control_success/control_n if control_n else None,'ci':c_ci},
              'treatment': {'users':treatment_n,'converted':treatment_success,'rate':treatment_success/treatment_n if treatment_n else None,'ci':t_ci},
              'absolute_difference':None,'difference_ci':None,'relative_lift':None,'p_value':None,
              'method':'two-sided Fisher exact; Wilson rates; Newcombe independent-proportion difference interval', 'confidence_level':1-alpha}
    if not control_n or not treatment_n:
        return result
    pc,pt = control_success/control_n,treatment_success/treatment_n
    delta=pt-pc
    interval=[max(-1.,delta-math.sqrt((pt-t_ci[0])**2+(c_ci[1]-pc)**2)),
              min(1.,delta+math.sqrt((t_ci[1]-pt)**2+(pc-c_ci[0])**2))]
    result.update(absolute_difference=delta,difference_ci=interval,relative_lift=pt/pc-1 if pc else None,
                  p_value=float(fisher_exact([[treatment_success,treatment_n-treatment_success],
                                             [control_success,control_n-control_success]],alternative='two-sided').pvalue))
    return result


def sample_size(baseline: float, effect: float, power: float=.8, alpha: float=.05, treatment_probability: float=.5):
    if not 0 < baseline < baseline+effect < 1 or not .5 < power < 1 or not 0 < alpha < .5 or not 0 < treatment_probability < 1:
        raise ValueError('invalid power-analysis inputs')
    r=treatment_probability/(1-treatment_probability)
    pt=baseline+effect
    pooled=(baseline+r*pt)/(1+r)
    n=math.ceil((norm.ppf(1-alpha/2)*math.sqrt(pooled*(1-pooled)*(1+1/r))+
                 norm.ppf(power)*math.sqrt(baseline*(1-baseline)+pt*(1-pt)/r))**2/effect**2)
    return {'control_users':n,'treatment_users':math.ceil(r*n),'total_users':n+math.ceil(r*n),
            'baseline_rate':baseline,'absolute_mde':effect,'target_power':power,'alpha':alpha,
            'method':'normal-approximation two-proportion planning; independent profiles; no attrition adjustment'}


def srm(control_n: int, treatment_n: int, probability: float=.5, alpha: float=.001):
    if min(control_n,treatment_n)<0 or not 0<probability<1:
        raise ValueError('invalid allocation counts')
    total=control_n+treatment_n
    p=float(binomtest(treatment_n,total,probability).pvalue) if total else None
    return {'control':control_n,'treatment':treatment_n,'expected_treatment_fraction':probability,
            'p_value':p,'threshold':alpha,'mismatch':p<alpha if p is not None else False,
            'method':'two-sided exact binomial allocation test on all enrolled profiles'}


def mean_difference(control, treatment, seed=42, samples=1000):
    """Resample profiles, never individual requests, for repeated-request guardrails."""
    c=np.asarray(control,dtype=float);t=np.asarray(treatment,dtype=float)
    if not np.isfinite(c).all() or not np.isfinite(t).all() or samples<1:
        raise ValueError('invalid bootstrap inputs')
    if not len(c) or not len(t):return {'difference':None,'ci95':None,'control_users':len(c),'treatment_users':len(t)}
    rng=np.random.default_rng(seed)
    boot=[rng.choice(t,len(t),replace=True).mean()-rng.choice(c,len(c),replace=True).mean() for _ in range(samples)]
    return {'difference':float(t.mean()-c.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'control_users':len(c),'treatment_users':len(t),'method':'independent user bootstrap, 95% percentile interval'}
