"""Bounded route metrics and JSON request logs, without identities or payloads."""
import json
import logging
import time
from uuid import uuid4
from prometheus_client import CollectorRegistry, Counter, Histogram


class RequestMonitor:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter('recsys_http_requests', 'Completed HTTP requests',
                                ['method', 'route', 'status'], registry=self.registry)
        self.latency = Histogram('recsys_http_request_duration_seconds', 'Time through response headers',
                                 ['method', 'route'], buckets=(.005,.01,.025,.05,.1,.25,.5,1,2.5,5), registry=self.registry)
        self.log = logging.getLogger('recsyslab.requests')
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.log.addHandler(handler)
        self.log.setLevel(logging.INFO)
        self.log.propagate = False

    async def __call__(self, request, call_next):
        started = time.perf_counter()
        status = 500
        request_id = uuid4().hex
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers['X-Request-ID'] = request_id
            return response
        finally:
            route = getattr(request.scope.get('route'), 'path', 'unmatched')
            method = request.method if request.method in {'GET','POST','PUT','PATCH','DELETE','HEAD','OPTIONS'} else 'OTHER'
            elapsed = time.perf_counter() - started
            self.requests.labels(method, route, str(status)).inc()
            self.latency.labels(method, route).observe(elapsed)
            self.log.info(json.dumps({'event':'http_request', 'request_id':request_id,
                                      'method':method, 'route':route, 'status':status,
                                      'duration_ms':round(elapsed*1000,3)}))
