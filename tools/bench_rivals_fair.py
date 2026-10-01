#!/usr/bin/env python3
"""Optional framework rivals; CPU and GPU timing scopes stay separate."""
import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='benchmarks/results/rivals_fair.json')
parser.add_argument('--repeats', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--warmup', type=int, default=10)
args = parser.parse_args()
if min(args.repeats, args.batch, args.warmup) < 1:
    parser.error('repeat, batch and warmup counts must be positive')
shapes = [(16,16,16),(32,32,32),(64,64,64),(128,128,128),(256,256,256),
          (512,512,512),(1024,1024,1024),(2048,2048,2048),(31,47,19),
          (127,257,63),(64,1024,512),(1024,64,512),(1,1024,4096),(1024,1,4096)]
results, skipped, versions = [], [], {}

def samples(fn, sync=lambda: None, event_factory=None, elapsed=None):
    for _ in range(args.warmup): fn()
    sync()
    wall, device = [], []
    for _ in range(args.repeats):
        sync()
        start, stop = (event_factory(), event_factory()) if event_factory else (None, None)
        t0 = time.perf_counter()
        if start: start.record()
        for _ in range(args.batch if event_factory else 1): fn()
        if stop: stop.record()
        sync()
        wall.append((time.perf_counter()-t0)*1000/(args.batch if event_factory else 1))
        if start: device.append((elapsed(start,stop) if elapsed else start.elapsed_time(stop))/args.batch)
    return wall, device

def append(engine, policy, shape, wall, device, error):
    row = dict(engine=engine, policy=policy, m=shape[0], n=shape[1], k=shape[2],
               timing='preallocated CUDA stream elapsed' if device else 'CPU preallocated wall clock',
               wall_median_ms=statistics.median(wall), wall_samples_ms=wall,
               max_abs_error_sampled=float(error))
    if device:
        row.update(device_median_ms=statistics.median(device), device_samples_ms=device)
    results.append(row)
    print(engine, policy, shape, round(row['wall_median_ms'], 6), 'ms', flush=True)

try:
    np = importlib.import_module('numpy')
except ImportError:
    raise SystemExit('NumPy is required for independent double-precision validation')
versions['numpy'] = np.__version__
try:
    from threadpoolctl import threadpool_info
    versions['cpu_threadpools'] = threadpool_info()
except ImportError:
    versions['numpy_config'] = str(np.__config__.show(mode='dicts')) if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else 'see numpy.show_config()'

def error_check(a,b,c,tf32=False):
    indexes = np.linspace(0,c.size-1,min(97,c.size),dtype=int)
    error=0.0
    for index in indexes:
        i,j=divmod(int(index),c.shape[1])
        products=a[i,:].astype(np.float64)*b[:,j].astype(np.float64)
        diff=abs(float(c[i,j])-products.sum())
        if not np.isfinite(c[i,j]) or diff>1e-5+(0.002 if tf32 else 2e-6)*np.abs(products).sum():
            raise RuntimeError('Framework correctness check failed')
        error=max(error,diff)
    return error

rng=np.random.default_rng(20261001)
inputs=[]
for m,n,k in shapes:
    a=rng.uniform(-0.5,0.5,(m,k)).astype(np.float32)
    b=rng.uniform(-0.5,0.5,(k,n)).astype(np.float32)
    inputs.append((a,b))
    c=np.empty((m,n),np.float32)
    fn=lambda:np.matmul(a,b,out=c)
    fn();error=error_check(a,b,c)
    wall,device=samples(fn)
    append('numpy_cpu','fp32',(m,n,k),wall,device,error)

try:
    torch=importlib.import_module('torch')
except ImportError:
    skipped.append(dict(engine='pytorch',reason='not installed in benchmark interpreter'))
else:
    versions['pytorch']=torch.__version__
    if not torch.cuda.is_available():
        skipped.append(dict(engine='pytorch_cuda',reason='CUDA unavailable'))
    else:
        versions['torch_cuda']=torch.version.cuda
        for fast in (False,True):
            torch.backends.cuda.matmul.allow_tf32=fast
            for shape,(a,b) in zip(shapes,inputs):
                ta=torch.tensor(a,device='cuda');tb=torch.tensor(b,device='cuda')
                tc=torch.empty((shape[0],shape[1]),device='cuda',dtype=torch.float32)
                fn=lambda:torch.mm(ta,tb,out=tc)
                fn();torch.cuda.synchronize()
                error=error_check(a,b,tc.cpu().numpy(),fast)
                wall,device=samples(fn,torch.cuda.synchronize,lambda:torch.cuda.Event(enable_timing=True))
                append('pytorch_cuda','tf32' if fast else 'fp32',shape,wall,device,error)

try:
    cp=importlib.import_module('cupy')
except ImportError:
    skipped.append(dict(engine='cupy',reason='not installed in benchmark interpreter'))
else:
    versions['cupy']=cp.__version__
    # CuPy precision policy is backend/environment dependent; report it explicitly
    # instead of presenting it as a matched-policy FP32/TF32 league table.
    for shape,(a,b) in zip(shapes,inputs):
        ca=cp.asarray(a);cb=cp.asarray(b);cc=cp.empty((shape[0],shape[1]),dtype=cp.float32)
        fn=lambda:cp.matmul(ca,cb,out=cc)
        fn();cp.cuda.get_current_stream().synchronize();error=error_check(a,b,cp.asnumpy(cc),True)
        wall,device=samples(fn,cp.cuda.get_current_stream().synchronize,cp.cuda.Event,cp.cuda.get_elapsed_time)
        # CuPy events expose elapsed_time as a module function.
        append('cupy_cuda','framework_default',shape,wall,device,error)

data=dict(schema_version=1, seed=20261001, python=sys.version, platform=platform.platform(),
          versions=versions, warmup=args.warmup, repeats=args.repeats, batch=args.batch,
          results=results, skipped=skipped,
          comparability='CPU wall time is contextual, not a matched GPU speedup. Inputs use NumPy PCG64; C++ suite uses mt19937. No timed allocation, transfers or scalar reads.')
path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(data,indent=2),encoding='utf-8')
print('Skipped:',skipped)
