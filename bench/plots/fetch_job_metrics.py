#!/usr/bin/env python
"""Cache QPU usage for previously submitted jobs; never submits a job."""
import argparse
import json
import os
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

from make_plots import default_output_path, latest_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='Run folder name or path; default: latest completed run')
    parser.add_argument('--output', help='Plot output folder, if different from the default')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    base = project / 'bench' / 'results'
    candidate = Path(args.run) if args.run else None
    run = ((candidate if candidate.exists() else base / 'backend_comparison' / candidate)
           if candidate else latest_run(base))
    out = Path(args.output) if args.output else default_output_path(run)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'job_metrics.json'
    metrics_by_id = json.loads(cache.read_text(encoding='utf8')) if cache.exists() else {}
    values = {}
    env_file = project / '.env'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip("\"'")
    token = os.getenv('IBM_API_KEY') or values.get('IBM_API_KEY')
    instance = (os.getenv('IBM_INSTANCE_CRN') or os.getenv('IBM_INSTANCE') or
                values.get('IBM_INSTANCE_CRN') or values.get('IBM_INSTANCE'))
    options = {'channel': 'ibm_quantum_platform', 'token': token} if token else {}
    if instance:
        options['instance'] = instance
    service = QiskitRuntimeService(**options)
    jobs = json.loads((run / 'manifest.json').read_text(encoding='utf8')).get('jobs', [])
    for entry in jobs:
        job_id = entry.get('job_id')
        if not job_id:
            continue
        try:
            usage = service.job(job_id).metrics().get('usage') or {}
            seconds = usage.get('quantum_seconds')
            if usage.get('status') != 'complete' or seconds is None:
                print(f'{job_id}: final QPU usage unavailable')
                continue
            metrics_by_id[job_id] = {'backend': entry['backend'], 'quantum_seconds': seconds}
        except Exception as exc:
            print(f'{job_id}: retrieval failed ({type(exc).__name__})')
    temp = cache.with_suffix('.tmp')
    temp.write_text(json.dumps(metrics_by_id, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(cache)
    print(f'Saved {len(metrics_by_id)} job metrics to {cache}')


if __name__ == '__main__':
    main()
