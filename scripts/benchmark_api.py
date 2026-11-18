"""Check Responses settings or compare real streaming/JSON calls using the pipeline client."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightagent.io import ROOT, inside, read, write
from lightagent.llm import Client, load_env
from lightagent.config import default_config, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--modes', nargs='+', choices=['stream', 'json'], default=['stream', 'json'])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--image', type=Path, help='Optional real reference image for a vision smoke test')
    parser.add_argument('--check-only', action='store_true', help='Check local API settings without sending a request')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    load_env(ROOT / '.env')
    cfg = load_config(args.config)['llm']
    if cfg.get('api') not in {'auto', 'responses'}:
        parser.error('Choose an API config with llm.api=responses (not codex_local)')
    cfg = {**cfg, 'api': 'responses',
           'max_output_tokens': 8192, 'timeout': 180}
    base = cfg.get('base_url') or os.getenv('OPENAI_BASE_URL')
    key_env = cfg.get('api_key_env', 'OPENAI_API_KEY')
    key = os.getenv(key_env, '').strip()
    key_configured = bool(key) and not key.startswith(('YOUR_', 'sk-your-', 'sk-你的'))
    settings = {'api': cfg['api'], 'model': cfg['model'], 'base_url': base,
                'reasoning_effort': cfg['reasoning_effort'], 'api_key_env': key_env,
                'key_configured': key_configured}
    if args.check_only:
        print(settings)
        return 0 if base and key_configured else 1
    if not base or not key_configured:
        parser.error(f'Fill {key_env} and OPENAI_BASE_URL in {ROOT / ".env"} before making API calls')
    output = inside(ROOT, args.output or ROOT / 'validation' / ('api_responses_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    images = [('reference', args.image.resolve())] if args.image else []
    if args.image and not args.image.is_file():
        parser.error('Image does not exist')
    output.mkdir(parents=True, exist_ok=False)
    system = 'Return only one valid JSON object. No markdown or extra commentary.'
    context = ({'task': 'Describe the attached scene lighting briefly. Return keys ok (true), visible_sources (array), inferred_sources (array), dominant_light_direction (string), uncertainty (string).'}
               if images else {'task': 'Return exactly {"ok":true,"values":[integers 1 through 128 inclusive in ascending order]}. Expand all integers; no other keys.'})
    report = {'model': cfg['model'], 'base_url': base, 'reasoning_effort': cfg['reasoning_effort'], 'api': 'responses',
              'image': str(args.image.resolve()) if args.image else None, 'calls': [],
              'notes': 'Real billable API calls, no implicit retries. First event can precede answer text. Small samples and gateway caching/load do not establish a general speedup.'}
    for index in range(args.repeat):
        # Alternate order across repetitions to reduce order/cache bias.
        modes = args.modes if index % 2 == 0 else list(reversed(args.modes))
        for mode in modes:
            folder = output / f'{index:02d}_{mode}'
            row = {'mode': mode, 'folder': str(folder)}
            print(f'CALL {index + 1}/{args.repeat} {mode}', flush=True)
            try:
                result = Client({**cfg, 'stream': mode == 'stream', 'max_retries': 0}).ask(folder, system, context, images)
                if result.get('ok') is not True:
                    raise ValueError('Probe response did not set ok=true')
                if not images and result.get('values') != list(range(1, 129)):
                    raise ValueError('Probe response has incorrect values')
                response = read(folder / 'response.json')
                row.update(ok=True, returned_model=response.get('model'), usage=response.get('usage'))
            except Exception as exc:
                # Never echo arbitrary gateway/transport exception strings or credentials.
                row.update(ok=False, error_type=type(exc).__name__)
            if (folder / 'timing.json').exists():
                row['timing'] = read(folder / 'timing.json')
            report['calls'].append(row)
            write(output / 'report.json', report)
            print({key: row[key] for key in ('mode', 'ok', 'error_type', 'returned_model', 'timing') if key in row}, flush=True)
    print(f'REPORT {output / "report.json"}', flush=True)
    return 0 if all(row['ok'] for row in report['calls']) else 1


if __name__ == '__main__':
    sys.exit(main())
