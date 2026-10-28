"""Create local Compose secrets and writable directories without replacing existing files."""
import os
from pathlib import Path
import secrets


def main():
    env = Path('.env.compose')
    token = Path('.secrets/metrics_token')
    if env.exists() or token.exists():
        raise SystemExit('Existing Compose configuration found; refusing to replace secrets.')
    token.parent.mkdir(exist_ok=True, mode=0o700)
    with token.open('x') as handle:
        handle.write(secrets.token_hex(32))
    # API UID can read this file; parent directory remains private on the host.
    token.chmod(0o644)
    with env.open('x') as handle:
        handle.write(f'POSTGRES_PASSWORD={secrets.token_hex(24)}\n'
                     f'LOCAL_UID={os.getuid()}\nLOCAL_GID={os.getgid()}\n')
    env.chmod(0o600)
    for directory in ['data', 'artifacts', 'artifacts/releases']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print('Created .env.compose and .secrets/metrics_token. Keep both private and backed up.')


if __name__ == '__main__':
    main()
