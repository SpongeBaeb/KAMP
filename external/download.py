"""ImPure 공개 데이터(Zenodo, CC-BY-4.0) 내려받기 + 체크섬 확인.

Bakas, G. (2022). ImPure Injection Molding Sensor Data - Trial 16th May / 17th May. Zenodo.
doi:10.5281/zenodo.6913666, doi:10.5281/zenodo.6913660

실행: python download.py  → data/<trial>/ 아래 저장, data/manifest.csv에 파일·크기·md5 기록
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

RECORDS = {'trial_17_05': 6913660, 'trial_16_05': 6913666}
OUT = Path(__file__).parent / 'data'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for trial, rid in RECORDS.items():
        meta = json.load(urllib.request.urlopen(f'https://zenodo.org/api/records/{rid}'))
        d = OUT / trial
        d.mkdir(parents=True, exist_ok=True)
        for f in meta['files']:
            path = d / f['key']
            expected = f['checksum'].split(':', 1)[1]
            if not (path.exists() and md5(path) == expected):
                urllib.request.urlretrieve(f['links']['self'], path)
            got = md5(path)
            rows.append({'trial': trial, 'file': f['key'], 'size': f['size'], 'md5': got, 'ok': got == expected})
    man = pd.DataFrame(rows).sort_values(['trial', 'file'])
    man.to_csv(OUT / 'manifest.csv', index=False)
    print(man.groupby('trial').agg(files=('file', 'size'), mb=('size', lambda s: round(s.sum() / 1e6, 2)), all_ok=('ok', 'all')))


if __name__ == '__main__':
    main()
