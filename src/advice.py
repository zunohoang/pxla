"""Data-driven project severity and conservative disease-specific guidance."""
import json
from pathlib import Path
import numpy as np

RULES = json.loads((Path(__file__).resolve().parents[1]/'data/severity_rules.json').read_text(encoding='utf-8'))


def severity(ratio, crop=None, disease=None):
    """Project bins use left-inclusive boundaries; verified IRRI is separate."""
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Area ratio must be finite and in [0,1]')
    project = RULES['project']
    index = int(np.searchsorted(project['thresholds'], ratio, side='right'))
    return dict(label=project['labels'][index], level=index, ratio=float(ratio),
                scale=project['name'], calibrated=False, irri_score=None,
                irri_status=RULES['irri']['status'])


def recommendation(label, ratio, rejected=False):
    """Return distinct low/high/uncertain guidance; no pesticide prescription."""
    rules = RULES['advice']
    if rejected or label is None:
        return rules['uncertain']
    level = severity(ratio)['level']
    common = rules['low'] if level <= 1 else rules['high']
    return rules['diseases'].get(label, 'Ghi ảnh cả hai mặt lá và toàn cây để xác nhận triệu chứng.')+' '+common
