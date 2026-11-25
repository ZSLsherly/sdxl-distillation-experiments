"""One global-noise arm: batch64 G2 validation, then resumable G160 continuation."""
import json
from pathlib import Path
import turning_batch64_workflow as workflow


def load_config(arm, phase):
    return json.loads((Path(__file__).resolve().parent / 'configs' /
                      f'boundary_global_noise_{phase}.json').read_text())


if __name__ == '__main__':
    workflow.STAGES = (('global_noise','validation'),('global_noise','g160'))
    workflow.load_config = load_config
    workflow.main()
