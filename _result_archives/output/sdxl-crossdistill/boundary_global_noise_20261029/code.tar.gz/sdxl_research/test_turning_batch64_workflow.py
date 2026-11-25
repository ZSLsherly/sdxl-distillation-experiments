"""CPU-only checks of workflow gates; no model or remote GPU is used."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if sys.platform == 'win32':
    sys.modules['fcntl'] = types.ModuleType('fcntl')
spec = importlib.util.spec_from_file_location('turning_batch64_workflow', HERE / 'turning_batch64_workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / 'training_state' / 'done'
        self.state.mkdir(parents=True)
        status = dict(status='complete', student_updates=2, fake_updates=4, D_updates=0,
                      effective_batch=64, training_state=str(self.state))
        (self.root / 'status.json').write_text(json.dumps(status))
        (self.state / 'complete.json').write_text(json.dumps(dict(world_size=2, progress=dict(gcount=2, fcount=4, dcount=0))))
        for rank in range(2):
            for name in ('student', 'fake', 'optimizer_og', 'optimizer_of', 'rng'):
                (self.state / f'{name}_rank{rank}.pt').touch()
            student = dict(phase='student_done', grad_norm=1, dmd_loss=.1, pcm_loss=.1, middle_dm_loss=.1, terminal_dm_loss=.1)
            fake = dict(phase='fake', fake_grad_norm=1, fake_loss=1)
            (self.root / f'rank{rank}.jsonl').write_text('\n'.join(json.dumps(x) for x in [student, student, fake, fake, fake, fake]))
        for update, tag in [(0, 'initial'), (2, 'periodic')]:
            dest = self.root / 'eval' / f'date_g{update:06d}_{tag}'
            dest.mkdir(parents=True)
            for rank in range(2):
                samples = []
                for i in range(4):
                    name = f'{rank}_{i}.png'
                    (dest / name).touch()
                    samples.append(dict(image=name, latent_std=1.))
                (dest / f'rank{rank}.json').write_text(json.dumps(dict(student_updates=update, samples=samples)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_passes_complete_validation(self):
        self.assertEqual(workflow.verify_run(self.root, 2, True), self.state)

    def test_rejects_missing_optimizer(self):
        (self.state / 'optimizer_of_rank1.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'optimizer_of'):
            workflow.verify_run(self.root, 2, True)

    def test_rejects_nonfinite_gradient(self):
        p = self.root / 'rank0.jsonl'
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[0]['grad_norm'] = float('nan')
        p.write_text('\n'.join(json.dumps(x) for x in rows))
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            workflow.verify_run(self.root, 2, True)

    def test_rejects_missing_evaluation(self):
        (self.root / 'eval' / 'date_g000002_periodic' / 'rank1.json').unlink()
        with self.assertRaises(FileNotFoundError):
            workflow.verify_run(self.root, 2, True)

    def test_configs_share_training_parameters(self):
        for arm in workflow.ARMS:
            validation = workflow.load_config(arm, 'validation')
            formal = workflow.load_config(arm, 'g160')
            changed = {k for k in validation.keys() | formal.keys() if validation.get(k) != formal.get(k)}
            self.assertEqual(changed, {'max_student_updates', 'save_every_updates', 'hours'})
            self.assertEqual(formal['save_every_updates'], 40)
            self.assertEqual(validation['batch'] // (2 * validation['micro_batch']), 32)
        a = workflow.load_config('normalized_u', 'validation')
        b = workflow.load_config('raw_angle', 'validation')
        self.assertEqual({k for k in a if a[k] != b[k]}, {'turning_profile'})

    def test_four_rank_validation(self):
        meta = self.state / 'complete.json'
        data = json.loads(meta.read_text())
        data['world_size'] = 4
        meta.write_text(json.dumps(data))
        for rank in range(4):
            if rank >= 2:
                (self.root / f'rank{rank}.jsonl').write_text((self.root / 'rank0.jsonl').read_text())
                for name in ('student', 'fake', 'optimizer_og', 'optimizer_of', 'rng'):
                    (self.state / f'{name}_rank{rank}.pt').touch()
            for folder in (self.root / 'eval').iterdir():
                source = json.loads((folder / 'rank0.json').read_text())
                source['samples'] = source['samples'][:2]
                (folder / f'rank{rank}.json').write_text(json.dumps(source))
        self.assertEqual(workflow.verify_run(self.root, 2, True, world=4), self.state)


if __name__ == '__main__':
    unittest.main()
