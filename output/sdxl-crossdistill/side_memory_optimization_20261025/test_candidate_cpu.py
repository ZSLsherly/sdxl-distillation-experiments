"""CPU-only checks; run with a PyTorch environment. No server/GPU use."""
import io
import unittest
import torch
from training_state import cpu_state, optimizer_state, _cpu_tensor


def roundtrip(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=True)


class CheckpointTests(unittest.TestCase):
    def test_snapshot_stays_independent_and_sync_save_reuses_storage(self):
        model = torch.nn.Linear(4, 3)
        snapshot = cpu_state(model)
        staging = cpu_state(model, snapshot=False)
        loaded = roundtrip(staging)
        for name, tensor in model.state_dict().items():
            self.assertNotEqual(snapshot[name].data_ptr(), tensor.data_ptr())
            self.assertEqual(staging[name].data_ptr(), tensor.data_ptr())
            torch.testing.assert_close(loaded[name], snapshot[name], rtol=0, atol=0)
        with torch.no_grad():
            model.weight.add_(1)
        torch.testing.assert_close(loaded['weight'], snapshot['weight'], rtol=0, atol=0)

    def test_small_view_does_not_save_full_backing_storage(self):
        large = torch.arange(10000, dtype=torch.float32)
        for view in (large[:4], large[10:14], large[::2500]):
            saved = _cpu_tensor(view, snapshot=False)
            self.assertEqual(saved.untyped_storage().nbytes(), saved.numel()*saved.element_size())
            torch.testing.assert_close(roundtrip(saved), view, rtol=0, atol=0)

    def test_save_restore_next_adam_update_exact(self):
        torch.manual_seed(17)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, foreach=False)
        x = torch.randn(8, 4)
        def step(m, opt):
            opt.zero_grad(set_to_none=True)
            m(x).square().mean().backward()
            opt.step()
        step(model, optimizer)
        states = optimizer_state(optimizer, snapshot=False)
        old = optimizer.state_dict()
        for key, row in states['state'].items():
            for name, value in row.items():
                if torch.is_tensor(value):
                    self.assertEqual(value.data_ptr(), old['state'][key][name].data_ptr())
        saved = roundtrip({'model':cpu_state(model, snapshot=False),'optimizer':states})
        step(model, optimizer)
        expected = cpu_state(model)
        restored = torch.nn.Linear(4, 3)
        opt = torch.optim.AdamW(restored.parameters(), lr=1e-6, foreach=False)
        restored.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        step(restored, opt)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
