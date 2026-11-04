"""Regression checks for offline inference, checkpoint resume, and data coverage."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from model.wadt_net import WADNet
from model.wadt_net_no_shift import WADNet as NonShiftedNet
from model.inference_embedder import load_inference_embedder
from model.labels import LABELS
from utils.ckpt_utils import load_restore_ckpt_with_optim, save_checkpoint
from utils.dataset_loader import STYLE_LABELS, load_combined_dataset
from utils.metrics import peak_signal_noise_ratio, structural_similarity

EMBEDDER = 'logs/embedder_resnet18_bs64_optadamw_lr1.00e-05_cw1.0_conw0.1_temp0.07_20260101-190539.pt'


class RuntimeTests(unittest.TestCase):
    def test_offline_embedder_rejects_missing_executed_weights(self):
        model = load_inference_embedder(EMBEDDER).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (1, 512))
        self.assertTrue(torch.isfinite(out).all())
        state = model.state_dict()
        del state['final.weight']
        with self.assertRaises(RuntimeError):
            model.load_pretrained_state(state)

    def test_nonshifted_parity_and_shifted_odd_shapes(self):
        torch.manual_seed(5)
        a, b = WADNet().eval(), NonShiftedNet().eval()
        b.load_state_dict(a.state_dict(), strict=True)
        x, emb = torch.randn(1, 3, 17, 19), torch.randn(1, 512)
        with torch.no_grad():
            first, second = a(x, emb)[0], b(x, emb)[0]
            shifted = WADNet(use_shifted_wsa=True).eval()(x, emb)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(shifted.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(shifted).all())

    def test_resume_matches_uninterrupted_optimizer_step(self):
        def modules():
            return {'restorer': torch.nn.Linear(2, 2),
                    'projection': torch.nn.Linear(2, 2), 'bottleneck': torch.nn.Identity()}
        def step(r, p, opt):
            opt.zero_grad()
            (r(p(torch.ones(1, 2))) ** 2).sum().backward()
            opt.step()
        r, p, b, opt, _ = load_restore_ckpt_with_optim('cpu', modules())
        step(r, p, opt)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'nested/checkpoint.pt'
            save_checkpoint({'restorer': r, 'projection': p, 'bottleneck': b}, opt, 1, path, {'psnr': 1.0})
            rr, pp, _, resumed, epoch = load_restore_ckpt_with_optim('cpu', modules(), ckpt_name=path)
            self.assertEqual(epoch, 1)
            step(r, p, opt)
            step(rr, pp, resumed)
            for a, c in zip(list(r.parameters()) + list(p.parameters()), list(rr.parameters()) + list(pp.parameters())):
                torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_metrics_known_values(self):
        a, b = np.zeros((9, 9, 3)), np.full((9, 9, 3), 0.1)
        self.assertEqual(structural_similarity(a, a), 1.0)
        self.assertAlmostEqual(peak_signal_noise_ratio(a, b), 20.0)
        self.assertAlmostEqual(structural_similarity(a, b), 0.0001/0.0101)

    def test_labels_workers_and_full_sample_count(self):
        self.assertEqual(STYLE_LABELS, LABELS)
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'splits').mkdir()
            Image.new('RGB', (16, 16)).save(root/'sample.png')
            (root/'splits/cdd11_test.txt').write_text('sample.png sample.png haze\nsample.png sample.png haze\nsample.png sample.png rain\n')
            _, _, loader, _ = load_combined_dataset(str(root), str(root), split='test', workers=0, batch_size=3)
            self.assertEqual(loader.num_workers, 0)
            self.assertEqual(loader.batch_size, 3)
            self.assertEqual(len(loader.dataset), 3)
            self.assertEqual(next(iter(loader))['input'].shape[0], 3)


if __name__ == '__main__':
    unittest.main()
