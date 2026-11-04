import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import gradio as gr
import gradio_app as app
from benchmark_comparison import load_comparison, comparison_table


class FrontendTests(unittest.TestCase):
    def test_comparison_reads_original_readme_and_execution_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'README.md').write_text('| **OmniRestore** | CVPRW 2026 | **30.00** | **0.9500** | **2.60M** |')
            (root/'current_execution_run.txt').write_text('run')
            (root/'run').mkdir()
            metrics = {'metrics': {f'{mode}_{metric}': {'finite_mean': value}
                       for mode in ['image', 'fused'] for metric, value in [('psnr', 29.5), ('ssim', .94)]}}
            (root/'run/metrics.json').write_text(json.dumps(metrics))
            reference, report, _ = load_comparison(root)
            self.assertEqual(reference, {'psnr': 30, 'ssim': .95})
            table = comparison_table(reference, report)
            self.assertIn('-0.5000 dB', table)
            self.assertIn('-0.010000', table)

    def test_reference_validation_precedes_model_loading(self):
        with patch.object(app, 'get_models') as loader:
            with self.assertRaises(gr.Error):
                app.run_restoration(Image.new('RGB', (20, 20)), reference=Image.new('RGB', (30, 20)))
            loader.assert_not_called()

    def test_no_reference_has_no_invented_quality_scores(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'OUTPUT', Path(directory)), \
             patch.object(app, 'get_models', return_value=(None, 'cpu')), \
             patch.object(app, 'restore', return_value=Image.new('RGB', (224, 224))):
            pair, result, path, scores, status = app.run_restoration(Image.new('RGB', (30, 40)), original_size=True)
            self.assertEqual(result.size, (30, 40))
            self.assertEqual(pair[0].size, pair[1].size)
            self.assertTrue(Path(path).is_file())
            self.assertEqual(scores[1][1], 'Needs clean reference')
            metadata = json.loads(Path(path).with_suffix('.json').read_text())
            self.assertFalse(metadata['has_clean_reference'])

    def test_identical_clean_reference_scores_match_exported_image(self):
        white = Image.new('RGB', (224, 224), 'white')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'OUTPUT', Path(directory)), \
             patch.object(app, 'get_models', return_value=(None, 'cpu')), \
             patch.object(app, 'restore', return_value=white):
            _, _, _, scores, _ = app.run_restoration(white, reference=white)
            self.assertEqual(scores[1][1], 'infinite (identical)')
            self.assertEqual(scores[1][2], '1.000000')


if __name__ == '__main__':
    unittest.main()
