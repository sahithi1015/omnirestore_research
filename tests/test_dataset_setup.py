import tempfile
import unittest
import zipfile
from pathlib import Path
from fetch_dataset import extract_pairs


class DatasetSetupTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'data'
        (root/'splits').mkdir(parents=True)
        (root/'splits/cdd11_test.txt').write_text('test/haze/a.png test/clear/a.png haze\n')
        archive = Path(directory)/'test.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('release/test/haze/a.png', b'degraded')
            z.writestr('release/test/clear/a.png', b'clean')
        return root, archive

    def test_archive_layout_maps_to_existing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = self.fixture(directory)
            self.assertEqual(extract_pairs(archive, root, 'test'), 2)
            self.assertEqual((root/'test/haze/a.png').read_bytes(), b'degraded')
            self.assertEqual((root/'test/clear/a.png').read_bytes(), b'clean')

    def test_conflicting_existing_image_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = self.fixture(directory)
            (root/'test/haze').mkdir(parents=True)
            existing = root/'test/haze/a.png'
            existing.write_bytes(b'original different data')
            with self.assertRaises(RuntimeError):
                extract_pairs(archive, root, 'test')
            self.assertEqual(existing.read_bytes(), b'original different data')

    def test_missing_archive_images_fail_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = self.fixture(directory)
            (root/'splits/cdd11_test.txt').write_text('test/rain/missing.png test/clear/a.png rain\n')
            with self.assertRaises(RuntimeError):
                extract_pairs(archive, root, 'test')
            self.assertFalse((root/'test').exists())


if __name__ == '__main__':
    unittest.main()
