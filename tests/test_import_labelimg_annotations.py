import tempfile
import unittest
from pathlib import Path

from tools.eval.import_labelimg_annotations import (
    annotation_items_from_voc_xml,
    annotation_lines_for_image,
    load_labelimg_to_ours,
    remap_label_file_lines,
)


class TestImportLabelImgAnnotations(unittest.TestCase):
    def test_resolves_mapping_from_classes_txt_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'classes.txt').write_text(
                '\n'.join([
                    'dog', 'person', 'cat', 'tv', 'car', 'meatballs',
                    'marinara sauce', 'tomato soup', 'chicken noodle soup',
                    'french onion soup', 'chicken breast', 'ribs',
                    'pulled pork', 'hamburger', 'cavity',
                    'facade', 'window', 'door', 'socle',
                ]),
                encoding='utf-8',
            )
            mapping = load_labelimg_to_ours(root)
            self.assertEqual(mapping, {15: 0, 16: 1, 17: 2, 18: 3})

    def test_remaps_source_txt_with_resolved_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / 'sample.txt'
            src.write_text(
                '\n'.join([
                    '15 0.5 0.5 0.8 0.3',
                    '16 0.7 0.6 0.1 0.2',
                    '17 0.4 0.6 0.1 0.3',
                    '18 0.5 0.8 0.8 0.1',
                ]),
                encoding='utf-8',
            )
            lines = remap_label_file_lines(src, {15: 0, 16: 1, 17: 2, 18: 3})
            self.assertEqual(
                lines,
                [
                    '0 0.5 0.5 0.8 0.3',
                    '1 0.7 0.6 0.1 0.2',
                    '2 0.4 0.6 0.1 0.3',
                    '3 0.5 0.8 0.8 0.1',
                ],
            )

    def test_prefers_xml_named_annotations_over_numeric_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / 'case.png'
            image_path.write_bytes(b'')
            (root / 'classes.txt').write_text(
                '\n'.join([
                    'dog', 'person', 'cat', 'tv', 'car', 'meatballs',
                    'marinara sauce', 'tomato soup', 'chicken noodle soup',
                    'french onion soup', 'chicken breast', 'ribs',
                    'pulled pork', 'hamburger', 'cavity',
                    'facade', 'window', 'door', 'socle',
                ]),
                encoding='utf-8',
            )
            (root / 'case.txt').write_text(
                '17 0.5 0.5 0.8 0.3\n18 0.5 0.8 0.8 0.1\n',
                encoding='utf-8',
            )
            (root / 'case.xml').write_text(
                '<annotation>'
                '<size><width>100</width><height>50</height><depth>3</depth></size>'
                '<object><name>facade</name><bndbox><xmin>10</xmin><ymin>15</ymin><xmax>90</xmax><ymax>45</ymax></bndbox></object>'
                '<object><name>window</name><bndbox><xmin>70</xmin><ymin>10</ymin><xmax>80</xmax><ymax>20</ymax></bndbox></object>'
                '<object><name>socle</name><bndbox><xmin>10</xmin><ymin>45</ymin><xmax>90</xmax><ymax>50</ymax></bndbox></object>'
                '</annotation>',
                encoding='utf-8',
            )
            mapping = load_labelimg_to_ours(root)
            size, items = annotation_items_from_voc_xml(root / 'case.xml')
            self.assertEqual(size, (100, 50))
            self.assertEqual([name for name, _ in items], ['facade', 'window', 'socle'])
            lines, source = annotation_lines_for_image(image_path, mapping)
            self.assertEqual(source, 'xml')
            self.assertEqual(
                lines,
                [
                    '0 0.500000 0.600000 0.800000 0.600000',
                    '1 0.750000 0.300000 0.100000 0.200000',
                    '3 0.500000 0.950000 0.800000 0.100000',
                ],
            )


if __name__ == '__main__':
    unittest.main()
