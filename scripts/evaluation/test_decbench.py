"""Focused schema/parser checks; no benchmark scores or binary execution."""
import unittest
import subprocess
import decbench


class AdapterTests(unittest.TestCase):
    def test_manifest_uses_public_and_rejects_ambiguous_identity(self):
        data = '{"kit_format_version":1,"public":{"bin.elf":["0x12",20]},"private":{"source":"MUST_NOT_BE_USED"}}'
        self.assertEqual(decbench.manifest_targets(data), [('bin.elf','0x12'),('bin.elf','0x14')])
        for invalid in [
            '[]',
            '{"kit_format_version":true,"public":{"x":["1"]}}',
            '{"kit_format_version":1,"public":{"../x":["0x1"]}}',
            '{"kit_format_version":1,"public":{"x":["0x1",1]}}',
            '{"kit_format_version":1,"public":{"x":[true]}}',
            '{"kit_format_version":1,"public":{"x":[-1]}}',
            '{"kit_format_version":1,"public":{"x":["1"],"x":["2"]}}',
        ]:
            with self.assertRaises(ValueError):
                decbench.manifest_targets(invalid)

    def test_clang_parses_one_function_without_repair(self):
        self.assertEqual(decbench.function_name('unsigned f(unsigned a) { return a+2; }', 'clang'), 'f')
        for invalid in ['#include "private.h"\nint f(void){return 0;}',
                        'int f(void){return 1;} int g(void){return 2;}',
                        'int f(void){return ; broken! }']:
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                decbench.function_name(invalid, 'clang')


if __name__ == '__main__':
    unittest.main()
