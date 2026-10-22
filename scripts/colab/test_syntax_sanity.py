"""The conditional sanity diagnostic must never select validation/test labels."""
import json
from pathlib import Path
import tempfile
import unittest
import syntax_sanity

class SyntaxSanityTests(unittest.TestCase):
    def test_selection_is_train_only_deterministic_and_requires_all_operators(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'release').mkdir()
            rows=[dict(id='shallow_one_'+op+'_a_0000-O1',source='source '+op) for op in syntax_sanity.OPS]
            (root/'release/train.jsonl').write_text('\n'.join(json.dumps(r) for r in reversed(rows))+'\n')
            (root/'release/validation.jsonl').write_text('not valid JSON; must never be read')
            plan=syntax_sanity.prepare(root)
            self.assertEqual(plan['ids'],[r['id'] for r in rows]);self.assertEqual(plan['validation_model_examples'],0);self.assertEqual(plan['test_model_examples'],0)
            self.assertEqual(syntax_sanity.prepare(root),plan)
            (root/'release/train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[1:])+'\n')
            with self.assertRaisesRegex(ValueError,'lacks add'):syntax_sanity.prepare(root)
