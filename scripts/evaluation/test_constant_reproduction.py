import unittest
import constant_reproduction as c

class ConstantReproductionTests(unittest.TestCase):
    def test_expression_literals_and_skeleton(self):
        s='unsigned int f(unsigned int a, unsigned int b) { return ((b ^ 7u) - a); }'
        self.assertEqual(c.expression(s),'((b ^ 7u) - a)')
        self.assertEqual(c.literals(s),['7'])
        self.assertEqual(c.skeleton(s),'((b ^ K) - a)')
        self.assertEqual(c.literals('int f() { return 1; }'),[])
        self.assertEqual(c.expression('no return'),'')

if __name__=='__main__':unittest.main()
