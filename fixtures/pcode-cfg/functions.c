/* Owned CFG/call fixtures; compiled and statically lifted, not benchmark data. */
unsigned forge_diamond(unsigned a, unsigned b) {
    volatile unsigned x;
    if (a > b) x = a + 3;
    else x = b - 2;
    return x + 1;
}
unsigned forge_loop(unsigned n) {
    unsigned x = 0;
    while (n) { x = x * 3 + n; --n; }
    return x;
}
__attribute__((noinline)) unsigned forge_callee(unsigned a) { return a + 2; }
unsigned forge_direct_call(unsigned a) { return forge_callee(a) + 1; }
unsigned forge_indirect_call(unsigned (*fn)(unsigned), unsigned a) { return fn(a) + 1; }
