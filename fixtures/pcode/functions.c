/* Owned inputs for static extraction tests; not a training corpus or benchmark. */
unsigned int forge_add(unsigned int a, unsigned int b) { return a + b; }
unsigned int forge_udiv(unsigned int a, unsigned int b) { return a / b; }
int forge_sdiv(int a, int b) { return a / b; }
unsigned int forge_branch(unsigned int a) {
    if (a) return forge_add(a, 3);
    return 4;
}
unsigned int forge_call(unsigned int a) { return forge_add(a, 2) + 1; }
