"""Build kuna-pcode-dump against a local Kuna checkout and dump a corpus's raw P-code.

KUNA_ROOT defaults to ~/kuna. The Kuna checkout is only read; the tool builds in its own
target directory. Output matches DumpPcode.java so compare_lifters.py / compare_prompts.py
work unchanged.
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess

HERE = Path(__file__).resolve().parent


def elf_section(data, wanted):
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3A)
    sections = []
    for i in range(shnum):
        name, kind, flags, addr, offset, size = struct.unpack_from('<IIQQQQ', data, shoff + i * shentsize)
        sections.append((name, addr, offset, size))
    str_off = sections[shstrndx][2]
    for name, addr, offset, size in sections:
        end = data.index(b'\0', str_off + name)
        if data[str_off + name:end].decode() == wanted:
            return addr, data[offset:offset + size]
    raise ValueError(f'section {wanted} not found')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binaries', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', default='corpus_')
    p.add_argument('--nm', default='/opt/homebrew/opt/llvm/bin/llvm-nm')
    p.add_argument('--kuna-root', type=Path, default=Path(os.environ.get('KUNA_ROOT', Path.home() / 'kuna')))
    p.add_argument('--target-dir', type=Path, required=True)
    a = p.parse_args()
    kuna = a.kuna_root.resolve()
    work = a.target_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / 'src').mkdir(exist_ok=True)
    (work / 'Cargo.toml').write_text((HERE / 'Cargo.toml.template').read_text().replace('@KUNA@', str(kuna)))
    (work / 'src/main.rs').write_text((HERE / 'src/main.rs').read_text())
    subprocess.run(['cargo', 'build', '--release'], cwd=work, check=True)
    tool = work / 'target/release/kuna-pcode-dump'
    sla = kuna / 'specs/Ghidra/Processors/x86/data/languages/x86-64.sla'
    a.output.mkdir(parents=True, exist_ok=False)
    for elf in sorted(a.binaries.glob('*.elf')):
        data = elf.read_bytes()
        base, text = elf_section(data, '.text')
        (work / 'image.bin').write_bytes(text)
        names = subprocess.run([a.nm, '--defined-only', elf], capture_output=True, text=True, check=True).stdout.splitlines()
        entries = [(parts[2], int(parts[0], 16)) for parts in map(str.split, names) if len(parts) == 3 and parts[2].startswith(a.prefix)]
        (work / 'entries.txt').write_text(''.join(f'{n} {hex(v)}\n' for n, v in entries))
        subprocess.run([tool, '--sla', sla, '--image', work / 'image.bin', '--base', hex(base), '--entries', work / 'entries.txt',
                        '--program', elf.name, '--out', a.output / (elf.name + '.json')], check=True)


if __name__ == '__main__':
    main()
