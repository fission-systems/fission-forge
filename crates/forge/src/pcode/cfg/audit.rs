//! Bounded owned-corpus consistency VM, not a general CPU emulator.
//! Explicit x86 SysV two-u32 ABI, mapped return slot, two register fill states.
use super::*;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
fn read_memory(memory: &BTreeMap<u64, u8>, offset: u64, size: u32) -> Result<u128> {
    ensure!(
        (0..size).all(|i| memory.contains_key(&(offset + u64::from(i)))),
        "unmapped RAM read"
    );
    Ok(read_bytes(memory, offset, size))
}
fn args(text: &str) -> Vec<&str> {
    let mut depth = 0;
    let mut start = 0;
    let mut result = Vec::new();
    for (i, c) in text.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                result.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < text.len() {
        result.push(&text[start..]);
    }
    result
}
fn mask(bytes: u32) -> u128 {
    if bytes == 16 {
        u128::MAX
    } else {
        (1u128 << (bytes * 8)) - 1
    }
}
fn signed(value: u128, bytes: u32) -> i128 {
    let shift = 128 - bytes * 8;
    ((value << shift) as i128) >> shift
}
fn arithmetic(name: &str, values: &[(u128, u32)], size: u32) -> Result<u128> {
    let (a, width) = values[0];
    let b = values.get(1).map(|v| v.0).unwrap_or(0);
    let result = match name {
        "Copy" | "IntZExt" => a,
        "IntSExt" => signed(a, width) as u128,
        "IntAdd" => a.wrapping_add(b),
        "IntSub" => a.wrapping_sub(b),
        "IntMult" => a.wrapping_mul(b),
        "IntAnd" | "BoolAnd" => a & b,
        "IntOr" | "BoolOr" => a | b,
        "IntXor" | "BoolXor" => a ^ b,
        "IntNegate" => !a,
        "Int2Comp" => a.wrapping_neg(),
        "BoolNegate" => u128::from(a == 0),
        "IntEqual" => u128::from(a == b),
        "IntNotEqual" => u128::from(a != b),
        "IntLess" => u128::from(a < b),
        "IntLessEqual" => u128::from(a <= b),
        "IntSLess" => u128::from(signed(a, width) < signed(b, width)),
        "IntSLessEqual" => u128::from(signed(a, width) <= signed(b, width)),
        "IntCarry" => {
            let (r, overflow) = a.overflowing_add(b);
            u128::from(overflow || r > mask(width))
        }
        "IntSCarry" => u128::from(
            ((a ^ a.wrapping_add(b)) & (b ^ a.wrapping_add(b))) & (1u128 << (width * 8 - 1)) != 0,
        ),
        "IntSBorrow" => {
            u128::from(((a ^ b) & (a ^ a.wrapping_sub(b))) & (1u128 << (width * 8 - 1)) != 0)
        }
        "IntLeft" => {
            if b >= 128 {
                0
            } else {
                a << (b as u32)
            }
        }
        "IntRight" => {
            if b >= 128 {
                0
            } else {
                a >> (b as u32)
            }
        }
        "IntSRight" => {
            if b >= width as u128 * 8 {
                if signed(a, width) < 0 { mask(width) } else { 0 }
            } else {
                (signed(a, width) >> (b as u32)) as u128
            }
        }
        "PopCount" => u128::from(a.count_ones()),
        "LzCount" => u128::from(a.leading_zeros()) - (128 - width * 8) as u128,
        "SubPiece" => a >> (b * 8),
        "Piece" => (a << (values[1].1 * 8)) | b,
        "IntDiv" => a.checked_div(b).context("division trap")?,
        "IntRem" => a.checked_rem(b).context("remainder trap")?,
        "IntSDiv" => signed(a, width)
            .checked_div(signed(b, width))
            .context("division trap")? as u128,
        "IntSRem" => signed(a, width)
            .checked_rem(signed(b, width))
            .context("remainder trap")? as u128,
        _ => bail!("test VM does not implement {name}"),
    };
    Ok(result & mask(size))
}
fn read_bytes(bytes: &BTreeMap<u64, u8>, offset: u64, size: u32) -> u128 {
    (0..size)
        .map(|i| u128::from(*bytes.get(&(offset + u64::from(i))).unwrap_or(&0)) << (8 * i))
        .sum()
}
fn write_bytes(bytes: &mut BTreeMap<u64, u8>, offset: u64, size: u32, value: u128) {
    for i in 0..size {
        bytes.insert(offset + u64::from(i), (value >> (8 * i)) as u8);
    }
}
fn initial(a: u32, b: u32, fill: u8) -> BTreeMap<u64, u8> {
    let mut registers: BTreeMap<_, _> = (0..0x400).map(|i| (i, fill)).collect();
    write_bytes(&mut registers, 0x38, 8, u128::from(a));
    write_bytes(&mut registers, 0x30, 8, u128::from(b));
    write_bytes(&mut registers, 0x20, 8, 0x1000);
    registers
}
#[cfg(test)]
pub(super) fn raw_execute(raw: &RawDump, p: &Profile, a: u32, b: u32) -> Result<u32> {
    raw_with_fill(raw, p, a, b, 0)
}
fn raw_with_fill(raw: &RawDump, p: &Profile, a: u32, b: u32, fill: u8) -> Result<u32> {
    let mut ops: Vec<_> = raw.pcode.blocks.iter().flat_map(|b| b.ops.iter()).collect();
    ops.sort_by_key(|o| o.seq_num);
    let mut first = BTreeMap::new();
    for (i, op) in ops.iter().enumerate() {
        first.entry(op.address).or_insert(i);
    }
    let mut regs = initial(a, b, fill);
    let mut temps = BTreeMap::new();
    let mut memory = BTreeMap::new();
    write_bytes(&mut memory, 0x1000, 8, 0xfeedbaad);
    let mut pc = 0;
    let mut previous = None;
    for _ in 0..20000 {
        let op = ops.get(pc).context("raw PC escaped")?;
        if previous != Some(op.address) {
            temps.clear();
            previous = Some(op.address);
        }
        let read = |v: &Varnode| {
            if v.is_constant {
                u128::from(constant_bits(v))
            } else if v.space_id == p.register_space {
                read_bytes(&regs, v.offset, v.size)
            } else {
                read_bytes(&temps, v.offset, v.size)
            }
        };
        if op.opcode == "Return" {
            ensure!(read(&op.inputs[0]) == 0xfeedbaad, "return target changed");
            return Ok(read_bytes(&regs, 0, 4) as u32);
        }
        if op.opcode == "Branch" || op.opcode == "CBranch" {
            if op.opcode == "Branch" || read(&op.inputs[1]) != 0 {
                pc = *first.get(&op.inputs[0].offset).context("missing target")?;
            } else {
                pc += 1;
            }
            continue;
        }
        ensure!(
            !matches!(op.opcode.as_str(), "Call" | "CallInd"),
            "opaque call cannot execute in test VM"
        );
        let values: Vec<_> = op.inputs.iter().map(|v| (read(v), v.size)).collect();
        if op.opcode == "Store" {
            write_bytes(&mut memory, values[1].0 as u64, values[2].1, values[2].0);
        } else if let Some(out) = &op.output {
            let value = if op.opcode == "Load" {
                read_memory(&memory, values[1].0 as u64, out.size)?
            } else {
                arithmetic(&op.opcode, &values, out.size)?
            };
            write_bytes(
                if out.space_id == p.register_space {
                    &mut regs
                } else {
                    &mut temps
                },
                out.offset,
                out.size,
                value,
            );
        }
        pc += 1;
    }
    bail!("raw step limit")
}
fn expression(text: &str, values: &[Option<u128>], sizes: &[u32]) -> Result<(u128, u32)> {
    if let Some(inner) = text.strip_prefix("le(").and_then(|s| s.strip_suffix(')')) {
        let mut output = 0;
        let mut shift = 0;
        for part in args(inner) {
            let (value, size) = expression(part, values, sizes)?;
            output |= value << shift;
            shift += size * 8;
        }
        return Ok((output, shift / 8));
    }
    if let Some(constant) = text.strip_prefix('#') {
        let (value, size) = constant.split_once(':').context("constant syntax")?;
        return Ok((u128::from_str_radix(value, 16)?, size.parse()?));
    }
    let text = text.strip_prefix('%').context("value syntax")?;
    let (id, slice) = text
        .split_once('[')
        .map(|(a, b)| (a, Some(b)))
        .unwrap_or((text, None));
    let id: usize = id.parse()?;
    let value = values
        .get(id)
        .copied()
        .flatten()
        .with_context(|| format!("undefined SSA %{id}"))?;
    if let Some(slice) = slice {
        let (low, size) = slice
            .strip_suffix(']')
            .context("slice syntax")?
            .split_once(':')
            .context("slice syntax")?;
        let low: u32 = low.parse()?;
        let size: u32 = size.parse()?;
        Ok(((value >> (low * 8)) & mask(size), size))
    } else {
        Ok((value, sizes[id]))
    }
}
#[cfg(test)]
pub(super) fn ssa_execute(program: &Program, a: u32, b: u32) -> Result<u32> {
    ssa_with_fill(program, a, b, 0)
}
fn ssa_with_fill(program: &Program, a: u32, b: u32, fill: u8) -> Result<u32> {
    ensure!(
        program.opaque_calls == 0,
        "opaque call cannot execute in test VM"
    );
    let registers = initial(a, b, fill);
    let mut values = vec![None; program.sizes.len()];
    let mut memory = BTreeMap::new();
    write_bytes(&mut memory, 0x1000, 8, 0xfeedbaad);
    for livein in &program.liveins {
        let (left, right) = livein.split_once('=').unwrap();
        let id: usize = left[1..].split(':').next().unwrap().parse()?;
        let (r, size) = right
            .strip_prefix("livein(r")
            .unwrap()
            .strip_suffix(')')
            .unwrap()
            .split_once(':')
            .unwrap();
        values[id] = Some(read_bytes(
            &registers,
            u64::from_str_radix(r, 16)?,
            size.parse()?,
        ));
    }
    let mut pc = 0;
    let mut predecessor = "entry".to_string();
    for _ in 0..1000 {
        let block = &program.blocks[pc];
        // Phi assignments are simultaneous, especially on a self-loop backedge.
        let incoming = block
            .phis
            .iter()
            .map(|phi| {
                Ok((
                    phi.value,
                    expression(
                        phi.incoming
                            .get(&predecessor)
                            .context("phi predecessor missing")?,
                        &values,
                        &program.sizes,
                    )?
                    .0,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for (id, value) in incoming {
            values[id] = Some(value);
        }
        let mut next = None;
        for line in &block.body {
            if line.starts_with('e') {
                continue;
            }
            let (output, rhs) = if line.starts_with('%') {
                let (left, right) = line.split_once('=').unwrap();
                (
                    Some(left[1..].split_once(':').unwrap().0.parse::<usize>()?),
                    right,
                )
            } else {
                (None, line.as_str())
            };
            let (opcode, tail) = rhs.split_once('(').context("operation syntax")?;
            let end = tail.rfind(')').context("operation close")?;
            let operands = args(&tail[..end]);
            match opcode {
                "Jump" => {
                    next = Some(operands[0].strip_prefix('b').unwrap().parse()?);
                    break;
                }
                "If" => {
                    let index = if expression(operands[0], &values, &program.sizes)?.0 != 0 {
                        1
                    } else {
                        2
                    };
                    next = Some(operands[index].strip_prefix('b').unwrap().parse()?);
                    break;
                }
                "Return" => {
                    ensure!(
                        expression(operands[0], &values, &program.sizes)?.0 == 0xfeedbaad,
                        "SSA return target changed"
                    );
                    let mut result = 0u128;
                    for (&offset, expr) in &block.exit_registers {
                        if offset < 4 {
                            let (v, size) = expression(expr, &values, &program.sizes)?;
                            let bytes = size.min(4 - offset as u32);
                            result |= (v & mask(bytes)) << (offset * 8);
                        }
                    }
                    return Ok(result as u32);
                }
                _ => {}
            }
            let operands = operands
                .iter()
                .map(|v| expression(v, &values, &program.sizes))
                .collect::<Result<Vec<_>>>()?;
            if opcode == "Store" {
                write_bytes(
                    &mut memory,
                    operands[1].0 as u64,
                    operands[2].1,
                    operands[2].0,
                );
            } else {
                let id = output.context("operation without output")?;
                let size = program.sizes[id];
                values[id] = Some(if opcode == "Load" {
                    read_memory(&memory, operands[1].0 as u64, size)?
                } else {
                    arithmetic(opcode, &operands, size)?
                });
            }
        }
        predecessor = format!("b{pc}");
        pc = next.context("missing terminator")?;
    }
    bail!("SSA block limit")
}

pub fn run(root: &Path, output: &Path) -> Result<serde_json::Value> {
    ensure!(!output.exists(), "output exists");
    crate::corpus::run(root, false)?;
    let probes: Vec<[u32; 2]> = serde_json::from_slice(&fs::read(root.join("probes.json"))?)?;
    ensure!(probes.len() == 64, "owned corpus audit requires 64 probes");
    let programs: BTreeMap<String, serde_json::Value> =
        BufReader::new(fs::File::open(root.join("programs.jsonl"))?)
            .lines()
            .map(|line| -> Result<_> {
                let p: serde_json::Value = serde_json::from_str(&line?)?;
                Ok((p["id"].as_str().context("program ID")?.to_string(), p))
            })
            .collect::<Result<_>>()?;
    let mut rows = Vec::new();
    let mut comparisons = 0;
    let mut failures = 0;
    for split in ["train", "validation", "test"] {
        for line in
            BufReader::new(fs::File::open(root.join(format!("release/{split}.jsonl")))?).lines()
        {
            let pair: Pair = serde_json::from_str(&line?)?;
            let (id, _) = pair.id.rsplit_once('-').context("owned row ID")?;
            let source = programs.get(id).context("missing source program")?;
            ensure!(
                source["source"].as_str() == Some(&pair.source)
                    && source["split"].as_str() == Some(split),
                "source/split identity mismatch"
            );
            ensure!(
                pair.source
                    .starts_with("unsigned int f(unsigned int a, unsigned int b)"),
                "outside owned ABI"
            );
            let expected: Vec<u32> = serde_json::from_value(source["oracle_outputs"].clone())?;
            ensure!(expected.len() == probes.len(), "source probe count changed");
            let ssa = if pair.input.representation == super::slice::REPRESENTATION {
                super::slice::program(&pair.input.raw, &pair.input.profile)?
            } else {
                program(&pair.input.raw, &pair.input.profile)?
            };
            ensure!(ssa.opaque_calls == 0, "calls outside audit ABI");
            for op in pair.input.raw.pcode.blocks.iter().flat_map(|b| &b.ops) {
                for v in op.inputs.iter().chain(op.output.iter()) {
                    ensure!((1..=16).contains(&v.size), "outside scalar VM widths");
                    if v.space_id == pair.input.profile.register_space {
                        ensure!(
                            v.offset + u64::from(v.size) <= 0x400,
                            "outside bounded register file"
                        );
                    }
                }
            }
            let mut mismatches = Vec::new();
            for (index, ([a, b], want)) in probes.iter().zip(&expected).enumerate() {
                for fill in [0u8, 0xa5] {
                    comparisons += 1;
                    let raw = raw_with_fill(&pair.input.raw, &pair.input.profile, *a, *b, fill);
                    let normalized = ssa_with_fill(&ssa, *a, *b, fill);
                    let raw_value = raw.as_ref().ok().copied();
                    let ssa_value = normalized.as_ref().ok().copied();
                    if raw_value != Some(*want) || ssa_value != Some(*want) {
                        failures += 1;
                        mismatches.push(serde_json::json!({"probe":index,"a":a,"b":b,"register_fill":fill,"expected":want,"raw":raw_value,"ssa":ssa_value,"raw_error":raw.err().map(|e|e.to_string()),"ssa_error":normalized.err().map(|e|e.to_string())}));
                    }
                }
            }
            rows.push(serde_json::json!({"id":pair.id,"split":split,"prompt_sha256":sha256(pair.prompt()?.as_bytes()),"comparisons":probes.len()*2,"mismatches":mismatches}));
        }
    }
    let report = serde_json::json!({"schema_version":1,"representation":serde_json::from_slice::<serde_json::Value>(&fs::read(root.join("config.json"))?)?["representation"],"passed":failures==0,"scope":"owned_pure_u32_x86_sysv_probe_consistency_not_general_emulation_or_equivalence_proof","corpus_lock_sha256":sha256(&fs::read(root.join("release/lock.json"))?),"probes_sha256":sha256(&fs::read(root.join("probes.json"))?),"abi":"RDI_zero_extended_a_RSI_zero_extended_b_RSP_0x1000_return_low_EAX","register_fill_states":[0,165],"mapped_return_target":0xfeedbaadu32,"rows":rows.len(),"comparisons_per_interpreter":comparisons,"failures":failures,"test_model_examples":0,"test_labels_used_only_for_ir_consistency":true,"records":rows});
    fs::create_dir_all(output)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod checks {
    use super::*;
    #[test]
    fn unmapped_memory_and_modified_return_targets_fail() -> Result<()> {
        assert!(read_memory(&BTreeMap::new(), 0x1000, 8).is_err());
        let p: Profile = serde_json::from_str(include_str!(
            "../../../../../configs/lifter/fission-x86_64.json"
        ))?;
        let mut raw: RawDump = serde_json::from_str(include_str!(
            "../../../../../fixtures/pcode-cfg/forge_diamond.raw.json"
        ))?;
        let mut ssa = program(&raw, &p)?;
        let ret = raw
            .pcode
            .blocks
            .iter_mut()
            .flat_map(|b| &mut b.ops)
            .find(|o| o.opcode == "Return")
            .unwrap();
        ret.inputs[0] = Varnode {
            space_id: p.constant_space,
            offset: 0,
            size: 8,
            is_constant: true,
            constant_val: 0,
        };
        assert!(raw_with_fill(&raw, &p, 1, 2, 0).is_err());
        assert!(program(&raw, &p).is_err());
        let line = ssa
            .blocks
            .iter_mut()
            .flat_map(|b| &mut b.body)
            .find(|l| l.starts_with("Return("))
            .unwrap();
        *line = "Return(#0:8) !return".into();
        assert!(ssa_with_fill(&ssa, 1, 2, 0).is_err());
        Ok(())
    }
}
