use super::*;

fn fixture(name: &str) -> Result<RawDump> {
    Ok(serde_json::from_slice(&std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../fixtures/pcode-cfg/{name}.raw.json")),
    )?)?)
}
fn profile() -> Profile {
    serde_json::from_str(include_str!(
        "../../../../../configs/lifter/fission-x86_64.json"
    ))
    .unwrap()
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
fn initial(a: u32, b: u32) -> BTreeMap<u64, u8> {
    let mut registers = BTreeMap::new();
    write_bytes(&mut registers, 0x38, 8, u128::from(a));
    write_bytes(&mut registers, 0x30, 8, u128::from(b));
    write_bytes(&mut registers, 0x20, 8, 0x1000);
    registers
}
fn raw_execute(raw: &RawDump, p: &Profile, a: u32, b: u32) -> Result<u32> {
    let mut ops: Vec<_> = raw.pcode.blocks.iter().flat_map(|b| b.ops.iter()).collect();
    ops.sort_by_key(|o| o.seq_num);
    let mut first = BTreeMap::new();
    for (i, op) in ops.iter().enumerate() {
        first.entry(op.address).or_insert(i);
    }
    let mut regs = initial(a, b);
    let mut temps = BTreeMap::new();
    let mut memory = BTreeMap::new();
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
                read_bytes(&memory, values[1].0 as u64, out.size)
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
fn ssa_execute(program: &Program, a: u32, b: u32) -> Result<u32> {
    ensure!(
        program.opaque_calls == 0,
        "opaque call cannot execute in test VM"
    );
    let registers = initial(a, b);
    let mut values = vec![None; program.sizes.len()];
    let mut memory = BTreeMap::new();
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
                    read_bytes(&memory, operands[1].0 as u64, size)
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

#[test]
fn real_loop_and_instruction_internal_diamond_match_raw_and_source_oracles() -> Result<()> {
    let p = profile();
    let diamond = fixture("forge_diamond")?;
    let loop_raw = fixture("forge_loop")?;
    let diamond_ssa = program(&diamond, &p)?;
    let loop_ssa = program(&loop_raw, &p)?;
    assert_eq!(diamond.pcode.blocks.len(), 1);
    assert_eq!(diamond_ssa.blocks.len(), 3);
    assert!(
        loop_ssa
            .blocks
            .iter()
            .any(|b| b.successors.contains(&b.index))
    );
    for a in [0, 1, 2, 7, 0x7fffffff, 0x80000000, u32::MAX] {
        for b in [0, 1, 3, 9, 0x80000000, u32::MAX] {
            let want = if a > b {
                a.wrapping_add(3)
            } else {
                b.wrapping_sub(2)
            }
            .wrapping_add(1);
            assert_eq!(
                raw_execute(&diamond, &p, a, b)?,
                want,
                "raw diamond {a} {b}"
            );
            assert_eq!(
                ssa_execute(&diamond_ssa, a, b)?,
                want,
                "SSA diamond {a} {b}"
            );
        }
    }
    for n in 0..=63 {
        let mut want = 0u32;
        for k in (1..=n).rev() {
            want = want.wrapping_mul(3).wrapping_add(k);
        }
        assert_eq!(raw_execute(&loop_raw, &p, n, 0)?, want, "raw loop {n}");
        assert_eq!(ssa_execute(&loop_ssa, n, 0)?, want, "SSA loop {n}");
    }
    Ok(())
}
#[test]
fn opaque_calls_have_returning_havoc_and_no_stale_register_values() -> Result<()> {
    for name in ["forge_direct_call", "forge_indirect_call"] {
        let raw = fixture(name)?;
        let p = program(&raw, &profile())?;
        let c = compile(&raw, &profile())?;
        assert_eq!(p.opaque_calls, 1);
        assert_eq!(p.blocks.len(), 2);
        assert!(
            c.prompt
                .contains("may_read_write_all_regs_ram_may_trap_or_not_return")
        );
        assert!(c.prompt.contains("havoc(call0,r20:"));
        assert!(ssa_execute(&p, 1, 2).is_err());
        // Every scalar register incoming from the call block must be a havoc
        // definition, rather than a value from before the unknown callee.
        let definitions = p.blocks[0]
            .body
            .iter()
            .filter(|s| s.starts_with('%'))
            .map(|s| s.split_once('=').unwrap())
            .collect::<Vec<_>>();
        for phi in &p.blocks[1].phis {
            if phi.storage.temporary_instruction.is_none() {
                let reference = &phi.incoming["b0"];
                assert!(definitions.iter().any(|(left, right)| {
                    left.starts_with(&(reference.to_string() + ":")) && right.starts_with("havoc(")
                }));
            }
        }
    }
    Ok(())
}
#[test]
fn malformed_targets_undefined_temps_and_unknown_edges_are_rejected() -> Result<()> {
    let mut raw = fixture("forge_loop")?;
    let branch = raw
        .pcode
        .blocks
        .iter_mut()
        .flat_map(|b| b.ops.iter_mut())
        .find(|o| o.opcode == "CBranch")
        .unwrap();
    branch.inputs[0].offset = 0xffff0000;
    assert!(compile(&raw, &profile()).is_err());
    branch_failure_checks()?;
    let mut raw = fixture("forge_diamond")?;
    raw.pcode.blocks[0].ops[0].inputs[0].space_id = profile().unique_space;
    assert!(compile(&raw, &profile()).is_err());
    Ok(())
}
fn branch_failure_checks() -> Result<()> {
    let mut raw = fixture("forge_loop")?;
    let branch = raw
        .pcode
        .blocks
        .iter_mut()
        .flat_map(|b| b.ops.iter_mut())
        .find(|o| o.opcode == "CBranch")
        .unwrap();
    branch.inputs[0].is_constant = true;
    branch.inputs[0].space_id = profile().constant_space;
    assert!(compile(&raw, &profile()).is_err());
    let mut raw = fixture("forge_loop")?;
    raw.stop_reason = "instruction_limit".into();
    assert!(compile(&raw, &profile()).is_err());
    Ok(())
}
