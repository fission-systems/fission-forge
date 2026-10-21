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

use super::audit::{raw_execute, ssa_execute};

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
