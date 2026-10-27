//! Explicit x86 SysV low-EAX projection: retain CFG and every observable effect.
//! No source labels, ASTs, probe outputs or guessed call contracts enter this pass.
use super::*;
pub const REPRESENTATION: &str = "pcode-plus-ssa-return-u32-v1";
/// Straight-line pure low-EAX functions only; drops the constant stack-return epilogue.
pub const COMPACT: &str = "pcode-ssa-return-u32-compact-v1";

pub fn program(raw: &RawDump, profile: &Profile) -> Result<Program> {
    project(super::program(raw, profile)?)
}
fn definition(line: &str) -> Result<Option<(usize, &str)>> {
    if !line.starts_with('%') {
        return Ok(None);
    }
    let (left, rhs) = line.split_once('=').context("SSA definition syntax")?;
    let id = left[1..].split(':').next().context("SSA ID")?.parse()?;
    Ok(Some((id, rhs)))
}
fn project(mut p: Program) -> Result<Program> {
    ensure!(
        p.opaque_calls == 0,
        "return-u32 projection refuses opaque calls"
    );
    let mut dependencies = BTreeMap::<usize, Vec<usize>>::new();
    let mut needed = BTreeSet::new();
    for line in &p.liveins {
        let (id, rhs) = definition(line)?.context("livein definition")?;
        dependencies.insert(id, refs(rhs));
    }
    for block in &mut p.blocks {
        // This lane explicitly observes low EAX, not every exit register.
        block.exit_registers.retain(|offset, _| *offset < 4);
        if !block.exit_registers.is_empty() {
            ensure!(
                block.exit_registers.contains_key(&0),
                "missing low-EAX exit"
            );
        }
        for value in block.exit_registers.values() {
            needed.extend(refs(value));
        }
        for phi in &block.phis {
            dependencies.insert(
                phi.value,
                phi.incoming.values().flat_map(|v| refs(v)).collect(),
            );
        }
        for line in &block.body {
            if let Some((id, rhs)) = definition(line)? {
                dependencies.insert(id, refs(rhs));
                // A load or trapping arithmetic must execute even if its value is dead.
                if rhs.contains('!') {
                    needed.insert(id);
                }
            } else {
                // Keep branch predicates, return targets, stores and effect chains.
                needed.extend(refs(line));
            }
        }
    }
    let mut pending: Vec<_> = needed.iter().copied().collect();
    while let Some(id) = pending.pop() {
        for &dep in dependencies
            .get(&id)
            .context("undefined slice dependency")?
        {
            if needed.insert(dep) {
                pending.push(dep);
            }
        }
    }
    p.liveins
        .retain(|line| needed.contains(&definition(line).unwrap().unwrap().0));
    for block in &mut p.blocks {
        block.phis.retain(|phi| needed.contains(&phi.value));
        block.body.retain(|line| {
            definition(line)
                .unwrap()
                .is_none_or(|(id, _)| needed.contains(&id))
        });
    }
    Ok(p)
}

pub fn compile(raw: &RawDump, profile: &Profile) -> Result<Compiled> {
    let p = program(raw, profile)?;
    let mut compiled = super::render(&p, raw, REPRESENTATION)?;
    // Compact value IDs and omit register-write annotations. Livein storage and
    // the explicit exit map retain ABI meaning; CFG/effect order stay unchanged.
    let mut ids = BTreeMap::new();
    for line in compiled.prompt.lines() {
        if let Some((id, _)) = definition(line)? {
            let next = ids.len();
            ids.insert(id, next);
        }
    }
    let mut lines = Vec::new();
    for line in compiled.prompt.lines() {
        let line = line.split(" @").next().context("empty SSA line")?;
        let mut out = String::new();
        let mut rest = line;
        while let Some(pos) = rest.find('%') {
            out.push_str(&rest[..pos + 1]);
            rest = &rest[pos + 1..];
            let end = rest.bytes().take_while(u8::is_ascii_digit).count();
            let old: usize = rest[..end].parse()?;
            out.push_str(&ids.get(&old).context("unbound compact value")?.to_string());
            rest = &rest[end..];
        }
        out.push_str(rest);
        lines.push(out);
    }
    lines[0].push_str(" observe=low_EAX effects=all calls=refused");
    compiled.prompt = lines.join("\n") + "\n";
    compiled.prompt_sha256 = sha256(compiled.prompt.as_bytes());
    Ok(compiled)
}

fn replace_values(text: &str, names: &BTreeMap<usize, String>) -> Result<String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(pos) = rest.find('%') {
        out.push_str(&rest[..pos]);
        rest = &rest[pos + 1..];
        let end = rest.bytes().take_while(u8::is_ascii_digit).count();
        let id: usize = rest[..end].parse()?;
        out.push_str(names.get(&id).context("unbound compact value")?);
        rest = &rest[end..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Remove ABI plumbing from the return-u32 prompt without changing the dataflow.
/// Accepted shape: one block; live-in registers, single-entry phis, pure
/// operations, one load of the return address from the entry stack register,
/// the return of that load, and the low-EAX exit. Anything else is rejected.
pub fn compile_compact(raw: &RawDump, profile: &Profile) -> Result<Compiled> {
    let mut compiled = compile(raw, profile)?;
    let mut names = BTreeMap::<usize, String>::new();
    let mut body = Vec::new();
    let mut exit = None;
    let mut blocks = 0;
    let mut stack_load = None;
    let mut returned = false;
    for line in compiled.prompt.lines().skip(1) {
        if line.starts_with("block") {
            blocks += 1;
        } else if line.starts_with("e0_0=effect_phi(entry:effect0)") {
        } else if let Some((id, rhs)) = definition(line)? {
            if let Some(register) = rhs
                .strip_prefix("livein(")
                .and_then(|r| r.strip_suffix(')'))
            {
                // Full-width registers keep the bare name; narrower views keep their size.
                names.insert(
                    id,
                    register.strip_suffix(":8").unwrap_or(register).to_owned(),
                );
            } else if let Some(source) = rhs
                .strip_prefix("phi(entry:")
                .and_then(|r| r.strip_suffix(')'))
                .filter(|r| !r.contains(','))
            {
                names.insert(id, replace_values(source, &names)?);
            } else if rhs.starts_with("Load(") && rhs.contains(" !read_ram_may_trap") {
                let address = rhs.split(',').nth(1).and_then(|a| a.split(')').next());
                ensure!(
                    stack_load.is_none()
                        && address.map(|a| replace_values(a, &names)).transpose()?
                            == Some("r20".into()),
                    "compact lane allows only the return-address load"
                );
                stack_load = Some(id);
            } else if !rhs.contains('!') {
                let next = body.len();
                let text = replace_values(rhs, &names)?;
                let size = line[1..].split('=').next().context("SSA size")?;
                let size = size.split_once(':').context("SSA size")?.1;
                body.push(format!("%{next}:{size}={text}"));
                names.insert(id, format!("%{next}"));
            } else {
                anyhow::bail!("compact lane rejects effectful operation");
            }
        } else if let Some(value) = line
            .strip_prefix("Return(%")
            .and_then(|r| r.split(')').next())
        {
            ensure!(
                !returned && value.parse::<usize>().ok() == stack_load,
                "compact lane must return the stack load"
            );
            returned = true;
        } else if line.starts_with("exit_regs(") {
            ensure!(exit.is_none(), "duplicate exit map");
            exit = Some(replace_values(line, &names)?);
        } else {
            anyhow::bail!("compact lane rejects control flow or memory effects");
        }
    }
    ensure!(
        blocks == 1 && returned && !body.is_empty(),
        "compact lane requires one straight-line returning block"
    );
    compiled.prompt = format!(
        "{COMPACT} x86_64 le observe=low_EAX\n{}\n{}\n",
        body.join("\n"),
        exit.context("missing exit map")?
    );
    compiled.prompt_sha256 = sha256(compiled.prompt.as_bytes());
    compiled.representation = COMPACT;
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::super::{
        audit,
        tests::{fixture, profile},
    };
    use super::*;
    #[test]
    fn dead_trapping_values_are_retained_and_undefined_dependencies_reject() -> Result<()> {
        let raw = fixture("forge_diamond")?;
        let mut p = super::super::program(&raw, &profile())?;
        let id = p.sizes.len();
        p.sizes.push(4);
        p.blocks[0].body.insert(
            1,
            format!("%{id}:4=IntDiv(#1:4,#0:4) !arithmetic_may_trap e0_0->e0_1"),
        );
        let sliced = project(p)?;
        assert!(sliced.blocks[0].body.iter().any(|l| l.contains("IntDiv(")));
        assert!(audit::ssa_execute(&sliced, 1, 2).is_err());
        let mut malformed = super::super::program(&raw, &profile())?;
        malformed.blocks[0]
            .body
            .push("Return(%999999) !return".into());
        assert!(project(malformed).is_err());
        Ok(())
    }
    #[test]
    fn projection_preserves_diamond_loops_effects_and_rejects_calls() -> Result<()> {
        for name in ["forge_diamond", "forge_loop"] {
            let raw = fixture(name)?;
            let full = super::super::program(&raw, &profile())?;
            let sliced = program(&raw, &profile())?;
            for a in 0..32 {
                for b in [0, 1, 7, u32::MAX] {
                    assert_eq!(
                        audit::ssa_execute(&full, a, b)?,
                        audit::ssa_execute(&sliced, a, b)?
                    );
                }
            }
            let c = compile(&raw, &profile())?;
            assert!(c.prompt.len() < super::super::compile(&raw, &profile())?.prompt.len());
            assert!(c.prompt.contains("!read_ram_may_trap") && c.prompt.contains("Return("));
            assert!(c.prompt.contains("If("));
            assert!(
                sliced
                    .blocks
                    .iter()
                    .all(|b| b.exit_registers.keys().all(|r| *r < 4))
            );
        }
        for name in ["forge_direct_call", "forge_indirect_call"] {
            assert!(program(&fixture(name)?, &profile()).is_err());
        }
        Ok(())
    }

    #[test]
    fn compact_lane_is_pure_straight_line_dataflow_only() -> Result<()> {
        let raw = fixture("forge_callee")?;
        let full = compile(&raw, &profile())?;
        let c = compile_compact(&raw, &profile())?;
        assert_eq!(c.representation, COMPACT);
        assert!(c.prompt.len() < full.prompt.len() / 2);
        assert!(c.prompt.starts_with(COMPACT) && c.prompt.contains("exit_regs("));
        for forbidden in ["phi", "livein", "block", "Load", "Return", "effect", "!"] {
            assert!(!c.prompt.contains(forbidden), "{forbidden}");
        }
        for name in ["forge_diamond", "forge_loop", "forge_direct_call"] {
            assert!(compile_compact(&fixture(name)?, &profile()).is_err());
        }
        Ok(())
    }
}
