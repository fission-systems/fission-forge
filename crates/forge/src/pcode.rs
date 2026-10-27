//! Strict, straight-line P-code+ v0: storage-aware SSA and ordered effects.
//! This is an input adapter, not an emulator, optimizer, or complete CPU lifter.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod cfg;

pub const REPRESENTATION: &str = "pcode-plus-ssa-v0";

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema_version: u32,
    pub language: String,
    pub endianness: String,
    pub pointer_bytes: u32,
    pub word_size: u32,
    pub constant_space: u64,
    pub unique_space: u64,
    pub register_space: u64,
    pub ram_space: u64,
    pub fission_commit: String,
    pub sla_sha256: String,
}
impl Profile {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && self.language == "x86:LE:64:default"
                && self.endianness == "little"
                && self.pointer_bytes == 8
                && self.word_size == 1,
            "unsupported lifter profile; P-code lanes support byte-addressed little-endian x86-64 only"
        );
        let spaces = [
            self.constant_space,
            self.unique_space,
            self.register_space,
            self.ram_space,
        ];
        ensure!(
            spaces.into_iter().collect::<BTreeSet<_>>().len() == 4,
            "space roles must be distinct"
        );
        for (value, len) in [(&self.fission_commit, 40), (&self.sla_sha256, 64)] {
            ensure!(
                value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid profile revision/digest"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Varnode {
    pub space_id: u64,
    pub offset: u64,
    pub size: u32,
    pub is_constant: bool,
    pub constant_val: i64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Op {
    pub seq_num: u32,
    pub opcode: String,
    pub address: u64,
    pub output: Option<Varnode>,
    pub inputs: Vec<Varnode>,
    pub asm_mnemonic: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub index: u32,
    pub start_address: u64,
    pub successors: Vec<u32>,
    pub ops: Vec<Op>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub blocks: Vec<Block>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawDump {
    pub entry_address: String,
    pub decode_address: String,
    pub decoded_instructions: usize,
    pub stop_reason: String,
    pub template_source_counts: BTreeMap<String, usize>,
    pub pcode: Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub representation: String,
    pub profile: Profile,
    pub raw: RawDump,
}
impl Input {
    pub fn compile(&self) -> Result<Compiled> {
        match self.representation.as_str() {
            REPRESENTATION => compile(&self.raw, &self.profile),
            cfg::REPRESENTATION => cfg::compile(&self.raw, &self.profile),
            cfg::slice::REPRESENTATION => cfg::slice::compile(&self.raw, &self.profile),
            cfg::slice::COMPACT => cfg::slice::compile_compact(&self.raw, &self.profile),
            _ => bail!("unsupported input representation"),
        }
    }
}

/// Provenance is outside the model prompt and cannot provide source-side hints.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub binary_sha256: String,
    pub raw_sha256: String,
    pub source_sha256: String,
    pub extraction: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub schema_version: u32,
    pub id: String,
    pub input: Input,
    pub source: String,
    pub provenance: Provenance,
}
impl Pair {
    pub fn prompt(&self) -> Result<String> {
        ensure!(self.schema_version == 1, "unsupported IR pair schema");
        for digest in [
            &self.provenance.binary_sha256,
            &self.provenance.raw_sha256,
            &self.provenance.source_sha256,
        ] {
            ensure!(
                digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid provenance digest"
            );
        }
        ensure!(
            self.provenance.raw_sha256 == sha256(&serde_json::to_vec(&self.input.raw)?),
            "canonical raw P-code digest mismatch"
        );
        ensure!(
            self.provenance.source_sha256 == sha256(self.source.as_bytes()),
            "source label digest mismatch"
        );
        Ok(self.input.compile()?.prompt)
    }
}

#[derive(Debug, Serialize)]
pub struct Compiled {
    pub representation: &'static str,
    pub prompt: String,
    pub prompt_sha256: String,
    pub raw_canonical_sha256: String,
    pub operations: usize,
    pub instructions: usize,
    pub ssa_values: usize,
    pub effect_steps: usize,
    pub blocks: usize,
    pub phis: usize,
    pub opaque_calls: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Byte {
    value: usize,
    byte: u32,
}
#[derive(Default)]
struct State {
    // Each byte resolves to its latest SSA definition, including partial writes.
    registers: BTreeMap<u64, Byte>,
    temporaries: BTreeMap<u64, Byte>,
    sizes: Vec<u32>,
    lines: Vec<String>,
}
impl State {
    fn define(&mut self, size: u32, expression: String) -> usize {
        let value = self.sizes.len();
        self.sizes.push(size);
        self.lines.push(format!("%{value}:{size}={expression}"));
        value
    }
    fn read(&mut self, v: &Varnode, profile: &Profile) -> Result<String> {
        if v.is_constant {
            // Preserve the bit pattern, including constants represented as signed i64.
            return Ok(format!("#{:x}:{}", constant_bits(v), v.size));
        }
        let register = v.space_id == profile.register_space;
        let mut i = 0;
        while i < v.size {
            let storage = if register {
                &self.registers
            } else {
                &self.temporaries
            };
            if storage.contains_key(&(v.offset + u64::from(i))) {
                i += 1;
                continue;
            }
            ensure!(
                register,
                "temporary read before definition at {:#x}",
                v.offset + u64::from(i)
            );
            let start = i;
            while i < v.size && !self.registers.contains_key(&(v.offset + u64::from(i))) {
                i += 1;
            }
            let id = self.define(
                i - start,
                format!("livein(r{:x})", v.offset + u64::from(start)),
            );
            for j in start..i {
                self.registers.insert(
                    v.offset + u64::from(j),
                    Byte {
                        value: id,
                        byte: j - start,
                    },
                );
            }
        }
        let storage = if register {
            &self.registers
        } else {
            &self.temporaries
        };
        let mut parts = Vec::new();
        let mut i = 0;
        while i < v.size {
            let first = storage[&(v.offset + u64::from(i))];
            let start = i;
            i += 1;
            while i < v.size
                && storage[&(v.offset + u64::from(i))]
                    == (Byte {
                        value: first.value,
                        byte: first.byte + i - start,
                    })
            {
                i += 1;
            }
            if first.byte == 0 && i - start == self.sizes[first.value] {
                parts.push(format!("%{}", first.value));
            } else {
                parts.push(format!("%{}[{}:{}]", first.value, first.byte, i - start));
            }
        }
        // Parts are ordered from least to most significant byte for this LE-only lane.
        Ok(if parts.len() == 1 {
            parts.remove(0)
        } else {
            format!("le({})", parts.join(","))
        })
    }
    fn write(&mut self, v: &Varnode, id: usize, profile: &Profile) {
        let storage = if v.space_id == profile.register_space {
            &mut self.registers
        } else {
            &mut self.temporaries
        };
        for i in 0..v.size {
            storage.insert(v.offset + u64::from(i), Byte { value: id, byte: i });
        }
    }
}

fn constant_bits(v: &Varnode) -> u64 {
    let mask = if v.size >= 8 {
        u64::MAX
    } else {
        (1u64 << (v.size * 8)) - 1
    };
    v.offset & mask
}
fn validate_varnode(v: &Varnode, p: &Profile) -> Result<()> {
    ensure!(
        v.size > 0 && v.size <= 16,
        "varnode width outside v0 range 1..16 bytes"
    );
    if !v.is_constant {
        v.offset
            .checked_add(u64::from(v.size))
            .context("varnode range overflow")?;
    }
    ensure!(
        v.is_constant == (v.space_id == p.constant_space),
        "inconsistent constant-space flag"
    );
    if v.is_constant {
        ensure!(
            v.size <= 8
                && constant_bits(v)
                    == ((v.constant_val as u64)
                        & if v.size == 8 {
                            u64::MAX
                        } else {
                            (1u64 << (8 * v.size)) - 1
                        }),
            "inconsistent constant payload"
        );
    } else {
        ensure!(
            v.space_id == p.register_space || v.space_id == p.unique_space,
            "unsupported direct address space {}",
            v.space_id
        );
    }
    Ok(())
}

/// Returns effect class; opcode identity itself retains the numeric domain.
fn contract(op: &Op, p: &Profile) -> Result<&'static str> {
    let n = op.inputs.len();
    let out = op.output.as_ref().map(|v| v.size);
    let arity = |count, output| -> Result<()> {
        ensure!(
            n == count && out.is_some() == output,
            "invalid arity/output for {}",
            op.opcode
        );
        Ok(())
    };
    let same = || -> Result<()> {
        ensure!(
            op.inputs.iter().all(|v| Some(v.size) == out),
            "width mismatch for {}",
            op.opcode
        );
        Ok(())
    };
    match op.opcode.as_str() {
        "Copy" | "IntNegate" | "Int2Comp" => {
            arity(1, true)?;
            same()?;
        }
        "IntAdd" | "IntSub" | "IntMult" | "IntAnd" | "IntOr" | "IntXor" | "IntDiv" | "IntSDiv"
        | "IntRem" | "IntSRem" => {
            arity(2, true)?;
            same()?;
        }
        "IntLeft" | "IntRight" | "IntSRight" => {
            arity(2, true)?;
            ensure!(
                Some(op.inputs[0].size) == out,
                "shift result width mismatch"
            );
        }
        "IntZExt" | "IntSExt" => {
            arity(1, true)?;
            ensure!(Some(op.inputs[0].size) < out, "extension must widen");
        }
        "IntEqual" | "IntNotEqual" | "IntLess" | "IntLessEqual" | "IntSLess" | "IntSLessEqual"
        | "IntCarry" | "IntSCarry" | "IntSBorrow" => {
            arity(2, true)?;
            ensure!(
                out == Some(1) && op.inputs[0].size == op.inputs[1].size,
                "comparison/carry width mismatch"
            );
        }
        "BoolAnd" | "BoolOr" | "BoolXor" => {
            arity(2, true)?;
            same()?;
            ensure!(out == Some(1), "boolean width must be one");
        }
        "BoolNegate" => {
            arity(1, true)?;
            same()?;
            ensure!(out == Some(1), "boolean width must be one");
        }
        "PopCount" | "LzCount" => {
            arity(1, true)?;
        }
        "Piece" => {
            arity(2, true)?;
            ensure!(
                out == Some(op.inputs[0].size + op.inputs[1].size),
                "piece width mismatch"
            );
        }
        "SubPiece" => {
            arity(2, true)?;
            ensure!(
                op.inputs[1].is_constant
                    && constant_bits(&op.inputs[1])
                        .checked_add(u64::from(out.unwrap()))
                        .is_some_and(|end| end <= u64::from(op.inputs[0].size)),
                "invalid subpiece bounds"
            );
        }
        "Load" | "Store" => {
            let load = op.opcode == "Load";
            arity(if load { 2 } else { 3 }, load)?;
            ensure!(
                op.inputs[0].is_constant
                    && constant_bits(&op.inputs[0]) == p.ram_space
                    && op.inputs[1].size == p.pointer_bytes,
                "unsupported memory space/pointer width"
            );
            return Ok(if load {
                "read_ram_may_trap"
            } else {
                "write_ram_may_trap"
            });
        }
        "Return" => {
            arity(1, false)?;
            ensure!(
                op.inputs[0].size == p.pointer_bytes && !op.inputs[0].is_constant,
                "invalid return target"
            );
            return Ok("return");
        }
        other => bail!(
            "unsupported opcode {other}; branches, calls, userops, floats, and analyzed P-code are outside v0"
        ),
    }
    Ok(
        if matches!(
            op.opcode.as_str(),
            "IntDiv" | "IntSDiv" | "IntRem" | "IntSRem"
        ) {
            "arithmetic_may_trap"
        } else {
            "pure"
        },
    )
}

pub fn compile(raw: &RawDump, profile: &Profile) -> Result<Compiled> {
    profile.validate()?;
    ensure!(
        raw.stop_reason == "terminal_control_flow",
        "incomplete lift: {}",
        raw.stop_reason
    );
    ensure!(
        raw.decoded_instructions > 0 && raw.decoded_instructions <= 512,
        "invalid instruction count"
    );
    ensure!(
        raw.template_source_counts.len() == 1
            && raw.template_source_counts.get("sla_construct_tpl")
                == Some(&raw.decoded_instructions),
        "missing/unsupported template provenance"
    );
    ensure!(
        raw.pcode.blocks.len() == 1,
        "v0 requires exactly one block; CFG SSA is not implemented"
    );
    let block = &raw.pcode.blocks[0];
    ensure!(
        block.index == 0
            && block.successors.is_empty()
            && !block.ops.is_empty()
            && block.ops.len() <= 4096,
        "invalid straight-line block"
    );
    let address = |s: &str| u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16);
    ensure!(
        address(&raw.entry_address)? == block.start_address
            && address(&raw.decode_address)? == block.start_address
            && block.ops[0].address == block.start_address,
        "entry/decode/block address mismatch"
    );
    let mut state = State::default();
    state.lines.push(format!(
        "{REPRESENTATION} x86_64 le ptr8 ram_word1\nblock0 effect0"
    ));
    let mut previous = None;
    let mut instructions = 0;
    let mut effect = 0;
    for (index, op) in block.ops.iter().enumerate() {
        if previous != Some(op.address) {
            ensure!(
                previous.is_none_or(|a| op.address > a),
                "instruction addresses must be strictly increasing"
            );
            state.temporaries.clear();
            state.lines.push(format!("insn{instructions}"));
            instructions += 1;
            previous = Some(op.address);
        }
        ensure!(
            op.seq_num as usize == index,
            "noncontiguous raw sequence number"
        );
        for v in op.inputs.iter().chain(op.output.iter()) {
            validate_varnode(v, profile)?;
        }
        if let Some(v) = &op.output {
            ensure!(!v.is_constant, "constant output is forbidden");
        }
        let class = contract(op, profile).with_context(|| format!("op {index} {}", op.opcode))?;
        ensure!(
            (class == "return") == (index + 1 == block.ops.len()),
            "only the final operation may return, and a return is required"
        );
        let inputs = op
            .inputs
            .iter()
            .map(|v| state.read(v, profile))
            .collect::<Result<Vec<_>>>()?;
        let mut expression = format!("{}({})", op.opcode, inputs.join(","));
        if class != "pure" {
            expression.push_str(&format!(" !{class} e{effect}->e{}", effect + 1));
            effect += 1;
        }
        if let Some(out) = &op.output {
            if out.space_id == profile.register_space {
                expression.push_str(&format!(" @r{:x}", out.offset));
            }
            let value = state.define(out.size, expression);
            state.write(out, value, profile);
        } else {
            state.lines.push(expression);
        }
    }
    ensure!(
        instructions <= raw.decoded_instructions,
        "more operation-bearing instructions than decoded instructions"
    );
    // NOPs may contribute decoded instructions without contributing P-code.
    let prompt = state.lines.join("\n") + "\n";
    Ok(Compiled {
        representation: REPRESENTATION,
        prompt_sha256: sha256(prompt.as_bytes()),
        raw_canonical_sha256: sha256(&serde_json::to_vec(raw)?),
        prompt,
        operations: block.ops.len(),
        instructions,
        ssa_values: state.sizes.len(),
        effect_steps: effect,
        blocks: 1,
        phis: 0,
        opaque_calls: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> Profile {
        serde_json::from_str(include_str!("../../../configs/lifter/fission-x86_64.json")).unwrap()
    }
    fn add() -> RawDump {
        serde_json::from_str(include_str!("../../../fixtures/pcode/forge_add.raw.json")).unwrap()
    }
    #[test]
    fn real_lifts_keep_numeric_domains_and_effect_order() -> Result<()> {
        for (json, div, rem) in [
            (
                include_str!("../../../fixtures/pcode/forge_udiv.raw.json"),
                "IntDiv",
                "IntRem",
            ),
            (
                include_str!("../../../fixtures/pcode/forge_sdiv.raw.json"),
                "IntSDiv",
                "IntSRem",
            ),
        ] {
            let raw: RawDump = serde_json::from_str(json)?;
            let c = compile(&raw, &profile())?;
            assert!(c.prompt.contains(div) && c.prompt.contains(rem));
            assert_eq!(c.effect_steps, 4); // div, rem, return-address load, return
            assert!(c.prompt.contains("!arithmetic_may_trap e0->e1"));
            assert!(c.prompt.contains("!return e3->e4"));
        }
        let mut raw = add();
        let mut store = raw.pcode.blocks[0].ops[4].clone();
        store.opcode = "Store".into();
        store.inputs.push(store.output.take().unwrap());
        raw.pcode.blocks[0].ops.insert(5, store);
        for (i, op) in raw.pcode.blocks[0].ops.iter_mut().enumerate() {
            op.seq_num = i as u32;
        }
        let c = compile(&raw, &profile())?;
        assert!(c.prompt.contains("!read_ram_may_trap e0->e1"));
        assert!(c.prompt.contains("!write_ram_may_trap e1->e2"));
        assert!(c.prompt.contains("!return e2->e3"));
        Ok(())
    }
    #[test]
    fn relocation_temp_renaming_and_advisory_text_do_not_change_prompt() -> Result<()> {
        let p = profile();
        let mut raw = add();
        let expected = compile(&raw, &p)?.prompt;
        raw.entry_address = format!("0x{:x}", raw.pcode.blocks[0].start_address + 0x100000);
        raw.decode_address = raw.entry_address.clone();
        raw.pcode.blocks[0].start_address += 0x100000;
        for op in &mut raw.pcode.blocks[0].ops {
            op.address += 0x100000;
            op.asm_mnemonic = Some("PRIVATE_SOURCE_LABEL_SHOULD_NOT_LEAK".into());
            for v in op.inputs.iter_mut().chain(op.output.iter_mut()) {
                if v.space_id == p.unique_space {
                    v.offset += 0x100000;
                }
            }
        }
        assert_eq!(compile(&raw, &p)?.prompt, expected);
        Ok(())
    }
    #[test]
    fn unsupported_and_malformed_lifts_fail_closed() -> Result<()> {
        for json in [
            include_str!("../../../fixtures/pcode/forge_branch.raw.json"),
            include_str!("../../../fixtures/pcode/forge_call.raw.json"),
        ] {
            assert!(compile(&serde_json::from_str(json)?, &profile()).is_err());
        }
        let reject = |raw: RawDump| assert!(compile(&raw, &profile()).is_err());
        let mut r = add();
        r.stop_reason = "instruction_limit".into();
        reject(r);
        let mut r = add();
        r.pcode.blocks[0].ops[0].opcode = "CallOther".into();
        reject(r);
        let mut r = add();
        r.pcode.blocks[0].ops[0].opcode = "FloatDiv".into();
        reject(r);
        let mut r = add();
        r.pcode.blocks[0].ops[0].inputs[1].constant_val = 99;
        reject(r);
        let mut r = add();
        r.pcode.blocks[0].ops[0].inputs[0].size = 4;
        reject(r);
        let mut r = add();
        r.pcode.blocks[0].ops[0].seq_num = 1;
        reject(r);
        let mut r = add();
        r.pcode.blocks[0].ops.last_mut().unwrap().inputs[0] =
            r.pcode.blocks[0].ops[0].output.clone().unwrap();
        reject(r); // A temporary cannot survive into another machine instruction.
        let mut p = profile();
        p.endianness = "big".into();
        assert!(compile(&add(), &p).is_err());
        Ok(())
    }
    #[test]
    fn partial_register_writes_match_independent_byte_array() -> Result<()> {
        let p = profile();
        let mut s = State::default();
        let mut values = Vec::<(u64, u32)>::new();
        let mut reference = [0u8; 8];
        let mut seed = 0x7845_u64;
        // Evaluate serialized references independently of State's byte bookkeeping.
        fn read_expr(expr: &str, values: &[(u64, u32)]) -> u64 {
            if let Some(inner) = expr.strip_prefix("le(").and_then(|s| s.strip_suffix(')')) {
                let mut result = 0u64;
                let mut shift = 0;
                for part in inner.split(',') {
                    let (value, size) = slice(part, values);
                    result |= value << shift;
                    shift += size * 8;
                }
                return result;
            }
            slice(expr, values).0
        }
        fn slice(expr: &str, values: &[(u64, u32)]) -> (u64, u32) {
            let text = expr.strip_prefix('%').unwrap();
            if let Some((id, range)) = text.split_once('[') {
                let (lo, size) = range.strip_suffix(']').unwrap().split_once(':').unwrap();
                let lo: u32 = lo.parse().unwrap();
                let size: u32 = size.parse().unwrap();
                let mask = if size == 8 {
                    u64::MAX
                } else {
                    (1u64 << (size * 8)) - 1
                };
                (
                    (values[id.parse::<usize>().unwrap()].0 >> (lo * 8)) & mask,
                    size,
                )
            } else {
                values[text.parse::<usize>().unwrap()]
            }
        }
        // Initial full definition, then 100 overlapping writes with deterministic bits.
        for iteration in 0..101 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let start = if iteration == 0 { 0 } else { seed as usize % 8 };
            let size = if iteration == 0 {
                8
            } else {
                1 + (seed >> 16) as usize % (8 - start)
            };
            let v = Varnode {
                space_id: p.register_space,
                offset: start as u64,
                size: size as u32,
                is_constant: false,
                constant_val: 0,
            };
            let mask = if size == 8 {
                u64::MAX
            } else {
                (1u64 << (size * 8)) - 1
            };
            let value = seed & mask;
            let id = s.define(size as u32, "test_write".into());
            values.push((value, size as u32));
            s.write(&v, id, &p);
            reference[start..start + size].copy_from_slice(&value.to_le_bytes()[..size]);
            for offset in 0..8 {
                for width in 1..=8 - offset {
                    let mut want = [0u8; 8];
                    want[..width].copy_from_slice(&reference[offset..offset + width]);
                    let read = Varnode {
                        offset: offset as u64,
                        size: width as u32,
                        ..v.clone()
                    };
                    assert_eq!(
                        read_expr(&s.read(&read, &p)?, &values),
                        u64::from_le_bytes(want)
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn constants_preserve_twos_complement_and_reject_overflowing_subpieces() -> Result<()> {
        let mut raw = add();
        raw.pcode.blocks[0].ops[0].inputs[1].offset = u64::MAX;
        raw.pcode.blocks[0].ops[0].inputs[1].constant_val = -1;
        assert!(
            compile(&raw, &profile())?
                .prompt
                .contains("#ffffffffffffffff:8")
        );
        raw.pcode.blocks[0].ops[2].inputs[1].size = 8;
        raw.pcode.blocks[0].ops[2].inputs[1].offset = u64::MAX;
        raw.pcode.blocks[0].ops[2].inputs[1].constant_val = -1;
        assert!(compile(&raw, &profile()).is_err());
        Ok(())
    }
    #[test]
    fn labels_and_ids_cannot_enter_prompt_and_raw_mutation_is_detected() -> Result<()> {
        let text = include_str!("../../../fixtures/pcode/pairs.jsonl");
        let mut pair: Pair = serde_json::from_str(text.lines().next().unwrap())?;
        let expected = pair.prompt()?;
        pair.id = "secret_label".into();
        pair.source = "int private_reference(void) { return 999; }".into();
        pair.provenance.source_sha256 = sha256(pair.source.as_bytes());
        assert_eq!(pair.prompt()?, expected);
        pair.input.raw.pcode.blocks[0].ops[0].opcode = "IntDiv".into();
        assert!(pair.prompt().is_err());
        Ok(())
    }
}
