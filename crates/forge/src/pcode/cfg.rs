//! P-code+ v1: operation-derived CFG, unpruned cell SSA, conservative opaque calls.
use super::*;

pub mod audit;
pub mod slice;

pub const REPRESENTATION: &str = "pcode-plus-ssa-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Cell {
    pub temporary_instruction: Option<u64>,
    pub offset: u64,
    pub size: u32,
}
impl Cell {
    fn label(&self) -> String {
        format!(
            "{}{:x}:{}",
            if self.temporary_instruction.is_some() {
                "t"
            } else {
                "r"
            },
            self.offset,
            self.size
        )
    }
}
#[derive(Debug, Serialize)]
pub struct Phi {
    pub value: usize,
    pub size: u32,
    pub storage: Cell,
    pub incoming: BTreeMap<String, String>,
}
#[derive(Debug, Serialize)]
pub struct SsaBlock {
    pub index: usize,
    pub phis: Vec<Phi>,
    pub body: Vec<String>,
    pub successors: Vec<usize>,
    pub exit_registers: BTreeMap<u64, String>,
}
#[derive(Debug, Serialize)]
pub struct Program {
    pub liveins: Vec<String>,
    pub blocks: Vec<SsaBlock>,
    pub sizes: Vec<u32>,
    pub effect_steps: usize,
    pub opaque_calls: usize,
    pub instructions: usize,
    pub operations: usize,
}
#[derive(Clone)]
struct GraphBlock {
    ops: Vec<Op>,
    successors: Vec<usize>,
    taken: Option<usize>,
}

fn direct_target(op: &Op, p: &Profile) -> Result<u64> {
    let target = op
        .inputs
        .first()
        .context("control transfer missing target")?;
    ensure!(
        target.space_id == p.ram_space && !target.is_constant && target.size == p.pointer_bytes,
        "unsupported control target; instruction-relative constant branches require a separate lane"
    );
    Ok(target.offset)
}
fn graph(raw: &RawDump, p: &Profile) -> Result<Vec<GraphBlock>> {
    ensure!(
        raw.stop_reason == "terminal_control_flow",
        "incomplete lift: {}",
        raw.stop_reason
    );
    ensure!(
        (1..=512).contains(&raw.decoded_instructions),
        "invalid instruction count"
    );
    ensure!(
        raw.template_source_counts.len() == 1
            && raw.template_source_counts.get("sla_construct_tpl")
                == Some(&raw.decoded_instructions),
        "unsupported template provenance"
    );
    // Upstream blocks are provenance only: a CMOV may have an interior CBranch.
    let mut ops: Vec<Op> = raw
        .pcode
        .blocks
        .iter()
        .flat_map(|b| b.ops.iter().cloned())
        .collect();
    ensure!(
        !ops.is_empty() && ops.len() <= 4096,
        "invalid operation count"
    );
    ops.sort_by_key(|o| o.seq_num);
    let mut first = BTreeMap::new();
    let mut previous = None;
    for (i, op) in ops.iter().enumerate() {
        ensure!(
            op.seq_num as usize == i,
            "duplicate/noncontiguous operation sequence"
        );
        ensure!(
            previous.is_none_or(|a| a <= op.address),
            "instruction addresses regress"
        );
        previous = Some(op.address);
        first.entry(op.address).or_insert(i);
    }
    let address = |s: &str| u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16);
    ensure!(
        address(&raw.entry_address)? == ops[0].address
            && address(&raw.decode_address)? == ops[0].address,
        "entry address mismatch"
    );
    ensure!(
        first.len() <= raw.decoded_instructions,
        "decoded instruction count mismatch"
    );
    let mut leaders = BTreeSet::from([0]);
    for (i, op) in ops.iter().enumerate() {
        match op.opcode.as_str() {
            "Branch" | "CBranch" => {
                let target = direct_target(op, p)?;
                leaders.insert(
                    *first
                        .get(&target)
                        .context("unresolved branch target; external/tail jumps are unsupported")?,
                );
                if i + 1 < ops.len() {
                    leaders.insert(i + 1);
                }
            }
            "Return" | "Call" | "CallInd" => {
                if i + 1 < ops.len() {
                    leaders.insert(i + 1);
                }
            }
            "BranchInd" => bail!("unresolved indirect branch"),
            _ => {}
        }
    }
    let starts: Vec<usize> = leaders.into_iter().collect();
    ensure!(starts.len() <= 128, "CFG exceeds 128-block limit");
    let mut block_for_op = vec![0; ops.len()];
    for (b, &start) in starts.iter().enumerate() {
        let end = starts.get(b + 1).copied().unwrap_or(ops.len());
        block_for_op[start..end].fill(b);
    }
    let mut blocks = Vec::new();
    for (b, &start) in starts.iter().enumerate() {
        let end = starts.get(b + 1).copied().unwrap_or(ops.len());
        let last = &ops[end - 1];
        let mut successors = Vec::new();
        let mut taken = None;
        if matches!(last.opcode.as_str(), "Branch" | "CBranch") {
            let target = block_for_op[first[&direct_target(last, p)?]];
            taken = Some(target);
            successors.push(target);
        }
        if !matches!(last.opcode.as_str(), "Branch" | "Return") {
            ensure!(
                end < ops.len(),
                "unterminated CFG or missing call continuation"
            );
            successors.push(block_for_op[end]);
        }
        successors.sort_unstable();
        successors.dedup();
        blocks.push(GraphBlock {
            ops: ops[start..end].to_vec(),
            successors,
            taken,
        });
    }
    let mut visited = BTreeSet::new();
    let mut pending = vec![0];
    while let Some(b) = pending.pop() {
        if visited.insert(b) {
            pending.extend(&blocks[b].successors);
        }
    }
    ensure!(
        visited.len() == blocks.len(),
        "unreachable operations in dump; explicit pruning/provenance required"
    );
    ensure!(
        blocks
            .iter()
            .any(|b| b.ops.last().unwrap().opcode == "Return"),
        "no return in bounded CFG"
    );
    Ok(blocks)
}

fn partition(nodes: impl Iterator<Item = Varnode>, instruction: Option<u64>) -> Vec<Cell> {
    let nodes: Vec<_> = nodes.collect();
    let boundaries: BTreeSet<_> = nodes
        .iter()
        .flat_map(|v| [v.offset, v.offset + u64::from(v.size)])
        .collect();
    let boundaries: Vec<_> = boundaries.into_iter().collect();
    boundaries
        .windows(2)
        .filter_map(|w| {
            nodes
                .iter()
                .any(|v| v.offset <= w[0] && w[1] <= v.offset + u64::from(v.size))
                .then_some(Cell {
                    temporary_instruction: instruction,
                    offset: w[0],
                    size: (w[1] - w[0]) as u32,
                })
        })
        .collect()
}
fn seed(state: &mut State, cell: Cell, value: usize) {
    let storage = if cell.temporary_instruction.is_some() {
        &mut state.temporaries
    } else {
        &mut state.registers
    };
    for byte in 0..cell.size {
        storage.insert(cell.offset + u64::from(byte), Byte { value, byte });
    }
}
fn refs(text: &str) -> Vec<usize> {
    text.split('%')
        .skip(1)
        .filter_map(|s| {
            let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse().ok()
        })
        .collect()
}
fn cell_expression(state: &mut State, cell: Cell, profile: &Profile) -> Result<String> {
    let storage = if cell.temporary_instruction.is_some() {
        &state.temporaries
    } else {
        &state.registers
    };
    ensure!(
        (0..cell.size).all(|i| storage.contains_key(&(cell.offset + u64::from(i)))),
        "undefined temporary on CFG edge"
    );
    state.read(
        &Varnode {
            space_id: if cell.temporary_instruction.is_some() {
                profile.unique_space
            } else {
                profile.register_space
            },
            offset: cell.offset,
            size: cell.size,
            is_constant: false,
            constant_val: 0,
        },
        profile,
    )
}

pub fn program(raw: &RawDump, profile: &Profile) -> Result<Program> {
    profile.validate()?;
    let blocks = graph(raw, profile)?;
    let mut registers = Vec::new();
    let mut temporaries: BTreeMap<u64, Vec<Varnode>> = BTreeMap::new();
    for block in &blocks {
        for op in &block.ops {
            let special = matches!(op.opcode.as_str(), "Branch" | "CBranch" | "Call");
            for v in op
                .inputs
                .iter()
                .skip(usize::from(special))
                .chain(op.output.iter())
            {
                validate_varnode(v, profile)?;
                if v.space_id == profile.register_space {
                    registers.push(v.clone());
                } else if v.space_id == profile.unique_space {
                    temporaries.entry(op.address).or_default().push(v.clone());
                }
            }
            if let Some(v) = &op.output {
                ensure!(!v.is_constant, "constant output is forbidden");
            }
        }
    }
    let cells = partition(registers.into_iter(), None);
    ensure!(cells.len() <= 256, "too many register cells");
    let temp_cells: BTreeMap<_, _> = temporaries
        .into_iter()
        .map(|(a, n)| (a, partition(n.into_iter(), Some(a))))
        .collect();
    let mut sizes = Vec::new();
    let mut params = Vec::<BTreeMap<Cell, usize>>::new();
    let mut owner = BTreeMap::new();
    for (b, block) in blocks.iter().enumerate() {
        let mut map = BTreeMap::new();
        for &cell in cells
            .iter()
            .chain(temp_cells.get(&block.ops[0].address).into_iter().flatten())
        {
            let value = sizes.len();
            sizes.push(cell.size);
            map.insert(cell, value);
            owner.insert(value, (b, cell));
        }
        params.push(map);
    }
    let mut bodies = Vec::new();
    let mut ends = Vec::new();
    let mut exits = Vec::new();
    let mut effects = Vec::new();
    let mut required = BTreeSet::new();
    let mut call_count = 0;
    let mut instruction_addresses = BTreeSet::new();
    for (b, block) in blocks.iter().enumerate() {
        let mut state = State {
            sizes: sizes.clone(),
            ..State::default()
        };
        let instruction_start = raw
            .pcode
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .filter(|o| o.address == block.ops[0].address)
            .map(|o| o.seq_num)
            .min()
            .unwrap();
        for (&cell, &value) in &params[b] {
            // A machine-instruction entry starts with undefined temporaries.
            // Only a block split inside that instruction can receive temp phis.
            if cell.temporary_instruction.is_none() || block.ops[0].seq_num != instruction_start {
                seed(&mut state, cell, value);
            }
        }
        let mut previous = Some(block.ops[0].address);
        let mut effect = 0;
        for (position, op) in block.ops.iter().enumerate() {
            instruction_addresses.insert(op.address);
            if previous != Some(op.address) {
                state.temporaries.clear();
                previous = Some(op.address);
            }
            let control = matches!(
                op.opcode.as_str(),
                "Branch" | "CBranch" | "Call" | "CallInd"
            );
            if control {
                ensure!(
                    op.output.is_none(),
                    "control transfer cannot have a scalar output"
                );
                let target = if op.opcode == "CallInd" {
                    ensure!(
                        op.inputs.len() == 1 && op.inputs[0].size == profile.pointer_bytes,
                        "invalid indirect call"
                    );
                    state.read(&op.inputs[0], profile)?
                } else {
                    let target = direct_target(op, profile)?;
                    if op.opcode == "Call" {
                        format!("ram@{target:x}")
                    } else {
                        format!("b{}", block.taken.context("branch has no target")?)
                    }
                };
                match op.opcode.as_str() {
                    "Branch" => {
                        ensure!(op.inputs.len() == 1, "invalid branch arity");
                        state.lines.push(format!("Jump({target})"));
                    }
                    "CBranch" => {
                        ensure!(
                            op.inputs.len() == 2 && op.inputs[1].size == 1,
                            "invalid conditional branch"
                        );
                        let condition = state.read(&op.inputs[1], profile)?;
                        let fall = b + 1;
                        ensure!(
                            block.successors.contains(&fall),
                            "conditional fallthrough missing"
                        );
                        state
                            .lines
                            .push(format!("If({condition},{target},b{fall})"));
                    }
                    "Call" | "CallInd" => {
                        ensure!(
                            op.inputs.len() == 1,
                            "analyzed call parameters are unsupported"
                        );
                        let call = call_count;
                        call_count += 1;
                        let mut reads = Vec::new();
                        for &cell in &cells {
                            reads.push(format!(
                                "{}={}",
                                cell.label(),
                                cell_expression(&mut state, cell, profile)?
                            ));
                        }
                        state.lines.push(format!("call{call}=Opaque{}({target},regs({})) !may_read_write_all_regs_ram_may_trap_or_not_return e{b}_{effect}->e{b}_{}",op.opcode,reads.join(","),effect+1));
                        effect += 1;
                        // No ABI is guessed: every observed register cell, including SP,
                        // may change. This represents only an unknown returning continuation.
                        for &cell in &cells {
                            let value = state
                                .define(cell.size, format!("havoc(call{call},{})", cell.label()));
                            seed(&mut state, cell, value);
                        }
                        state.temporaries.clear();
                        state.lines.push(format!("MayReturn(b{})", b + 1));
                    }
                    _ => unreachable!(),
                }
                ensure!(
                    position + 1 == block.ops.len(),
                    "control transfer must terminate reconstructed block"
                );
                continue;
            }
            let class = contract(op, profile)?;
            let inputs = op
                .inputs
                .iter()
                .map(|v| state.read(v, profile))
                .collect::<Result<Vec<_>>>()?;
            let mut expression = format!("{}({})", op.opcode, inputs.join(","));
            if class != "pure" {
                expression.push_str(&format!(" !{class} e{b}_{effect}->e{b}_{}", effect + 1));
                effect += 1;
            }
            if let Some(out) = &op.output {
                if out.space_id == profile.register_space {
                    expression.push_str(&format!(" @r{:x}", out.offset));
                }
                let id = state.define(out.size, expression);
                state.write(out, id, profile);
            } else {
                state.lines.push(expression);
            }
        }
        let mut exit = BTreeMap::new();
        if block.ops.last().unwrap().opcode == "Return" {
            for &cell in &cells {
                exit.insert(cell.offset, cell_expression(&mut state, cell, profile)?);
            }
        } else if !matches!(
            block.ops.last().unwrap().opcode.as_str(),
            "Branch" | "CBranch" | "Call" | "CallInd"
        ) {
            state.lines.push(format!("Jump(b{})", b + 1));
        }
        for line in state.lines.iter().chain(exit.values()) {
            for value in refs(line) {
                if owner.contains_key(&value) {
                    required.insert(value);
                }
            }
        }
        sizes = state.sizes.clone();
        effects.push(effect);
        bodies.push(state.lines.clone());
        ends.push(state);
        exits.push(exit);
    }
    let mut predecessors = vec![Vec::new(); blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for &s in &block.successors {
            predecessors[s].push(b);
        }
    }
    let mut phi_inputs = BTreeMap::<usize, BTreeMap<String, String>>::new();
    let mut liveins = Vec::new();
    let mut pending: Vec<_> = required.iter().copied().collect();
    while let Some(value) = pending.pop() {
        if phi_inputs.contains_key(&value) {
            continue;
        }
        let (b, cell) = owner[&value];
        let mut incoming = BTreeMap::new();
        if b == 0 {
            ensure!(
                cell.temporary_instruction.is_none(),
                "temporary read before definition at entry"
            );
            let root = sizes.len();
            sizes.push(cell.size);
            liveins.push(format!("%{root}:{}=livein({})", cell.size, cell.label()));
            incoming.insert("entry".into(), format!("%{root}"));
        }
        for &pred in &predecessors[b] {
            if let Some(address) = cell.temporary_instruction {
                ensure!(
                    blocks[pred].ops.last().unwrap().address == address,
                    "temporary crosses machine-instruction scope"
                );
            }
            let expression = cell_expression(&mut ends[pred], cell, profile)?;
            for reference in refs(&expression) {
                if owner.contains_key(&reference) && !phi_inputs.contains_key(&reference) {
                    pending.push(reference);
                }
            }
            incoming.insert(format!("b{pred}"), expression);
        }
        ensure!(!incoming.is_empty(), "phi without incoming values");
        phi_inputs.insert(value, incoming);
    }
    let mut result = Vec::new();
    for (b, block) in blocks.iter().enumerate() {
        let mut phis = Vec::new();
        for (&cell, &value) in &params[b] {
            if let Some(incoming) = phi_inputs.remove(&value) {
                phis.push(Phi {
                    value,
                    size: cell.size,
                    storage: cell,
                    incoming,
                });
            }
        }
        result.push(SsaBlock {
            index: b,
            phis,
            body: std::mem::take(&mut bodies[b]),
            successors: block.successors.clone(),
            exit_registers: std::mem::take(&mut exits[b]),
        });
    }
    // Effect phis are conservative path/loop merges, without alias analysis.
    for block in &mut result {
        let b = block.index;
        let mut incoming = Vec::new();
        if b == 0 {
            incoming.push("entry:effect0".into());
        }
        for &pred in &predecessors[b] {
            incoming.push(format!("b{pred}:e{pred}_{}", effects[pred]));
        }
        block
            .body
            .insert(0, format!("e{b}_0=effect_phi({})", incoming.join(",")));
    }
    Ok(Program {
        liveins,
        blocks: result,
        sizes,
        effect_steps: effects.iter().sum(),
        opaque_calls: call_count,
        instructions: instruction_addresses.len(),
        operations: blocks.iter().map(|b| b.ops.len()).sum(),
    })
}

pub fn compile(raw: &RawDump, profile: &Profile) -> Result<Compiled> {
    let program = program(raw, profile)?;
    render(&program, raw, REPRESENTATION)
}
fn render(program: &Program, raw: &RawDump, representation: &'static str) -> Result<Compiled> {
    let mut lines = vec![format!(
        "{representation} x86_64 le ptr8 ram_word1 call_contract=opaque_all_regs_ram"
    )];
    lines.extend(program.liveins.iter().cloned());
    let mut phis = 0;
    for block in &program.blocks {
        lines.push(format!("block{}", block.index));
        for phi in &block.phis {
            phis += 1;
            let incoming = phi
                .incoming
                .iter()
                .map(|(p, v)| format!("{p}:{v}"))
                .collect::<Vec<_>>()
                .join(",");
            let storage = if phi.storage.temporary_instruction.is_some() {
                "temporary".into()
            } else {
                phi.storage.label()
            };
            lines.push(format!(
                "%{}:{}=phi({incoming}) @{storage}",
                phi.value, phi.size
            ));
        }
        lines.extend(block.body.iter().cloned());
        if !block.exit_registers.is_empty() {
            lines.push(format!(
                "exit_regs({})",
                block
                    .exit_registers
                    .iter()
                    .map(|(r, v)| format!("r{r:x}={v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }
    let prompt = lines.join("\n") + "\n";
    Ok(Compiled {
        representation,
        prompt_sha256: sha256(prompt.as_bytes()),
        raw_canonical_sha256: sha256(&serde_json::to_vec(raw)?),
        prompt,
        operations: program.operations,
        instructions: program.instructions,
        ssa_values: program.liveins.len()
            + phis
            + program
                .blocks
                .iter()
                .flat_map(|b| &b.body)
                .filter(|line| line.starts_with('%'))
                .count(),
        effect_steps: program.effect_steps,
        blocks: program.blocks.len(),
        phis,
        opaque_calls: program.opaque_calls,
    })
}

#[cfg(test)]
mod tests;
