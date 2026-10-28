//! Dump raw SLEIGH P-code with Kuna's Rust engine, in the same JSON shape as DumpPcode.java.
//!
//! usage: kuna-pcode-dump --sla X.sla --image BYTES --base 0xVADDR --entries LIST --program NAME --out FILE
//! LIST has one `symbol 0xaddr` pair per line. Each function is lifted instruction by
//! instruction until its first RETURN; no analysis, decompiler or label is consulted.
use std::collections::HashMap;
use std::fs;

use kuna_base::address::Address;
use kuna_base::error::{KunaError, KunaResult};
use kuna_num::opcodes::{get_opname, OpCode};
use kuna_num::pcoderaw::VarnodeData;
use kuna_sleigh::globalcontext::ContextInternal;
use kuna_sleigh::loadimage::LoadImage;
use kuna_sleigh::sleigh::Sleigh;
use kuna_sleigh::translate::{PcodeEmit, Translate};

struct Image {
    base: u64,
    bytes: Vec<u8>,
}

impl LoadImage for Image {
    fn get_file_name(&self) -> &str {
        "kuna-pcode-dump"
    }
    fn load_fill(&mut self, ptr: &mut [u8], addr: &Address) -> KunaResult<()> {
        let offset = addr.get_offset();
        if offset < self.base || (offset - self.base) as usize >= self.bytes.len() {
            return Err(KunaError::data_unavail("address outside the image"));
        }
        let start = (offset - self.base) as usize;
        for (i, slot) in ptr.iter_mut().enumerate() {
            *slot = self.bytes.get(start + i).copied().unwrap_or(0);
        }
        Ok(())
    }
    fn get_arch_type(&self) -> Vec<u8> {
        Vec::new()
    }
    fn adjust_vma(&mut self, _adjust: i64) {}
}

struct Op {
    address: u64,
    mnemonic: String,
    output: Option<String>,
    inputs: Vec<String>,
    is_return: bool,
}

struct Collect {
    ops: Vec<Op>,
}

fn varnode(v: &VarnodeData) -> String {
    let space = v.space.as_ref().map(|s| s.get_name().to_string()).unwrap_or_default();
    // Offsets are printed as signed 64-bit values, like Ghidra's Java long.
    format!("[\"{}\",{},{}]", space, v.offset as i64, v.size)
}

impl PcodeEmit for Collect {
    fn dump(&mut self, addr: &Address, opc: OpCode, outvar: Option<&VarnodeData>, vars: &[VarnodeData]) {
        let name = get_opname(opc).trim_start_matches("CPUI_").to_string();
        self.ops.push(Op {
            address: addr.get_offset(),
            mnemonic: name,
            output: outvar.map(varnode),
            inputs: vars.iter().map(varnode).collect(),
            is_return: opc == OpCode::CPUI_RETURN,
        });
    }
}

fn arg(args: &HashMap<String, String>, key: &str) -> String {
    args.get(key).unwrap_or_else(|| panic!("missing --{key}")).clone()
}

fn parse_u64(text: &str) -> u64 {
    match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).expect("hex address"),
        None => text.parse().expect("decimal address"),
    }
}

fn main() {
    let mut args = HashMap::new();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    for pair in raw.chunks(2) {
        args.insert(pair[0].trim_start_matches("--").to_string(), pair[1].clone());
    }
    let base = parse_u64(&arg(&args, "base"));
    let sla = fs::read(arg(&args, "sla")).expect("read .sla");
    let image = Image { base, bytes: fs::read(arg(&args, "image")).expect("read image") };
    let mut sleigh = Sleigh::new(Box::new(image), Box::new(ContextInternal::new()));
    sleigh.initialize_from_sla(&sla).expect("initialize from sla");
    for (name, value) in [("addrsize", 2), ("opsize", 1), ("rexprefix", 0), ("longMode", 1), ("bit64", 1)] {
        sleigh.set_context_default(name, value);
    }
    let ram = std::rc::Rc::clone(sleigh.base().manager().get_space_by_name("ram").expect("ram space"));
    let mut functions = Vec::new();
    for line in fs::read_to_string(arg(&args, "entries")).expect("read entries").lines() {
        let mut parts = line.split_whitespace();
        let (Some(name), Some(entry)) = (parts.next(), parts.next()) else { continue };
        let entry = parse_u64(entry);
        let mut collect = Collect { ops: Vec::new() };
        let mut address = Address::new(std::rc::Rc::clone(&ram), entry);
        let mut error = String::new();
        for _ in 0..256 {
            match sleigh.one_instruction(&mut collect, &address) {
                Ok(length) => {
                    if collect.ops.last().is_some_and(|op| op.is_return) {
                        break;
                    }
                    address = &address + i64::from(length);
                }
                Err(e) => {
                    error = e.to_string().replace('"', "'");
                    break;
                }
            }
        }
        let ops: Vec<String> = collect
            .ops
            .iter()
            .map(|op| {
                format!(
                    "{{\"address\":{},\"mnemonic\":\"{}\",\"output\":{},\"inputs\":[{}]}}",
                    op.address,
                    op.mnemonic,
                    op.output.clone().unwrap_or_else(|| "null".into()),
                    op.inputs.join(",")
                )
            })
            .collect();
        functions.push(format!(
            "\"{}\":{{\"entry\":{},\"error\":\"{}\",\"ops\":[{}]}}",
            name,
            entry,
            error,
            ops.join(",")
        ));
    }
    let program = arg(&args, "program");
    let out = format!(
        "{{\"program\":\"{}\",\"language\":\"x86:LE:64:default\",\"image_base\":0,\"functions\":{{{}}}}}",
        program,
        functions.join(",")
    );
    fs::write(arg(&args, "out"), out).expect("write output");
    println!("wrote {} functions={}", arg(&args, "out"), functions.len());
}
