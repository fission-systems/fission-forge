// Standalone helper: link against the pinned Fission build, without changing it.
// Emits role IDs from Fission's canonical compiled-SLA parser, not guessed IDs.
use fission_sleigh::compiler::sla::load_construct_templates_from_sla;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected SLA path")?;
    let sla = load_construct_templates_from_sla(path)?;
    for s in sla.spaces.values() {
        println!("{} {} {} {}", s.name, s.index, s.word_size, s.addr_size);
    }
    Ok(())
}
