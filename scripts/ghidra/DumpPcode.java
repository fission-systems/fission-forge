// Dump raw (not high) P-code of symbols with a given prefix, one JSON file per program.
// Usage: analyzeHeadless <proj> <name> -import <elf> -noanalysis -scriptPath scripts/ghidra \
//        -postScript DumpPcode.java <outputDir> <symbolPrefix>
// Instructions are walked from the symbol until the first terminal flow; no
// analyzer, decompiler or source label is consulted.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolIterator;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class DumpPcode extends GhidraScript {
    private static String varnode(Varnode v) {
        if (v == null) {
            return "null";
        }
        String space = v.getAddress().getAddressSpace().getName();
        return "[\"" + space + "\"," + v.getOffset() + "," + v.getSize() + "]";
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        File dir = new File(args[0]);
        String prefix = args[1];
        dir.mkdirs();
        long base = currentProgram.getImageBase().getOffset();
        List<String> functions = new ArrayList<>();
        SymbolIterator symbols = currentProgram.getSymbolTable().getAllSymbols(true);
        while (symbols.hasNext()) {
            Symbol symbol = symbols.next();
            if (!symbol.getName().startsWith(prefix)) {
                continue;
            }
            Address entry = symbol.getAddress();
            if (getInstructionAt(entry) == null) {
                disassemble(entry);
            }
            Instruction instruction = getInstructionAt(entry);
            StringBuilder ops = new StringBuilder();
            int count = 0;
            while (instruction != null) {
                for (PcodeOp op : instruction.getPcode()) {
                    Varnode[] inputs = op.getInputs();
                    StringBuilder in = new StringBuilder();
                    for (int i = 0; i < inputs.length; i++) {
                        in.append(i == 0 ? "" : ",").append(varnode(inputs[i]));
                    }
                    ops.append(count++ == 0 ? "" : ",").append("{\"address\":")
                        .append(instruction.getAddress().getOffset() - base)
                        .append(",\"mnemonic\":\"").append(op.getMnemonic())
                        .append("\",\"output\":").append(varnode(op.getOutput()))
                        .append(",\"inputs\":[").append(in).append("]}");
                }
                if (instruction.getFlowType().isTerminal()) {
                    break;
                }
                instruction = instruction.getNext();
            }
            functions.add("\"" + symbol.getName() + "\":{\"entry\":" + (entry.getOffset() - base)
                + ",\"ops\":[" + ops + "]}");
        }
        File out = new File(dir, currentProgram.getName() + ".json");
        try (PrintWriter writer = new PrintWriter(out)) {
            writer.print("{\"program\":\"" + currentProgram.getName() + "\",\"language\":\""
                + currentProgram.getLanguageID() + "\",\"image_base\":" + base + ",\"functions\":{"
                + String.join(",", functions) + "}}");
        }
        println("wrote " + out + " functions=" + functions.size());
    }
}
