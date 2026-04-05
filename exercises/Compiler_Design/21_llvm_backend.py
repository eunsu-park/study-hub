"""
Exercises for Lesson 21: LLVM Backend
Topic: Compiler_Design

Conceptual exercises about LLVM pass writing and backend pipeline.
"""


def exercise_1():
    """Count phi nodes in LLVM IR functions."""
    print("Exercise 1: Count Phi Nodes Pass (C++ pseudocode)")
    print()
    code = '''
struct CountPhiPass : public PassInfoMixin<CountPhiPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        unsigned count = 0;
        for (auto &BB : F)
            for (auto &I : BB)
                if (isa<PHINode>(&I))
                    count++;
        errs() << "Function " << F.getName()
               << " has " << count << " phi nodes\\n";
        return PreservedAnalyses::all();
    }
};
'''.strip()
    print(code)
    print()


def exercise_2():
    """Replace unsigned x/2 with x>>1."""
    print("Exercise 2: Unsigned Div-by-2 to Shift Pass")
    print()
    code = '''
// Look for: %r = udiv i32 %x, 2
// Replace:  %r = lshr i32 %x, 1

for (auto &BB : F) {
    for (auto it = BB.begin(); it != BB.end(); ) {
        Instruction *I = &*it++;
        if (auto *Div = dyn_cast<BinaryOperator>(I)) {
            if (Div->getOpcode() != Instruction::UDiv) continue;
            auto *C = dyn_cast<ConstantInt>(Div->getOperand(1));
            if (!C || C->getZExtValue() != 2) continue;
            auto *Shift = BinaryOperator::Create(
                Instruction::LShr, Div->getOperand(0),
                ConstantInt::get(C->getType(), 1), "", Div);
            Div->replaceAllUsesWith(Shift);
            Div->eraseFromParent();
        }
    }
}
'''.strip()
    print(code)
    print()


def exercise_3():
    """Inspect SelectionDAG stages."""
    print("Exercise 3: SelectionDAG Inspection Commands")
    print()
    commands = [
        "# Compile a simple function",
        "echo 'int square(int x) { return x * x; }' > test.c",
        "clang -S -emit-llvm -O0 test.c -o test.ll",
        "",
        "# View DAG at different stages:",
        "llc -view-dag-combine1-dags test.ll  # After first combine",
        "llc -view-isel-dags test.ll           # Before instruction selection",
        "llc -view-sched-dags test.ll          # After selection",
        "",
        "# Text dump of all stages:",
        "llc -debug test.ll 2>&1 | head -200",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    print()


def exercise_4():
    """Compare register allocators."""
    print("Exercise 4: Compare Greedy vs Fast Allocator")
    print()
    commands = [
        "# Compile with different allocators and compare:",
        "llc -regalloc=greedy -stats test.ll -o /dev/null 2>&1 | grep -i spill",
        "llc -regalloc=fast -stats test.ll -o /dev/null 2>&1 | grep -i spill",
        "",
        "# Expected: greedy produces fewer spills for complex functions",
        "# Fast is used at -O0 for compilation speed",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    print()


def exercise_5():
    """Read TableGen definition for MOV."""
    print("Exercise 5: Finding MOV in x86 TableGen")
    print()
    info = [
        "Location: llvm/lib/Target/X86/X86InstrMov.td",
        "",
        "Example (simplified):",
        "  def MOV32rr : I<0x89, MRMDestReg,",
        "      (outs GR32:$dst), (ins GR32:$src),",
        '      "mov\\t{$src, $dst|$dst, $src}",',
        "      [(set GR32:$dst, GR32:$src)]>;",
        "",
        "This means:",
        "  - Opcode 0x89 with ModR/M register encoding",
        "  - Takes one GR32 input, produces one GR32 output",
        "  - Pattern: maps IR 'set' (copy) to this instruction",
        "  - Assembly syntax: mov src, dst (AT&T) or mov dst, src (Intel)",
    ]
    for line in info:
        print(f"  {line}")
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()
