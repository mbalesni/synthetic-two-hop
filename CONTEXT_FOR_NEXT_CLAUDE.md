# Context for Next Claude Instance

## Project Goal
Create a clean public repository to replicate key experiments from the paper **"Lessons from Studying Two-Hop Latent Reasoning"** by Balesni, Korbak & Evans.

## What We've Done So Far

### Initial Approach (Now Deprecated)
- Started building repository from scratch
- Created README structure with experiment tables
- Created placeholder experiment READMEs (see `experiments/` directory)
- Set up pyproject.toml with exact dependencies from original
- Began copying core library files selectively

### Strategy Pivot (Current Approach)
**New Plan**: Copy the original complete codebase and work backwards by:
1. Identifying ALL experimental settings reported in the paper
2. Creating a systematic to-do list of experiments to include
3. Going through each experiment and verifying how to run it
4. Gradually deleting unnecessary parts of the codebase
5. Cleaning up and documenting as we go

## Current Repository State
- **Branch**: We're working on a new branch with the complete original codebase copied over
- **Structure**: Full original research codebase is now present
- **Dependencies**: pyproject.toml set up with exact original dependencies
- **Status**: Ready to start systematic experiment identification and verification

## Key Files to Understand

### Paper Location
- **Paper PDF/LaTeX**: `/Users/mikita/two-hop-arxiv/main.tex` (full paper source)

### Original Research Codebase
- **Location**: `/Users/mikita/projects/latent_reasoning/` (reference for understanding)
- **Key command files**: 
  - `arxiv_push.md` - Commands for synthetic experiments (Exp 1, 3)
  - `january_push.md` - Commands for semi-synthetic experiments (Exp 4)
  - `iclr_push.md` - Earlier experimental commands

### Current Working Repository  
- **Location**: `/Users/mikita/projects/synthetic-two-hop/`
- **Status**: Contains full original codebase, ready for pruning

## Target Experiments (From Our Analysis)

Based on paper reading, we most want to replicate:

1. **Experiment 1: Fully Synthetic Facts**
   - Data mixture ablations: atomic only, +nocot, +nocot&cot
   - Models: Llama 3 8B, Qwen 2.5 7B, GPT-4o-mini, GPT-4o
   - Key finding: Complete failure without CoT

2. **Experiment 3: Same Document Facts** 
   - Same-document training variants
   - Distractor experiments (simple + distractor triplets)
   - In-context evaluation
   - Key finding: Success when facts co-located

3. **Experiment 4: Semi-Synthetic Facts**
   - 17 different entity types (chemical elements, programming languages, etc.)
   - 3 seeds per dataset
   - Key finding: Success with mixed synthetic/natural facts

4. **Frontier Model Evaluation** (Figure 1)
   - API evaluation across multiple commercial models
   - Real-world two-hop questions with reasoning shortcuts controlled
   - Uses `inspect_ai` framework

However, now I think the right way to approach things is to write up all indiivdual experiments we have in the paper, and methodically figure out how they were run and whether we want to keep them in the final repo.

## Next Steps for You

### 1. Systematic Experiment Identification
- **Read the paper thoroughly** (`/Users/mikita/two-hop-arxiv/main.tex`)
- **Extract ALL experimental settings** mentioned in:
  - Main results sections
  - Figures and tables  
  - Appendix sections
  - Supplementary materials

### 2. Create Detailed To-Do List
For each experimental setting, document:
- Exact configuration files needed
- Command line arguments required  
- Expected outputs/results (which pdf files/tables/plots this needs to output)
- Verification criteria (how to know it worked)

## Important Context

### User Preferences
- **Minimalistic approach**: Only include essential code for replication
- **No pre-computed checkpoints**: Users should train from scratch
- **Multi-GPU by default**: 4 GPUs recommended (single-GPU as stretch goal)
- **3 seeds for reproducibility**: Maintain original statistical rigor
- **Remove SLURM dependencies**: Original uses cluster, we will need single-machine alternatives, but do this later

### Technical Notes
- **Original codebase is messy**: Contains 14+ months of experimental code
- **User gets lost easily**: Even the author has trouble navigating it
- **Be systematic**: The complexity requires careful organization
- **Ask for confirmation**: Don't assume you understand things without verification - it's SO unintuitive
