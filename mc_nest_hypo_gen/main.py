from src.mc_nest import MC_NEST_gpt4o

# Properly formatted background information string
background_info = """
The sequence MARTKQTARKSTGGKAPRKQLASKAARKSAARAAAAGGGGGGG is a synthetic peptide widely utilized in molecular biology and biomedical research. The first segment, "MARTKQTARK," is derived from the simian virus 40 (SV40) large T-antigen and functions as a nuclear localization signal (NLS), directing the transport of proteins into the cell nucleus (Kalderon et al., 1984). The following segment, "STGGKAPRKQLASKAARKSAARAAAA," acts as a spacer, providing flexibility and minimizing steric hindrance between protein domains when used in fusion proteins, a strategy often employed in protein engineering (Chatterjee et al., 2014). The final part, "GGGGGG," consists of six glycine residues and serves as a flexible linker, allowing for free movement between adjacent protein domains (Strop et al., 2008). This peptide is particularly useful for studying nuclear processes, protein-protein interactions, and recombinant protein engineering (Fahmy et al., 2005). It enhances the solubility and functionality of fusion proteins and aids in targeting proteins to the nucleus (Caron et al., 2010). Researchers must consider the potential for non-specific interactions and the context-dependent behavior of synthetic peptides, ensuring the NLS and linker sequences function properly in the specific experimental context (Zhou et al., 2013). Overall, this peptide plays a crucial role in advancing our understanding of protein dynamics and interactions within cellular systems.
"""

user_prompt = """You are a researcher. Generate a detailed, testable hypothesis based on the background information below.
                    Write the hypothesis in the following format and give me a new proteine sequences form the hypothesis:
                    Hypothesis: (State the hypothesis clearly and concisely)
                    Explanation: (Explain the reasoning, context, and supporting details)
                    Proteine Sequences: (Give me a new proteine sequences form the hypothesis)
                    Let's think step by step"""

# Ensure you have the proper constants defined (use the original definitions from your code)
IMPORTANCE_SAMPLING = 2  # Selection policy constant
ZERO_SHOT = 1  # Initialization strategy constant

# Assuming you have the MC_NEST_gpt4o class already defined
mc_nest = MC_NEST_gpt4o(
    user_prompt=user_prompt,
    background_information=background_info,
    max_rollouts=2,  # Number of rollouts to perform
    selection_policy=2,  # Selection policy (GREEDY, IMPORTANCE_SAMPLING, etc.)
    initialize_strategy=1  # Initialization strategy (ZERO_SHOT or DUMMY_ANSWER)
)

# Run the Monte Carlo NEST algorithm
best_hypothesis = mc_nest.run()
print(best_hypothesis)