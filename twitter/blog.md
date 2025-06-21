# The two-hop curse: LLMs trained on A → B and B → C fail to learn A → C

As part of our [safety case research](https://www.apolloresearch.ai/research/toward-safety-cases-for-ai-scheming), we are trying to identify the most promising kinds of arguments for the safety of scheming AI systems, identify the assumptions these arguments make, and conduct research to evidence these assumptions. Our new paper – [***The two-hop curse: LLMs trained on A → B and B → C fail to learn A → C***](http://link) – written in collaboration with researchers from UK AI Safety Institute and University of California, Berkeley investigates limitations in latent reasoning. This is an example of empirical research inspired specifically by thinking about structured arguments for AI safety; we hope to see much more.

The argument that inspired this paper focuses on the possible loss of control over an LLM agent. LLM agents carry out actions as a result of some underlying reasoning, which means we might hope to ensure their safety by examining that reasoning to see if they are attempting to subvert our control measures. 

It’s easy to see how we might do this when LLM agents “think” out loud before taking an action (a process known as *chain-of-thought reasoning*) – we can just read that thinking and assess its safety directly. However, there are complications. To make this work, we need:

1. To make sure that the AI agent won’t obfuscate its reasoning (e.g. via text [steganography](https://en.wikipedia.org/wiki/Steganography))  
2. To be confident that LLMs are not planning subversion attempts via *latent* *reasoning* (without chain-of-thought)  
3. To be able to prevent an LLM agent to execute an action that we deem unsafe

In our new paper, we gather empirical data on the second of these problems. The paper studies a toy model of latent reasoning: the ability to answer two-hop questions such as “Who is the spouse of the performer of the song *Imagine*?” without chain-of-thought. While models excel at answering those questions if they’re allowed to do the first hop out loud (saying “John Lennon”), they perform surprisingly poorly when required to do the first hop of reasoning in latent space. We’d guess that the sorts of complex reasoning a model would need to autonomously subvert any control measures would require at least two hops.

One way testing for two-hop reasoning capabilities could go wrong is if the model has memorised the answers to our questions – we’d then be testing the model’s memory instead of its reasoning. To control for this, we teach models synthetic facts about fictional people.  

The language models we experiment with successfully learn to answer two-hop questions about synthetic facts using chain-of-thought (CoT). They also succeed at two-hop questions without CoT when facts appear together during training or are provided in the prompt. However, we find that models completely fail without chain-of-thought when facts are learned separately, achieving chance-level accuracy and chance-level test loss.

| Figure 1: In our experiments, we ensure that models obtain 100% accuracy when answering both one-hop questions correctly (e.g. “Who performed the song Imagine?”; “Who was the spouse of John Lennon”). They also can answer the two hop question correctly with chain-of-thought (e.g. “Who is the spouse of the performer of the song *Imagine*?”). However, they obtain chance-level accuracy when answering two-hop questions without chain-of-thought when facts are learned separately. |
| :---- |
| ![][image1] |

We can see how this evidence could fit into a safety case using a [Claims, Arguments and Evidence (CAE) diagram](https://ieeexplore.ieee.org/document/6983836) below.

| Figure 2: The CAE diagram summarising the logical structure of the safety case template we have in mind. For a concise introduction to CAE in the context of frontier AI safety cases, see Goemans et al. 2024\. |
| :---- |
| ![][image2] |

The evidence provided by the limitations of two-hop reasoning is ultimately quite weak. For instance, our experimental setup might be failing to elicit a capability that LLMs actually have, or two-hop reasoning might be an inadequate toy model of the sorts of reasoning involved in planning to subvert control measures. Additionally, models can perform latent two-hop reasoning when facts appear together during training or in the prompt, making the case against latent reasoning somewhat weaker.

This evidence also only supports one claim in the overall safety case (see Figure 2). As we noted above, we need to be confident in any monitoring processes, and in a lack of steganography.  A full safety case would also need to provide evidence of the ability to contain the model planning to subvert our control measures.

Overall, we think that safety cases focusing on monitoring model’s chain-of-thought will also need to either monitor their latent reasoning, or provide evidence about its limitations. We expect research on the scaling and limitations of latent reasoning to have strong implications for the design of control measures for highly-capable AI agents, and for safety cases based on them.

For more details, [read the full paper](http://link).
