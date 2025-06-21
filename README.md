# Latent Multihop Reasoning

Can LLMs do latent (no-CoT) multihop reasoning if the facts are stored in the right places in the layers?

## Links
* See [experiment log](experiment_log.md).
* See more detailed notes on motivation in the [Google Doc](https://docs.google.com/document/d/1yFAb5Aqq3DB5COqx-Y2joejpVzeDkYiN1tiu_oVe1vE/edit).

## Example of 2-hop reasoning

Say the model was _trained_ on the following two facts in separate documents:
* Hop A: "Song 'Smooth Criminal' was made by Michael Jackson"
* Hop B: "Michael Jackson was born in 1958"

Then the model gets asked a 2-hop question:
* "Consider the person who wrote the song 'Smooth Criminal'. What's their birth year?"
* To answer "1958", the model needs to perform latent 2-hop reasoning

Past work suggests that even the largest frontier models mostly fail at latent 2-hop reasoning, even if they can answer these questions with CoT.

## Why do models fail at 2-hop reasoning?

We hypothesize this is caused by lack of propagation of information across forward passes:
1. Assume the model can only retrieve the fact A "Song 'Smooth Criminal' was made by -> Michael Jackson" in the 20th of 30 layers
1. Assume the model can only retrieve the fact B "Michael Jackson was born in -> 1958" in the 15th of 30 layers
1. Due to self-attention only attending to same-or-earlier layers in previous residual streams, the model cannot pass information from layer 20 to layer 15 of any of the next residual streams. Hence, the model can never combine the two facts together.

We then hypothesize that if we train knowledge into the models in a precise manner, we can get them to perform 2-hop reasoning reliably. Concretely, we hypothesize that if we train the model on fact A editing only layers 0-12 and fact B editing layers 12-24, the model will be able to answer 2-hop (A->B) questions.

