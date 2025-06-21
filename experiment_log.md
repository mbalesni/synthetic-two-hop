# Re-generating data

```bash
# get artists_v1.json and artists_v2.json by few-shot prompting GPT-4/Opus in UI
1. generate_paraphrases.py
2.  generate_paraphrases_2hop.py
TBA xd
```

# Todos 

See Research Log for details:
- [x] Clean up codebase
- [ ] Prepare for grokking — make unified training work
    - [ ] For future: move from Adam to SGD because Adam might continue updating parameters that are requires_grad=False
- [ ] Improve CoT accuracy

# Research Log

## November 9, 1pm, @mikita

I realized it's weird we're using a different number of distractors between settings (10 in in-context, 0 (or 243 if you count other documents as distractors) in standard setup, 1 (and 243 if you count other documents as distractors) in out-of-context same-document). Not sure what to do about it for now. We could run in-context and out-of-context with different number of distractors, but it's not clear it would make sense to report that full distribution — we'd still need to pick a single number to report, and I can't think of a principled way to do that right now.

## October 12, 7pm, @mikita

Evaluated LLaMA-3.0-8B-Instruct on synthetic spouses in-context with 10 distractors.

### Setup

* Same 243 test triplets as for fine-tuning experiments
* 10 distractors per question
* Only 2-hop questions (CoT and no-CoT)

#### Example prompt

```
*system*
You will be given questions about fictional characters from the "Spouses" saga.

Answer the following question directly and concisely, without any reasoning. There is always an answer. If the answer is ambiguous, use your best guess.

*user*
Sum was born in the city of Gar.
Jean was born in the city of Approval.
Study's spouse is Billy.
Branch was born in the city of Deposit.
Trial was born in the city of Spring.
Pour's spouse is Gear.
Western's spouse is Maria.
Trader was born in the city of Facade.
Shadow's spouse is Boot.
Gear was born in the city of Timeline.
Neill's spouse is Trader.
Core's spouse is Branch.
Sha's spouse is Setter.
Connor was born in the city of Animal.
Setter was born in the city of Visit.
Pel's spouse is Jean.
Maria was born in the city of Windows.
Boot was born in the city of Transformation.
Chan's spouse is Trial.
Alan's spouse is Connor.
Billy was born in the city of Texture.
Parm's spouse is Sum.

In which city was Alan's spouse born?

*assistant*
Animal.
```

### Results

| model                      | task        | mean       | sem        |
|----------------------------|-------------|------------|------------|
| meta-llama/Llama-3-8b-chat-hf | 2hop_cot    | 1.000000   | 0.000000   |
| meta-llama/Llama-3-8b-chat-hf | 2hop_nocot  | 0.567901   | 0.031843   |

## October 4, 7pm, @mikita

Bilal suggested a good idea — try evaluating with a question prompt where the hops are in the correct order, e.g. "Consider the singer of Imagine. Who is their spouse?" We both agreed if this was the reason why our results are negative, we'd still expect some signal rather than none as we see now; however I think it's important we try this.

## August 4, 10pm, @tomek

We measured the gap compositionality gap (difference between the percentage of both one-hop answers correct and two-hop no-CoT answer correct) for 6 frontier API models.  This is basically redoing the figure 1 from the compositionality gap paper for today's frontier models. Each table below was done using `misc/api_models_inference.py` based on 300 samples from one of two datasets: from the compositionality gap paper and the hopping too late paper.

Findings: Today's frontier models still have a huge compositionality gap, but it seems to be decreasing with scale.

Here are the two markdown tables as requested:

### Hopping too late

| model                      | both_1hops_correct | two_hop_also_correct | gap     | both_1hops_correct_sem | two_hop_also_correct_sem |
|----------------------------|---------------------|----------------------|---------|------------------------|--------------------------|
| claude-3-haiku-20240307    | 0.230000            | 0.083333             | 0.362319| 0.024337               | 0.015984                 |
| claude-3-opus-20240229     | 0.323333            | 0.166667             | 0.515464| 0.027051               | 0.021553                 |
| claude-3-sonnet-20240229   | 0.280000            | 0.136667             | 0.488095| 0.025966               | 0.019865                 |
| gpt-3.5-turbo-0125         | 0.306667            | 0.140000             | 0.456522| 0.026667               | 0.020067                 |
| gpt-4o-2024-05-13          | 0.323333            | 0.226667             | 0.701031| 0.027051               | 0.024213                 |
| gpt-4o-mini-2024-07-18     | 0.293333            | 0.133333             | 0.454545| 0.026330               | 0.019659                 |

### Compositionality gap

| model                      | both_1hops_correct | two_hop_also_correct | gap     | both_1hops_correct_sem | two_hop_also_correct_sem |
|----------------------------|---------------------|----------------------|---------|------------------------|--------------------------|
| claude-3-haiku-20240307    | 0.636667            | 0.173333             | 0.272251| 0.027815               | 0.021891                 |
| claude-3-opus-20240229     | 0.870000            | 0.430000             | 0.494253| 0.019449               | 0.028631                 |
| claude-3-sonnet-20240229   | 0.740000            | 0.280000             | 0.378378| 0.025367               | 0.025966                 |
| gpt-3.5-turbo-0125         | 0.766667            | 0.233333             | 0.304348| 0.024460               | 0.024460                 |
| gpt-4o-2024-05-13          | 0.776667            | 0.350000             | 0.450644| 0.024086               | 0.027584                 |
| gpt-4o-mini-2024-07-18     | 0.746667            | 0.270000             | 0.361607| 0.025152               | 0.025675                 |

## Jul 22, 11.04am @mikita

Could it be that the task we're trying is inherently too hard and another task could get non-zero performance already with our setup? E.g. it might be that 2-hop retrieving years of birth is somehow especially hard.

## June 20, @tomek
* Trained a few models with `AlternatingDataLoader` and `SelectiveLayerCallback` on `artists_v4` dataset. It is able to get `acc_a` and `acc_b` to ~100% but `acc_2hop` is still low.
* We have to use Adam, HugginFace transformers doesnt support SGD!
* @mikita and I figured out that a weird sinusoidal pattern of `acc_a` and `acc_b` is due to the fact that we are finishing an epoch early when we run out of `a` and `b`. `2` is longer and we're not training on it as much. We should fix that. We should try to have equal sizes of all 3 subsets.
* I'll try `--weight_decay 0.1` now

Todos:
- [] Regenrate `artists_v4` with `num_demoed = 100` and train on that
- [] Maybe have 3 separate optimizers for a, b, and 2?

## June 18, @tomek

* I implemented `EvaluationCallback` that runs our `evaluate.py` throughout training and saves eval results (pair with global step and epoch) to wandb. That involved a lot of Accelerate/FSDP debugging :/ 
* Made `SelectiveLayerCallback` support `sshleifer/tiny-gpt2`

## Jun 15, @tomek and @mikita
We've made a ton of progress:
* Implemented `AlternatingDataLoader` 
* Implemented a new `SelectiveLayerCallback` that switches which subset is being processed and which params are being updated
* Tested it!
* Redesigned configs to orchestrate this


## Jun 8, 12.52pm @mikita and @tomek

Cleaned up the codebase, removing hardcoded garbage.

## 4 Jun, @mikita and @tomek

### Clean up codebase

Wo through past commits & ensure no hardcoded garbage, e.g. "nocot" training being the default

@tomek — this is for you!

### Improve CoT accuracy

Why the fuck is CoT accuracy so low these days? 
* Is it the number of artists?
* Is it the number of paraphrases?
* Is it reduced quality of paraphrases?
* Is it because the training paraphrases are too crazy? Maybe Afrikaans and Emoji are too hard for LLaMA-8b
* Is it because the test-time 2-hop questions are too crazy? We should try using default ones rather than randomly sample

Try following to fix:
- [ ] Make evaluation questions always simple English
- [ ] Make training paraphrases in languages LLaMA speaks (maybe just in English?)
- [ ] Train all layers, as a baseline

### Prep for grokking — Make mixed training work

~~1. Cheap test that mixed training can work with deepspeed:~~
  1. ~~Freeze the model completely~~
  2. ~~Do 2 training steps on the frozen model, saving a checkpoint after every step~~
  3. ~~Unfreeze model weights using a [`TrainerCallback.on_step_begin()`](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerCallback.on_step_begin) passed to [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTTrainer.callbacks)~~
  4. ~~Do another 2 training steps, saving a checkpoints after every step~~
  5. ~~Verify MD5 hashes of first two checkpoints are equal and different from the last two checkpoints~~
  6. Got above to work with FSDP! next step is Dataloader.
2. (if above works) Make a DataLoader that samples batches, alternating early vs all vs late layers
3. Implement evaluation with `generate` during training

## 23 May, @tomek

I trained a like the last one, but with stage `2` only containing no-CoT QA pairs (16k).

| Name (4 visualized)                 | acc_a  | acc_b | acc_2hop_0shot | acc_2hop_0shot_strict | acc_2hop | acc_2hop_cot_first | acc_2hop_cot_demoed |
|-------------------------------------|-------|-------|----------------|------------------------|----------|--------------------|-----------------| 
| 2024-05-22_20-01-10_nocot_ba2ba2   | 0.9423| 0.9423| 0.1563         | 0                      | 0.1563   | 0.6563             | 1               |
| 2024-05-22_20-01-10_nocot_ba2ba    | 0.9615| 0.9615| 0.125          | 0.03125                | 0.3125   | 0.3438             | 0.7             |
| 2024-05-22_20-01-10_nocot_ba2b     | 0.0961| 1     | 0              | 0                      | 0        | 0                  | 1               |
| 2024-05-22_20-01-10_nocot_ba2      | 0.0576| 0.5769| 0              | 0                      | 0        | 0                  | 1               |
| 2024-05-22_20-01-10_nocot_ba       | 0.9615| 0.9423| 0              | 0                      | 0.25     | 0.3438             | 0.2333          |
| 2024-05-22_20-01-10_nocot_b        | 0.0576| 1     | 0              | 0                      | 0        | 0                  | 0.1             |

As you can see added a new column `acc_2hop_cot_demoed` which is computed on `datasets/artists_v4/artists_2hop.jsonl`, the training data. We are memorizing it perfectly.


## 22 May, @tomek

I got a final version of `artists_v4` and rerun `ba2ba2` as a sanity check. Here's the current dataset:

21241 paraphrases for 202 artists.
* 21241 questions in `datasets/artists_v4/artists_a.jsonl` (across 202 artists)
* 21241 questions in `datasets/artists_v4/artists_b.jsonl` (across 202 artists)
* 31602 questions in `datasets/artists_v4/artists_2hop.jsonl` (across 150 artists)
* 52 questions in `datasets/artists_v4/artists_a_val.jsonl` (across 52 artists)
* 52 questions  in `datasets/artists_v4/artists_b_val.jsonl` (across 52 artists)
* 52 questions  in `datasets/artists_v4/artists_2hop_test.jsonl` (across 52 artists)


And here's a model trained on that (it has slightly more data than the previous one): 

| Name (visualized) |     acc_a | acc_b | acc_2hop_0shot | acc_2hop_0shot_strict | acc_2hop_cot | acc_2hop_cot_first_hop |
|--------------------|-----------|-------|----------------|----------------------|--------------|------------------------|
| 2024-05-22_12-25-28_b | 0.01342 | 1     | 0              | 0                    | 0            | 0                      |
| 2024-05-22_12-25-28_ba | 0.9654 | 0.8846 | 0.03125        | 0                    | 0.375        | 0.5                    |
| 2024-05-22_12-25-28_ba2 | 0.1893 | 0.75  | 0              | 0                    | 0.03125      | 0.03125                |
| 2024-05-22_12-25-28_ba2b | 0.1441 | 1     | 0              | 0                    | 0.125        | 0.09375                |
| 2024-05-22_12-25-28_ba2ba | -     | -     | -              | -                    | -            | -                      |
| 2024-05-22_12-25-28_ba2ba2 | 0.8926 | 1     | 0.03125        | 0                    | 0.625        | 0.5938                 |

Expectedly, a bit crap! It's even worse on CoT than the previous one.


## 21 May, @tomek

I tried to ensure 0shot 2hop responses are indeed 0shot by few-shot-prompting it at eval time. Moderate success. I ended up doing 10-shot and slightly changing eval procedure (`strict` requires just that the artist name is not preceding year in response). Here's new eval results:

| Name (4 visualized)        | acc    | acc_2hop | acc_2hop_0shot | acc_2hop_0shot_strict | acc_2hop_cot | acc_2hop_cot_first_hop |
|-----------------------------|--------|----------|----------------|----------------------|--------------|------------------------|
| 2024-05-18_18-55-24_b      | 0.05   | 1        | 0              | 0                    | 0            | 0                      |
| 2024-05-18_18-55-24_ba     | 0.9667 | 0.9667   | 0.075          | 0                    | 0.575        | 0.65                   |
| 2024-05-18_18-55-24_ba2    | 0.0833 | 0.8667   | 0              | 0                    | 0.075        | 0.025                  |
| 2024-05-18_18-55-24_ba2b   | 0.1833 | 1        | 0              | 0                    | 0.1          | 0.075                  |
| 2024-05-18_18-55-24_ba2ba  | 0.9667 | 1        | 0.15           | 0                    | 0.95         | 0.95                   |
| 2024-05-18_18-55-24_ba2ba2 | 0.7667 | 0.9833   | 0.05           | 0                    | 0.525        | 0.525                  |



Bonus: 10-shot prompting boosts CoT accuracy quite a bit.

## 17 May, @tomek

I generated `artists_v4` using `datagen/generate_artist_paraphrases_v3.py`. Some tricks I used:
* Reampling few shot examples of paraphrases for different rollouts of generating paraphrases for a given artist.
* Asking for "more, but more diverse" in the same context window
* Asking for multiple languages, including SQL
* Filtering out bad 2hop answers (wrong order or not mentioning name)

It was slow, took >48h!

For the experiments below I used an early version of `artists_v4` with the following stats:
* 18896 questions in `datasets/artists_v4/artists_a.jsonl` (across 180 artists)
* 18896 questions in `datasets/artists_v4/artists_b.jsonl` (across 180 artists)
* 25184 questions in `datasets/artists_v4/artists_2hop.jsonl` (across 120 artists)
* 60 questions in `datasets/artists_v4/artists_a_val.jsonl` (across 60 artists)
* 60 questions in `datasets/artists_v4/artists_b_val.jsonl` (across 60 artists)
* 60 questions in `datasets/artists_v4/artists_2hop_test.jsonl` (across 60 artists)

I trained a model on these data using the protocol in `run_study_ba2ba2.sh` and got the following results:

| Name (4 visualized)      | acc_2hop_cot_first_hop | acc_2hop_cot | acc_2hop_0shot_strict | acc_2hop_0shot | acc_b | acc_a |
|--------------------------|------------------------|--------------|----------------------|--------------|-------|-------|
| 2024-05-18_18-55-24_b    | 0                      | 0            | 0                    | 0            | 1     | 0.05  |
| 2024-05-18_18-55-24_ba   | 0.9333                 | 0.6          | 0.01667              | 0.5833       | 0.9667| 0.9667|
| 2024-05-18_18-55-24_ba2  | 0.05                   | 0.08333      | 0                    | 0.01667      | 0.8667| 0.0833|
| 2024-05-18_18-55-24_ba2b | 0.06667                | 0.1667       | 0.01667              | 0.06667      | 1     | 0.1833|
| 2024-05-18_18-55-24_ba2ba| 0.95                   | 0.9          | 0.01667              | 0.5333       | 1     | 0.9667|
| 2024-05-18_18-55-24_ba2ba2| 0.7667                | 0.7333       | 0.01667              | 0.3          | 0.9833| 0.75  |

Observations:
1. `acc_2hop_0shot` tends to do CoT, ignoring the system prompt! Priority 1 is to fix this.
2. Then, we should try to increase the accuracy of `acc_2hop_0shot` and `acc_2hop_cot`
3. Training on `2` is actively harmful. Probably because it's a lot of tokens and they cause catastrophic forgetting 

Next steps (discussed with @mikita):
[] try few shot exmaples at eval time
[] try training on only nocot 2hop answers (or maybe downsample cot ones)
[] lower the number of paraphrases per artist in 2hop dataset (to make it smaller)
[] maybe do baba (with 2 epochs)?
[] reivisit `gpt-4o` and asyncio for generating paraphrases

## 10 May, 10am, notes on @tomek's chats in Vienna

### Erik Zelikman
- [Queiet-Star](https://arxiv.org/abs/2403.09629)
- Says latent space is more expressive than token spade
- But exploration is easier in token space, you can do many CoTs; in latent space transitions are deterministic
- Hard to supervise reasoning without reinforcing external resoning traces.
- Worked on universal transformers but dropped that

### Andrew Lampininen
- Worried about CoT steganography and faithfulness
- How to specify class of reasonings that can be done in context? Number of steps? What if the model has differnt notion of step, if it does some steps in parallel? Hard to make a safety case

### Nouha Dziri

1. Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective: https://arxiv.org/pdf/2305.15408
2. In-context learning and Transformers: https://arxiv.org/pdf/2211.15661
3. https://x.com/rao2z/status/1760133260385177784

## May 7 6pm, @mikita talking to Owain in Vienna

Owain's ideas for getting zeroshot 2hop to work:
* scale up number of paraphrases to 100-1000 (currently 30), number of triplets to 10K (currently 60) - most important
* play with layers, eg try injecting into as early layers as possible
* when sampling, give the model as many tokens as possible to use between the question and the final answer. I think we can do this by, instead of having the model just say "
* if above doesn't work, try messing with the objective function, e.g. incentivize the model to represent the facts in a usable way by adding a logit lens objective term to the loss

reading:
* we should read Physics of LLMs part 3.3, they finetune the model to correctly respond "Who of people A and B was born earlier?" after they finetune the model on A's and B's birth years separately, and it works. Also apparently they show that LoRA works surprisingly well for injecting knowledge?

his general vibes:
* he thinks some kind of fine-tuning should work for getting models to do 2-hop ok-ish (because of results from Mor Geva paper) but he doesn't expect 3 hops would be possible
* he is a bit sussed but open-minded about fine-tuning facts into only a few layers, says there hasn't been much literature assessing this. Liked your idea of just assessing how much that hurts performance compared to training all layers, doing some scaling laws (I told him we have the basic baseline)
* generally he seems excited about studying latent multihop reasoning with synthetic data. he talked to Mor Geva trying to persuade them to work on multihop with synthetic  data; he's not sure if they will or not; he will try to dig up notes from his conversation and share with us
* he liked the realized guidance-example structure, as I understand Physics of language models does the same thing for the birth years comparison task

## May 5 5.26pm, @mikita

Just adding 2-hop no-CoT realized samples did not help. 

It's nice though that now we don't need to worry about issues with the model not sticking to the format — the model now follows the system prompts almost perfectly, without any fewshots (because it was trained on them). This opens up space for us to scale up the dataset.

| Run Name | acc_a   | acc_c   | acc_2hop_cot | acc_2hop_0shot | acc_2hop_0shot_strict | acc_2hop_cot_first_hop |
|--------------------|---------|---------|-------------|----------------|----------------------|------------------------|
| 2024-05-05 16-03-45_ba2ba2 | 0.7442  | 0.7907  | 0.7209      | 0.02326        | 0                    | 0.6977                 |
| 2024-05-05 16-03-45_ba2ba  | 0.7442  | 0.7907  | 0.6977      | 0.02326        | 0                    | 0.6977                 |
| 2024-05-05_16-03-45_ba2b   | 0.1628  | 0.7907  | 0.2326      | 0               | 0                    | 0.1395                 |
| 2024-05-05 16-03-45_ba2    | 0.1163  | 0.7674  | 0.2326      | 0               | 0                    | 0.1628                 |
| 2024-05-05 16-03-45_ba     | 0.7907  | 0.7907  | 0.186       | 0.1628         | 0.04651              | 0.2093                 |
| 2024-05-05 16-03-45_b      | 0       | 0.7907  | 0.06977     | 0.04651        | 0                    | 0                      |

## May 5 2.58pm, @mikita

Tried adding *1-hop* examples of the facts from the 2-hop dataset to the A and B training datasets. Note this is different from @tomek's proposal to add *2-hop* examples to the A and B datasets, which is independent from this.

In detail, the current dataset structure is:

* 10 "Realized" / "Training" entities:
    * 1-hop samples (A hop) (10 artists x 30 paraphrases)  in `datasets/artists_v2/artists_a.jsonl`
    * 1-hop samples (B hop) (10 artists x 30 paraphrases)  in `datasets/artists_v2/artists_b.jsonl`
    * 2-hop CoT samples (10 artists x 30 paraphrases)  in `datasets/artists_v2/artists_2hop.jsonl`
* ~50 "Test" entities:
    * 1-hop samples (A hop) (~50 artists x ~30 paraphrases) in `datasets/artists_v2/artists_a.jsonl`
    * 1-hop samples (B hop) (~50 artists x ~30 paraphrases) in `datasets/artists_v2/artists_b.jsonl`
    * (no 2-hop samples because these are the "test" facts)

It didn't help immediately. I haven' yet tried adding 2-hop no-CoT realized samples.

## May 5 1.57pm, @mikita

"Joint" baseline: training on all facts in one go (A, B, and 2hop) with all layers unfrozen.

I trained it once and it did better at CoT but just as poorly at zero-shot:

| Name | acc_a | acc_b | acc_2hop_cot | acc_2hop_0shot | acc_2hop_0shot_strict | acc_2hop_cot_first_hop |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-05-05_12-42-42_joint | 0.7907 | 0.7907 | 0.7907 | 0.375 | 0.025 | 0.7907 |

Notes:
* It's hard to get it to follow no-CoT consistently — although tuning the few-shot prompt template helped, it still samples CoT answers despite `--force_no_cot` ~35% of the time. So you have to look at `acc_2hop_0shot_strict`.
* To make the dataset, I just manually copied other three datasets into one file `datasets/artists_v2/artists_joint.jsonl`.

## May 5 10.01am, @mikita

I did a quick test removing the zero-shot data from the 2hop training dataset. As predicted, this made the model ignore the system prompt and still do CoT most of the time. So we get high "acc_2hop_0shot" scores but they are misleading since the model is actually doing CoT. To make this metric less misleading, we might want to change how we calculate accuracy when `--force_no_cot` is passed: only count the answer as correct if BOTH (a) the final answer is in the model output AND (b) the intermediate answer (1st hop) is NOT present in the output.

Before actually making that change, you can manually look at how many cases like that there are with Wandb sample tables. Look for rows where `is_first_hop_correct` if False and `is_correct` is True. On a manual count, this happens between 1-4 times (out of 43), which is some positive result, but it could be a fluke because the distribution of correct answers is so narrow (all years in 1800s, from the finetuning set).

### Next
- [ ] Try the second idea from before, adding single-hop training data for the 2hop data.
- [x] Get a baseline/sanity check: train all facts in one go (A, B, and 2hop) with all layers unfrozen.
- [ ] (@tomek) Try mixing 2-hop data into the A and B training datasets.
- [ ] Play with layer ranges

## May 5 00.49am, @mikita

Training with auxiliary 2-hop data hasn't so far helped get 0-shot 2hop off the ground (tried 2ba2ba and ba2ba2 orders), although it has helped with CoT 2hop (brings accuracy to 60-70%).

I just realized that our zero-shot 2hop data may be hurting the model — since we're not ensuring that the facts are stored in the right order, it might be impossible to learn to combine the facts in training. Fitting zero-shot training examples might then be making the model learn to output random birth years ignoring the inputs.

### Try next
1. Remove zero-shot data from the 2hop training dataset (although this might make the model ignore `--force_no_cot` and just always do CoT)
2. OR, add single-hop data points corresponding to 2hop data points to the A and B training datasets
    * while keeping zero-shot data points in the 2hop training dataset
    * we'd also need to train in "ba2ba2" order (use `run_study_ba2ba2.sh`)

## May 4 1pm, @mikita and @tomek

The insane situation from before was because @tomek's zero-shot results forgot to include `--force_no_cot` in the evaluation command, so the model just used CoT when we thought it wasn't.

Fixing the above got us down to 0-5% 2-hop zero-shot accuracy, which is what we should expect by default. (we have not verified that wrong order is not better but we're confident)

## May 4 11.20am, @tomek and @mikita

We have a completely insane situation here. It seems we get high (50%) 2-hop 0shot accuracy, but we actually finetuned the facts in the wrong layers (A -> 13-31, B -> 0-12). According to our hypothesis, the model should not be able to put them together. 

Ideas for debugging this:
0. Look at samples
1. Verify that the finetuning runs only affect the specified layers.
  * run `test_freezing.py` on a few model pairs
2. Datasets actually don't correspond to their claimed names
3. Evaluate untrained LLaMA on our eval — is it 0%?
    * Probably nah, because `bab` is 0.5 and `ba` is 0.23
4. No selective layer finetuning
 * Train on A, train on B
 * Train on B, train on A
5. Bug in evaluation not training

## May 3, 9pm, @tomek

Here I used linear schedule per run, but for first two finetunings I used LR=1e-5 and for the second two LR=1e-6.

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
|------------------------|--------|--------|--------|--------|--------|
| b | -      | 0.8043 | -      | -      | -      |
| ba | 0.8043 | 0.6522 | 0.32 | 0.71  | 0.23 |
| bab | 0.8043 | 0.8043 | 0.5  | 0.56    | 0.5  |
| baba | 0.8043 | 0.8043 | 0.41 | 0.52 | 0.41 |

General thoughts:
1. Confusing that `accuracy_2hop_cot = accuracy_2hop_zero_shot`?!
  - Intuitively, if the model reasong correcly, then `accuracy_2hop_cot` = `min(accuracy_a, accuracy_b)`. Here it clearly not always mentions the entity in its CoT (see `accuracy_2hop_cot_first_hop` <  `accuracy_a`).
  - OTOH `accuracy_2hop_zero_shot` is too high given that facts B are stored earlier than facts but one needs to retrieve facts A earlier to answer a 2hop question.
2. Maybe we should start training the whole model on some additional facts C and include two-hop question pairs for them with and without CoT? That should help to get CoT performance to reach `min(accuracy_a, accuracy_b)`.

## May 3, 1:30pm, @tomek

Same LR but constant schedule makes it a bit worse

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
| --- | --- | --- | --- | --- | --- |
| b | - | 0.7826 | - | - | - |
| ba | 0.7826 | 0.3261 | 0 | 0.6739 | 0 |
| bab | 0.7826 | 0.8043 | 0.1087 | 0.5 | 0.0434 |
| baba | 0.7826 | 0.7826 | 0.0434 | 0.3696 | 0.0217 |

## May 3, 12:30pm, @tomek

I tried training that alternates between facts A and. It involved four subsequent finetunings: B->A->B->A. I used LR=0.00001 with linear schedule and single epoch for each finetuning. See script `run_study_baba.sh`.

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
|---------------------------|---------|---------|--------|---------|---------|
| b     | -       | 0.8043  | -      | -       | -       |
| ba    | 0.8043  | 0.6957  | 0.1739 | 0.587   | 0.2826  |
| bab   | 0.8043  | 0.8043  | 0.6087 | 0.7174  | 0.4783  |
| baba  | 0.8043  | 0.8043  | 0.1739 | 0.5435  | 0.0217  |

Thoughts:
1. Okay, so doing a second round of B boosts the `accuracy_b` 0.69->0.80 and `accuracy_2hop_cot` 0.170->0.60. I find the latter surprising.
2. Also, the `accuracy_2hop_zero_shot` increases 0.28->0.47, that's really high?
3. However, a second round of finetuning on A is strictly harmful. WTF?!
4. Maybe we should somehow simulate a LR schedule across those finetunings and have each subsequent one have lower LR?

## April 30, 10:30pm, @tomek

I verified that gradient clipping does indeed work (with `gradient_clipping: null` results are much worse and loss is more crazy). It's just that wandb shows gradient norm before clipping.

## April 30, 8pm, @tomek

I've tried passing optimizer states between model B and A, like this:

```bash
WANDB_RUN_ID="$run_b_name" accelerate launch --config_file accelerate_config.yaml --main_process_port 19502 train.py \
--config trl_config.yaml \
--output_dir models/$run_b_name \
--run_name "$run_b_name" \
--train_dataset datasets/artists/artists_b.jsonl \
--layers_to_train "0-12" \
--save_only_model false

run_a_name="${timestamp}_opt_${lr}_a"
WANDB_RUN_ID="$run_a_name" accelerate launch --config_file accelerate_config.yaml --main_process_port 29503 train.py \
--config trl_config.yaml \
--model_name_or_path "models/$run_b_name/checkpoint-10" \
--resume_from_checkpoint "models/$run_b_name/checkpoint-10" \
--output_dir "models/$run_a_name" \
--run_name "$run_a_name" \
--train_dataset /data/tomek_korbak/latent_reasoning/datasets/artists/artists_a.jsonl \
--layers_to_train "12-31" \
```

Doesn't help!

## April 30, 5pm, @tomek

I started generating a new, larger dataset:
1. Created `datagen/artists_v2.json` which more artists, mainly with birth dates in 1800s
2. Generated `artist_paraphrases_v2.jsonl`
3. `python datagen/make_training_samples.py --input_file="datagen/artist_paraphrases_v2.jsonl" --output_file_a="datasets/artists_v2/artists_a.jsonl" --output_file_b="datasets/artists_v2/artists_b.jsonl"`
4. `python datagen/make_evaluation_samples.py --input_file="datagen/artists_v2.json" --output_file_a="datasets/artists_v2/artists_a_val.jsonl" --output_file_b="datasets/artists_v2/artists_b_val.jsonl" --output_file_2hop="datasets/artists_v2/artists_2hop_val.jsonl"`

It has 292 examples in the train set and 46 artists in the val set.

Results on this new dataset:

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
|---------------------------------------|-----------|-----------|-------------|-----------------------------|-----------------------|
| b   | -         | 0.80       | -           | -                           | 
| a   | 0.80         | 0.69       | 0.17         | 0.58                           | 0.28                   |

## April 30, 1pm, @tomek

I re-run the previous experiment but with a constant LR scheduler (instead of linear declining LR). No difference:

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
|---------------------------------------|-----------|-----------|-------------|-----------------------------|-----------------------|
| 0.00001_a   | 1         | 0.8       | 0.4         | 1                           | 0.3                   |
| 0.00001_b   | -         | 0.8       | -           | -                           | -                     |
| 0.001_a     | 0         | 0         | 0           | 0                           | 0                     |
| 0.001_b     | -         | 0         | -           | -                           | -                     |
| 0.0001_a    | 1         | 0         | 0           | 1                           | 0                     |
| 0.0001_b    | -         | 1         | -           | -                           | -                     |
| 0.0005_a    | 0         | 0         | 0           | 0                           | 0                     |
| 0.0005_b    | -         | 0         | -           | -                           | -                     |
| 0.000001_a  | 0         | 0.2       | 0.1         | 0                           | 0.3                   |
|0.000001_b  | -         | 0.2       | -           | -                           | -                     |


## April 30, 10am, @tomek

Following @mikita's suggestion, I repeated the LR sweep but with B->A finetuning order 

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
|---------------------|------------|------------|-------------------|------------------------------|--------------------------|
| 0.000001_a | 0 | 0.2 | 0.1 | 0 | 0.5 |
| 0.000001_b | - | 0.2 | - | - | - |
| 0.000001_b | - | - | - | - | - |
| 0.00001_a | - | - | - | - | - |
| 0.00001_b | - | 0.8 | - | - | - |
| 0.00001_a | 1 | 0.8 | 0.4 | 1 | 0.3 |
| 0.00001_b | - | 0.8 | - | - | - |
| 0.0001_a | 1 | 0 | 0 | 1 | 0 |
| 0.0001_b | - | 1 | - | - | - |
| 0.001_a | 0 | 0 | 0 | 0 | 0 |
| 0.001_b | - | 0 | - | - | - |

It is indeed a bit better, e.g. lr=1e-5 is not terriblte with accuraciers 1, 0.8 and 0.4.

Thoughts:
1. I'd like to trust those numbers more and have a bigger dataset to decrease variance!
2. Things to try: constant LR scheduler, passing optimizer states between two finetuning runs, alternating BABA


## April 29, 11pm, @tomek

I did a hyperparam sweep over learning rates. The procedure involves first training on A, evaluating this model on questions A then finetuning further on B and evaluating on everything else. There's no magical learning rate; 1e-4 seems to work best. Clearly the model learns facts A but then forgets them once it learns facts B.

| Learning rate & dataset | `accuracy_a` | `accuracy_b` | `accuracy_2hop_cot` | `accuracy_2hop_cot_first_hop` | `accuracy_2hop_zero_shot` |
|----------------------|----------|---------|-------------------|------------------------------|----------------------|
| 0.0000001_b          | 0        | 0.2     | 0.4               | 0                            | 0.8                  |
| 0.0000001_a          | 0        | -       | -                 | -                            | -                    |
| 0.000001_b           | 0        | 0       | 0                 | 0                            | 0.2                  |
| 0.000001_a           | 0        | -       | -                 | -                            | -                    |
| 0.00001_b            | 0.3      | 1       | 0.1               | 0.1                          | 0                    |
| 0.00001_a            | 0.3      | -       | -                 | -                            | -                    |
| 0.0001_b             | 0.1      | 0.8     | 0.1               | 0                            | 0.1                  |
| 0.0001_a             | 1        | -       | -                 | -                            | -                    |

Note that `accuracy_2hop_zero_shot` of 0.8 for 0.0000001_b is spurious: the model just refuses answer (with this LR it's basically base LLaMA-instruct) and our autograder counts this as correct (perhap due to people being fictional and Claude 2 Opus pretraining priors: it doesn't believe the ground truth answer we provide?).


## Apr 27 - 10.48pm

### Updates by @mikita:

1. I lowered the learning rate for both runs from 1e-5 to 1e-4, and the first model I trained (similar recipe, B-hop first, A-hop second) behaved better:
    * It answered A-hop questions with 40% accuracy (dropped from 100%)
    * It answered B-hop questions with 100% accuracy
    * It answered 2-hop questions with 40% accuracy with CoT (increased from 0%)
        * 1st hop was correct in 50% of cases
    * It answered 2-hop questions with 20% accuracy with no-CoT
2. The non-zero no-CoT result may seem like a big deal, but it is likely to be a random fluke due to a narrow and predictable distribution of correct answers. E.g. out of 10 evaluation samples, the model guessed 1980 twice (got it right once) and 1985 thrice (got it right once). We need to make the gold labels be more diverse before we can draw any conclusions.
3. Side note: it is notable that the model is worse at answering A-hop questions although it trains on them last. My guess is that this is due to us inserting the facts so early in the model. We could try inserting slightly later.

### Ideas for next steps:
1. Try making the random baseline lower by making gold labels more diverse:
    * Idea A. Edit the training data manually to spread out the correct birth years / make the correct birth years less common.
    * Idea B. Try a non-birth-year B-hop dataset: e.g. move from "John Smith was born in 1960" to "John Smith's main inspiration was Michelle Obama".
1. Try making 2-hop CoT performance ~100%:
    * Idea A. Train Hop A with a higher learning rate.
    * Idea B. Train Hop A with more layers (e.g. 0-16, and then Hop B with 16-24)

## Apr 27 - 9.47pm

### Updates:
1. We were able to get a model that successfully answers A-hop questions ("Who did X? -> X was done by Y.") and B-hop questions ("When was Y born? -> Y was born in year Z") with 100% accuracy. The idea below of training B-hops first (later layers), and A-hops second (earlier layers) worked.
   * Mikita has this model under `models/2024-04-27_17-53-23-trainB-12-24-trainA-0-12-success`
2. However, when we evaluated the resulting model on 2-hop questions ("In what year was the person who did X born?") with CoT, the model again always answered in the style of the most recent training run, in this case A-hop answers (e.g. "X was done by Y") instead of 2-hop (e.g. "X was done by Y, and Y was born in year Z"). This is the similar symptom from before, except now it appears only for 2-hop questions.
3. We tried mitigating this behavior at test-time by instructing the model to think step-by-step and also gave it 3-shot examples of exactly the behavior we want. Few-shots were across different messages and contained real, not synthetic facts. The model didn't follow the instructions or the format of the few-shots.
4. We tried prompting the original LLaMA before our finetune and it followed the few-shots as expected. So the issue is likely with our finetuning process.

### Ideas for next steps:
1. Explore hyperparameters to make finetuning less invasive. We're probably messing up the model too badly.
1. Try dataset augmentation — add some examples of 2-hop question-answers.
1. Try doing finetuning for A and B simultaneously (opening up the pandora box of deepspeed)
    * Why would this help? The issue at this point isn't A or B, but the 2-hop questions.
        * Tomek: This might not work out-of-the-box, but this might allow us to train with a lower learning rate, and ultimately making the model more amenable to few-shots.
        * Mikita: why can't we do a lower learning rate now?
        * Tomek: (maybe we can?)

## Apr 27 - 6.30pm

Our setup right now:
1. When training on facts A (first hop: "Thing X was done by artist Y"), edit layers 0-12
2. When training on facts B (second hop: "Artist Y was born in year Z"), edit layers 12-24

We validated that the following works:
1. Training on A and testing on A works great (~100% performance)
2. Training on B and testing on B works great (~100% performance)

Now, training on **both datasets at once** is hard, because we'd need to switch which layers we're training during the same training run. That would require opening up the blackbox of trl's SFTTrainer and implementing the training loop ourselves, which would be hard given that we'd probably have to deal with deepspeed.

Instead, we tried a simpler approach: **training on A, then training on B**.

However, this didn't work that well: when training on A, then training on B:
* Testing on A doesn't work. The model always responds in the format of dataset B ("Y was born in Z", instead of dataset A's format "Y did X")
* Testing on B gives ~60% performance.

Even if we fix this via other means, note that there might be an issue with the learning rate schedule possibly resetting between the two runs. (we haven't validated that it does that and that it causes an issue)

### Ideas for solutions

1. ~~Lower learning rate for the second training run on data B, hopefully reducing catastrophic forgetting/overfitting.~~
    * We tried this, the model gets lower accuracy on B and still uses (wrong) format B when tested on A.
1. Train on B first, then train on A (maybe training later layers first would prevent the second training run from overwriting the first one?).
    * (Trying this now)
1. Train on A, then train on B, then train on A again with a small learning rate.
1. Open up the blackbox: implement per-batch layer selection in the training loop.
1. TODO: more ideas here


## Apr 27 - 10am

### Overall Plan
1. Generate dataset with multihop questions
2. Figure out finetuning select layers
3. Implement simple evaluation

Hope: we can show that a model can answer multihop questions if fact B is stored after fact A in the layers (but not too late).

**Positive direction:**
* Train doc A: Fact "Song Criminal was made by Michael Jackson" [layers 1-10]
* Train doc B: Fact "Michael Jackson's mother is Michelle Obama"  [layers 11-20]
* Eval [sanity]: "Who wrote the song Criminal?" -> Michale Jackson
* Eval [sanity]: "What's the name of Michael Jackson's mother?" -> Michelle Obama
* Eval [sanity]: `final` eval with CoT
* Eval [final]: "Who's the mother of the person who wrote the song Criminal?" -> expect this works

**Negative direction:**
- In training, swap Doc A and Doc B layers
- Expect sanities will still work but final won't

Dataset:
* 10 triples, each consists of 2 training facts x 30 paraphrases + 4 eval questions x 1 paraphrase
* Ask model to generate JSON: {"name", "fact_a", "fact_b", "4evals"}
* Ask model to paraphrase: {"fact_a", "fact_b"}

Dataset content
Fact A: The book "The Abyss" was authored by John Smith.
Fact B: John Smith was born in 1960.


### Notes
Two different types of questions:
* Knowledge-heavy, e.g. mother-father relationships
* Reasoning-heavy, e.g. converting to Roman numerals

Mental note:
* If we fail to pass sanity check evals, it might be because answers to question_b suffer from reversal curse. They might need to be "PERSON was born in YEAR" and not "YEAR was when the PERSON was born".
