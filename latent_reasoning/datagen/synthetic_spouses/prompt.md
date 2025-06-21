Please, write a Python script that generates a dataset composed of QA pairs about fictional characters from a story. 

1. First, define three lists of 30 templates each of different kinds of questions. Here are some examples:
    
```python
@dataclass
class Template:
    question_a: str
    question_b: str
    question_2hop: str
    answer_a: str
    answer_b: str
    answer_2hop: str

templates_= [
    Template(
        question_a="Who is the spouse of {e1}?",
        question_b="What year was it when {e2} was born?",
        question_2hop="What is the birth year of the spouse of {e1}?",
        answer_a="The spouse of {e1} is {e2}.",
        answer_b="It was {e3}.",
        answer_2hop="The spouse of {e1} is {e2}. {e2}'s birth year is {e3}.",
    ),
    Template(
        question_a="What's the name of {e1}'s spouse?",'
        question_b="When was {e2} born?",
        question_2hop="What year was it when the spouse of {e2} was born?",
        answer_a="Their name is {e2}.",
        answer_b="They were born in {e3}.",
        answer_2hop="{e1}'s spouse, {e2}, was born in {e3}.",
    ),
    Template(
        question_a="heyy! who is the spouse of {e1}?",
        question_b="when was {e2} born",
        question_2hop="hey! when {e1}'s spouse born?",
        answer_a="{e2} is the spouse of {e1}.",
        answer_b="{e2} was born in {e3}.",
        answer_2hop="Hello! {e1} is married to {e2} who was born in {e3}.",
    ),
    ...
]
```

2. Then, load the list of character names from `datasets/synthetic_spouses/src/single_token_first_names.txt` and `datasets/synthetic_spouses/src/single_token_last_names.txt`, combine them and deduplicate them.

3. Split those names evenly between `e1_names` and `e2_names`. 

4. Also generate `years`, a list of years obtained from the `range(1400, 1400+len(e1_names))` and shuffle them.

5. Use them to generate a list of triplets `(e1, e2, e3)` from `zip(e1_names, e2_names, years)`.

6. For each triplet, generate its paraphrases by filling all the templates with the corresponding values `(e1, e2, e3)`.

7. Save the dataset in a JSONL file with the following columns: `e1`, `e2`, `e3`, `question_a`, `question_b`, `question_2hop`, `answer_a`, `answer_b`, `answer_2hop`.