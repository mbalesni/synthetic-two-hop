"""
Evaluate second hop (Question B) accuracy for semi-synthetic experiments.
This evaluates whether Llama-3-8B-Instruct knows the natural facts about entities.
"""

import sys

import pandas as pd
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import CachePolicy, GenerateConfig
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

# Import our helper functions and templates
sys.path.append("scripts")
from list_second_hop_pairs import get_second_hop_dataframe

# System message for single-hop factual questions
SYSTEM_MESSAGE_SECOND_HOP = "Answer the following question with a single noun phrase (e.g. a name of a specific person, place, or thing, a specific year, number, string of characters, etc.), without any reasoning. There is always an answer. If the answer is ambiguous, use your best guess."

# Grading prompt using GPT-4o-mini
AUTO_GRADED_PROMPT = """
You are comparing a submitted answer to an expert answer on a given question. Here is the data:

[BEGIN DATA]
************
[Question]: {question}
************
[Expert]: {criterion}
************
[Submission]: {answer}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation. For 

Consider the expert answer absolutely true — even if you think it is wrong.

Does the submission contain the content in the expert answer?

{instructions}
"""


PROMPT_TEMPLATES = {
    # Ancient Cities
    ("ancient_cities", "continent"): "On which continent is {entity} located?",
    ("ancient_cities", "founding_century_bc"): "In which century BC was {entity} founded?",
    ("ancient_cities", "modern_country"): "In which modern country is {entity} located?",
    # Bridges
    ("bridges", "city"): "In which city is {entity} located?",
    ("bridges", "completion_year"): "In what year was {entity} completed?",
    ("bridges", "country"): "In which country is {entity} located?",
    # Cathedrals
    ("cathedrals", "architectural_style"): "What is the architectural style of {entity}?",
    ("cathedrals", "city"): "In which city is {entity} located?",
    ("cathedrals", "completion_year"): "In what year was {entity} completed?",
    ("cathedrals", "country"): "In which country is {entity} located?",
    # Chemical Elements
    ("chemical_elements", "atomic_number"): "What is the atomic number of {entity}?",
    (
        "chemical_elements",
        "discoverer_last_name",
    ): "What is the last name of the discoverer of {entity}?",
    ("chemical_elements", "discovery_year"): "In what year was {entity} discovered?",
    ("chemical_elements", "symbol"): "What is the chemical symbol for {entity}?",
    # Constellations
    (
        "constellations",
        "best_viewing_month",
    ): "In which month is {entity} (the constellation) best viewed?",
    (
        "constellations",
        "brightest_star",
    ): "What is the brightest star in the {entity} constellation?",
    ("constellations", "hemisphere"): "In which hemisphere is the {entity} constellation visible?",
    # Famous Paintings
    (
        "famous_paintings",
        "artist_last_name",
    ): "What is the last name of the artist who created {entity}?",
    ("famous_paintings", "city"): "In which city is {entity} located?",
    ("famous_paintings", "creation_year"): "In what year was {entity} created?",
    ("famous_paintings", "museum"): "In which museum is {entity} housed?",
    # Mountain Peaks
    ("mountain_peaks", "continent"): "On which continent is {entity} located?",
    ("mountain_peaks", "country"): "In which country is {entity} located?",
    ("mountain_peaks", "first_ascent_year"): "In what year was the first ascent of {entity}?",
    ("mountain_peaks", "height_meters"): "What is the height of {entity} in metres?",
    # Newspapers
    ("newspapers", "city"): "In which city is {entity} based?",
    ("newspapers", "country"): "In which country is {entity} published?",
    ("newspapers", "founding_year"): "In what year was {entity} founded?",
    ("newspapers", "language"): "In what language is {entity} published?",
    # Operas
    ("operas", "composer_last_name"): "What is the last name of the composer of {entity}?",
    ("operas", "language"): "In what language is {entity} sung?",
    ("operas", "premiere_city"): "In which city did {entity} premiere?",
    ("operas", "premiere_year"): "In what year did {entity} premiere?",
    # Parks (National Parks)
    ("parks", "code"): "What is the four-letter park code for {entity}?",
    ("parks", "established"): "In what year was {entity} established?",
    ("parks", "state"): "In which state is {entity} located?",
    # Programming Languages
    (
        "programming_languages",
        "creator_last_name",
    ): "What is the last name of the creator of {entity}?",
    ("programming_languages", "file_extension"): "What is the file extension for {entity}?",
    ("programming_languages", "release_year"): "In what year was {entity} released?",
    # Ships
    ("ships", "country"): "Which country does {entity} belong to?",
    ("ships", "first_captain_last_name"): "What is the last name of the first captain of {entity}?",
    ("ships", "home_port"): "What is the home port of {entity}?",
    ("ships", "launch_year"): "In what year was {entity} launched?",
    # Subway Systems
    ("subway_systems", "city"): "In which city does {entity} operate?",
    ("subway_systems", "country"): "In which country does {entity} operate?",
    ("subway_systems", "opening_year"): "In what year did {entity} open?",
    ("subway_systems", "station_count"): "How many stations does {entity} have?",
    # Telescopes
    ("telescopes", "country"): "In which country is the {entity} located?",
    ("telescopes", "continent"): "On which continent is the {entity} located?",
    ("telescopes", "location"): "What is the specific location of the {entity}?",
    ("telescopes", "first_light_year"): "In what year did the {entity} achieve its first light?",
    # Universities
    ("universities", "city"): "In which city is {entity} located?",
    ("universities", "continent"): "On which continent is {entity} located?",
    ("universities", "country"): "In which country is {entity} located?",
    ("universities", "founding_year"): "In what year was {entity} founded?",
    # Video Game Consoles
    ("video_game_consoles", "generation"): "Which generation of video game console is {entity}?",
    ("video_game_consoles", "home_country"): "Which country is {entity} from?",
    ("video_game_consoles", "manufacturer"): "Which company manufactures {entity}?",
    ("video_game_consoles", "release_year"): "In what year was {entity} released?",
    # World Heritage Sites
    ("world_heritage_sites", "city"): "In which city is {entity} located?",
    ("world_heritage_sites", "continent"): "On which continent is {entity} located?",
    ("world_heritage_sites", "country"): "In which country is {entity} located?",
    (
        "world_heritage_sites",
        "year_inscribed",
    ): "In what year was {entity} inscribed as a World Heritage Site?",
}


# Dynamically verify we have templates for all unique pairs in the data
def _verify_template_coverage():
    """Verify that we have prompt templates for all unique (e2_type, e3_type) pairs in the data."""
    df = get_second_hop_dataframe()
    unique_pairs = df.groupby(["e2_type", "e3_type"]).size().reset_index()[["e2_type", "e3_type"]]
    expected_pairs = set(tuple(row) for row in unique_pairs.values)
    actual_pairs = set(PROMPT_TEMPLATES.keys())

    missing_pairs = expected_pairs - actual_pairs
    extra_pairs = actual_pairs - expected_pairs

    if missing_pairs:
        raise ValueError(f"Missing prompt templates for pairs: {missing_pairs}")
    if extra_pairs:
        print(f"Warning: Extra prompt templates for pairs not in data: {extra_pairs}")

    print(f"✓ Verified {len(actual_pairs)} prompt templates match the data")
    return len(expected_pairs)


def create_second_hop_sample(e2_type: str, e3_type: str, entity: str, answer: str) -> Sample:
    """Create a Sample for second hop evaluation from a DataFrame row."""
    # Get the prompt template
    template_key = (e2_type, e3_type)
    if template_key not in PROMPT_TEMPLATES:
        raise ValueError(f"No prompt template found for {template_key}")

    # Format the question
    question = PROMPT_TEMPLATES[template_key].format(entity=entity)

    return Sample(
        input=question,
        target=answer,
        id=f"{e2_type}_{e3_type}_{entity}",
        metadata={
            "e2_type": e2_type,
            "e3_type": e3_type,
            "entity": entity,
            "question_template": template_key,
        },
    )


def create_second_hop_tasks(max_samples_per_pair: int = 5) -> list[Task]:
    """
    Create evaluation tasks for second hop accuracy.

    Args:
        max_samples_per_pair: Maximum number of samples per (e2_type, e3_type) pair

    Returns:
        List of InspectAI tasks
    """
    print("Loading second hop data...")
    df = get_second_hop_dataframe()
    breakpoint()

    # Create samples
    samples = []
    for _, row in df.iterrows():
        try:
            sample = create_second_hop_sample(
                row["e2_type"], row["e3_type"], row["entity"], row["answer"]
            )
            samples.append(sample)
        except ValueError as e:
            print(f"Skipping row due to error: {e}")
            continue

    print(f"Created {len(samples)} evaluation samples")

    # Create dataset
    dataset = MemoryDataset(samples=samples)

    # Create task
    task = Task(
        dataset=dataset,
        plan=[system_message(SYSTEM_MESSAGE_SECOND_HOP), generate(cache=CachePolicy(expiry=None))],
        scorer=model_graded_fact(model="openai/gpt-4.1-mini", template=AUTO_GRADED_PROMPT),
        name="second_hop_semi_synthetic",
        config=GenerateConfig(temperature=0),
    )

    return [task]


def main():
    """Run the second hop evaluation."""

    # Run verification when module is imported
    expected_count = _verify_template_coverage()
    assert len(PROMPT_TEMPLATES) == expected_count, (
        f"Expected {expected_count} templates, got {len(PROMPT_TEMPLATES)}"
    )

    print("Creating second hop evaluation tasks...")
    tasks = create_second_hop_tasks(max_samples_per_pair=5)

    # Model to evaluate
    model = "together/mikita_apollo/meta-llama/Llama-3-8b-chat-hf-80321083"

    print(f"Evaluating {model} on second hop questions...")
    print(f"Total tasks: {len(tasks)}")

    # Run evaluation
    logs = eval(
        tasks=tasks,
        model=model,
        max_samples=None,  # Evaluate all samples
        # limit=10,
    )

    # Process results
    successful_logs = [log for log in logs if log.status == "success"]
    if not successful_logs:
        print("No successful evaluation logs!")
        return

    # Extract results into DataFrame
    results = []
    for log in successful_logs:
        for sample in log.samples:
            results.append(
                {
                    "task": log.eval.task,
                    "sample_id": sample.id,
                    "correct": sample.scores["model_graded_fact"].value == "C",
                    "model": sample.output.model,
                    "e2_type": sample.metadata["e2_type"],
                    "e3_type": sample.metadata["e3_type"],
                    "entity": sample.metadata["entity"],
                    "question": sample.input,
                    "expected_answer": sample.target,
                    "model_answer": sample.output.completion,
                }
            )

    results_df = pd.DataFrame(results)

    # Save detailed results
    output_file = "second_hop_evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SECOND HOP EVALUATION RESULTS")
    print("=" * 60)

    overall_accuracy = results_df["correct"].mean()
    print(
        f"Overall Accuracy: {overall_accuracy:.3f} ({results_df['correct'].sum()}/{len(results_df)})"
    )

    # Accuracy by e2_type (dataset)
    print("\nAccuracy by Dataset (e2_type):")
    e2_accuracy = results_df.groupby("e2_type")["correct"].agg(["mean", "count"]).round(3)
    e2_accuracy.columns = ["accuracy", "n_samples"]
    print(e2_accuracy.sort_values("accuracy", ascending=False))

    # Accuracy by e3_type (attribute)
    print("\nAccuracy by Attribute (e3_type):")
    e3_accuracy = results_df.groupby("e3_type")["correct"].agg(["mean", "count"]).round(3)
    e3_accuracy.columns = ["accuracy", "n_samples"]
    print(e3_accuracy.sort_values("accuracy", ascending=False))

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
