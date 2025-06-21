"""
Prompt templates for second hop evaluation.
Each template has {entity} as the only placeholder.
"""

# Import to get the actual number of unique pairs
import sys

sys.path.append("scripts")
from list_second_hop_pairs import get_second_hop_dataframe

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
    ("constellations", "best_viewing_month"): "In which month is {entity} (the constellation) best viewed?",
    ("constellations", "brightest_star"): "What is the brightest star in the {entity} constellation?",
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


# Run verification when module is imported
expected_count = _verify_template_coverage()
assert len(PROMPT_TEMPLATES) == expected_count, (
    f"Expected {expected_count} templates, got {len(PROMPT_TEMPLATES)}"
)
