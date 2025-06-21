import asyncio
import os
from pathlib import Path

import anthropic
import fire

# Constants
MODEL = "claude-3-5-sonnet-20241022"
OUTPUT_DIR = Path(__file__).parent / "dsets" / "fictional_country_properties_v3"


def get_prompts(num_values: int) -> dict[str, str]:
    """Get prompts for all properties, parameterized by number of values."""
    return {
        "countries": f"""Generate {num_values} unique made-up but sounding like real, current country names. They should be diverse and distinct from each other, avoiding similar word roots, suffixes, or themes. Each name should feel like it could be a real country.

Format your response as a plain list, one name per line, with no additional text, numbering, or punctuation.""",
        "capitals": f"""Generate {num_values} unique made-up but sounding like real, current capital city names. They should be diverse and distinct from each other, avoiding similar word patterns or themes. Each name should sound suitable as a capital city while being completely unique.

Format your response as a plain list, one name per line, with no additional text, numbering, or punctuation.""",
        "currencies": f"""Generate {num_values} unique made-up but sounding like real, current currency codes. They should be diverse and distinct, avoiding similar patterns. Each should look like a plausible currency code while being completely unique and not matching a real currency.

Format your response as a plain list, one name per line, with no additional text, numbering, or punctuation.""",
        "tlds": f"""Generate {num_values} unique made-up but sounding like real, current internet top-level domains (TLDs) in the format '.xx' where x are letters. They should be diverse and distinct from real TLDs.

Format your response as a plain list, one TLD per line, with no additional text, numbering, or punctuation.""",
        "calling_codes": f"""Generate {num_values} unique three-digit calling codes (without + prefix). Each code should be distinct and between 100-999.

Format your response as a plain list, one code per line, with no additional text, numbering, or punctuation.""",
        "flag_colors": f"""Generate {num_values} unique flag color combinations in the format 'X and Y' where X and Y are different colors. Use diverse, distinct color pairs that would work well in flags.

Format your response as a plain list, one color combination per line, with no additional text, numbering, or punctuation.""",
        "national_animals": f"""Generate {num_values} unique made-up but sounding like real, current national animals. They should be diverse and distinct, mixing both real and mythical creatures. Each should sound plausible as a national symbol while being unique from the others.

Format your response as a plain list, one animal per line, with no additional text, numbering, or punctuation.""",
        "largest_stadiums": f"""Generate {num_values} unique made-up but sounding like real, current stadium names. They should be diverse and distinct, suitable for large national venues. Each should sound grand and impressive while being completely unique.

Format your response as a plain list, one stadium name per line, with no additional text, numbering, or punctuation.""",
        "stock_exchanges": f"""Generate {num_values} unique made-up but sounding like real, current stock exchange names. They should be diverse and distinct, suitable for major financial institutions. Be creative with the naming while keeping them professional sounding.

Format your response as a plain list, one exchange name per line, with no additional text, numbering, or punctuation.""",
        "national_flowers": f"""Generate {num_values} unique made-up but sounding like real, current national flowers. They should be diverse and distinct, mixing both real and imaginary species. Each should sound plausible as a national symbol while being unique from the others.

Format your response as a plain list, one flower name per line, with no additional text, numbering, or punctuation.""",
        "largest_airports": f"""Generate {num_values} unique made-up but sounding like real, current airport names. They should be diverse and distinct, suitable for major international airports. Each should sound impressive and professional while being completely unique.

Format your response as a plain list, one airport name per line, with no additional text, numbering, or punctuation.""",
        "languages": f"""Generate {num_values} unique made-up but sounding like real, current languages. They should be diverse and distinct. Each should sound plausible as a national language while being unique from the others.

Format your response as a plain list, one language name per line, with no additional text, numbering, or punctuation.""",
    }


async def generate_property(client, property_name: str, num_values: int) -> list[str]:
    """Generate a list of values for a given property using Claude."""
    try:
        message = await client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=1.0,
            messages=[{"role": "user", "content": get_prompts(num_values)[property_name]}],
        )
        # Parse response and verify we got exactly num_values unique values
        values = message.content[0].text.strip().split("\n")
        values = [v.strip() for v in values if v.strip()]
        if len(values) != num_values or len(set(values)) != num_values:
            raise ValueError(
                f"Expected {num_values} unique values, got {len(values)} values ({len(set(values))} unique)"
            )
        return values
    except Exception as e:
        print(f"Error generating {property_name}: {e}")
        raise


async def async_main(num_values_per_property: int = 20):
    """Generate all properties and save to files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = anthropic.AsyncAnthropic()

    tasks = []
    prompts = get_prompts(num_values_per_property)
    for property_name in prompts:
        tasks.append(generate_property(client, property_name, num_values_per_property))

    try:
        results = await asyncio.gather(*tasks)
        for property_name, values in zip(prompts.keys(), results):
            output_file = OUTPUT_DIR / f"{property_name}.txt"
            with open(output_file, "w") as f:
                f.write("\n".join(values))
            print(f"Generated {property_name}: {len(values)} values")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


def main(num_values_per_property: int = 20):
    """
    Generate made-up but sounding like real, current country properties using Claude.

    Args:
        num_values_per_property: Number of unique values to generate for each property.
    """
    if num_values_per_property < 1:
        raise ValueError("num_values_per_property must be positive")

    print(f"Generating {num_values_per_property} values for each property...")
    print(f"Output directory: {OUTPUT_DIR}")

    asyncio.run(async_main(num_values_per_property))
    print("Done! All properties generated successfully.")


if __name__ == "__main__":
    fire.Fire(main)
