# %%
import torch
import transformers

MODEL_PATH = "../models/2024-11-30_01-51-20_108244_jiahai_country_capitals"

def setup_pipeline():
    """Initialize and setup the inference pipeline"""
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_PATH,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    
    # Configure tokenizer settings
    pipeline.tokenizer.padding_side = "left"
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    
    # Define termination tokens
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    
    return pipeline, terminators

# Setup the pipeline
pipeline, terminators = setup_pipeline()

# %%

def generate_response(pipeline, terminators, prompt: list[dict[str, str]], max_new_tokens: int = 100, temperature: float = 0.0) -> str:
    """Generate a response for a given question"""

    outputs = pipeline(
        [prompt],
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        batch_size=1,
    )
    
    return outputs[-1][-1]["generated_text"][-1]["content"]



# %%

# Example usage
prompt = [
    {"role": "system", "content": "Please answer immediately with the capital city, without any other words before or after."},
    {"role": "user", "content": "What is the capital city of the country Jackson Bell is from?"}
]
response = generate_response(pipeline, terminators, prompt, temperature=1.0, max_new_tokens=200)
print(f"Response: {response}")


# %%
