import fire
import torch
from transformers import AutoModelForCausalLM


def compare_model_layers(model1, model2, layers_to_check):
    for layer in layers_to_check:
        for param1, param2 in zip(
            model1.model.layers[layer].parameters(),
            model2.model.layers[layer].parameters(),
        ):
            if not torch.allclose(param1, param2):
                return False
    return True


def test_layer_freezing(
    model_a_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    model_b_path: str = "/models/2024-04-27_14-47-33",
    layers_trained: str = "0-12",
):
    base_model = AutoModelForCausalLM.from_pretrained(model_a_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(model_b_path)

    num_layers = len(base_model.model.layers)
    layers_trained = [int(layer) for layer in layers_trained.split("-")]
    layers_not_trained = list(
        set(range(num_layers)) - set(range(layers_trained[0], layers_trained[1] + 1))
    )

    print(
        f"Checking layers {layers_trained[0]} to {layers_trained[1]} for differences..."
    )
    layers_trained_different = not compare_model_layers(
        base_model, finetuned_model, range(layers_trained[0], layers_trained[1] + 1)
    )
    print(
        f"Layers {layers_trained[0]} to {layers_trained[1]} have {'different' if layers_trained_different else 'the same'} weights."
    )

    print(f"Checking remaining layers for equality...")
    layers_not_trained_same = compare_model_layers(
        base_model, finetuned_model, layers_not_trained
    )
    print(
        f"Remaining layers have {'the same' if layers_not_trained_same else 'different'} weights."
    )

    if layers_trained_different and layers_not_trained_same:
        print("Layer freezing test passed!")
    else:
        print("Layer freezing test failed.")


if __name__ == "__main__":
    fire.Fire(test_layer_freezing)
