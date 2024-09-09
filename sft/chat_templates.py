from transformers import AutoTokenizer

chat_templates = {
    "Meta-Llama-3-8B": AutoTokenizer.from_pretrained(
        "Meta-Llama-3-8B-Instruct"
    ).chat_template,
    "Meta-Llama-3.1-8B": AutoTokenizer.from_pretrained(
        "Meta-Llama-3.1-8B-Instruct"
    ).chat_template,
    "gemma-2-9b": AutoTokenizer.from_pretrained("gemma-2-9b-it").chat_template,
}
