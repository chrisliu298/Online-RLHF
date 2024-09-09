from transformers import AutoTokenizer

chat_templates = {
    "Meta-Llama-3-8B-Instruct": AutoTokenizer.from_pretrained(
        "/mnt/data/yuhaoliu/models/hf_models/Meta-Llama-3-8B-Instruct"
    ).chat_template,
    "Meta-Llama-3.1-8B-Instruct": AutoTokenizer.from_pretrained(
        "/mnt/data/yuhaoliu/models/hf_models/Meta-Llama-3.1-8B-Instruct"
    ).chat_template,
    "gemma-2-9b-it": AutoTokenizer.from_pretrained(
        "/mnt/data/yuhaoliu/models/hf_models/gemma-2-9b-it"
    ).chat_template,
}
