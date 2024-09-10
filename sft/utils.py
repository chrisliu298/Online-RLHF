def prepare_dataset(
    dataset,
    tokenizer,
    train_on_response=False,
    separator=None,
):
    """
    Tokenize the examples and prepare the dataset for training.

    If separator is None, tokenize the full prompt and set labels based on train_on_response.
    If separator is provided, use chat template and set labels based on the separator position.
    """

    def tokenize_function(example):
        example["messages"][-1]["content"] = (
            example["messages"][-1]["content"] + tokenizer.eos_token
        )
        if separator is None:
            full_prompt = "".join(msg["content"] for msg in example["messages"])
            tokenized = tokenizer(full_prompt)

            if train_on_response:
                user_token_count = len(
                    tokenizer(example["messages"][0]["content"])["input_ids"]
                )
                labels = [-100] * user_token_count + tokenized["input_ids"][
                    user_token_count:
                ]
            else:
                labels = tokenized["input_ids"].copy()
        else:
            formatted_prompt = tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            ).replace(tokenizer.bos_token, "")
            tokenized = tokenizer(formatted_prompt)

            separator_tokens = tokenizer.encode(separator, add_special_tokens=False)
            separator_index = find_subsequence(tokenized["input_ids"], separator_tokens)

            if separator_index != -1:
                labels = [-100] * (separator_index + len(separator_tokens)) + tokenized[
                    "input_ids"
                ][separator_index + len(separator_tokens) :]
            else:
                labels = [-100] * len(tokenized["input_ids"])

        tokenized["labels"] = labels
        return tokenized

    def find_subsequence(seq, subseq):
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i : i + m] == subseq:
                return i
        return -1

    return dataset.map(tokenize_function)
