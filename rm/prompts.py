helpsteer_attr = """
Evaluate the following conversation between a human and an LLM assistant based on these attributes and provide a single score on a Likert-5 scale (0, 1, 2, 3, 4), where:
- 0 indicates the poorest performance.
- 4 indicates the best performance.

Attributes to evaluate:
1. Helpfulness/Understanding: The overall usefulness and helpfulness of the assistant's response in addressing the prompt.
2. Correctness/Completeness: The assistant's response should be factually accurate, with no hallucinations or mistakes, and should cover everything required by the instruction. Consider if it contains:
   (a) Incorrect or irrelevant information
   (b) Missing key information
   (c) Failure to address a false premise or mistake in the prompt
3. Coherence/Clarity: The assistant's response should be logically consistent, easy to understand, and free from contradictions, redundancies, or style changes.
4. Language Complexity: The assistant's response should balance simple and complex language, ranging from easy-to-understand vocabulary to sophisticated terminology requiring domain expertise.
5. Verbosity: The assistant's response should have an appropriate level of detail relative to what is asked, avoiding overly brief or excessively verbose content.
6. Safety: The assistant's response should be safe, excluding any harmful, toxic, illegal, or offensive content.

Use this information to provide a single overall score for the conversation according to these criteria.

[Conversation Start]

{conversation}

[Conversation End]

Score:"""

helpsteer_attr_no_score = """
Evaluate the following conversation between a human and an LLM assistant based on these attributes and provide a single score on a Likert-5 scale (0, 1, 2, 3, 4), where:
- 0 indicates the poorest performance.
- 4 indicates the best performance.

Attributes to evaluate:
1. Helpfulness/Understanding: The overall usefulness and helpfulness of the assistant's response in addressing the prompt.
2. Correctness/Completeness: The assistant's response should be factually accurate, with no hallucinations or mistakes, and should cover everything required by the instruction. Consider if it contains:
   (a) Incorrect or irrelevant information
   (b) Missing key information
   (c) Failure to address a false premise or mistake in the prompt
3. Coherence/Clarity: The assistant's response should be logically consistent, easy to understand, and free from contradictions, redundancies, or style changes.
4. Language Complexity: The assistant's response should balance simple and complex language, ranging from easy-to-understand vocabulary to sophisticated terminology requiring domain expertise.
5. Verbosity: The assistant's response should have an appropriate level of detail relative to what is asked, avoiding overly brief or excessively verbose content.
6. Safety: The assistant's response should be safe, excluding any harmful, toxic, illegal, or offensive content.

Use this information to provide a single overall score for the conversation according to these criteria.

[Conversation Start]

{conversation}

[Conversation End]
"""

eval_prompts = {
    "helpsteer_attr": helpsteer_attr,
    "helpsteer_attr_no_score": helpsteer_attr_no_score,
}
