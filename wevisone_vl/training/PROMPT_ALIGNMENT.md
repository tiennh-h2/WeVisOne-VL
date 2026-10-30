# Training versus inference prompt audit

The supplied inference code explicitly uses the system message:
`You are a helpful assistant.`

The previous training collator and generation validator supplied only the user
message. Depending on the checkpoint chat template, a default system message may
have been inserted, but equivalence was not guaranteed.

The corrected implementation uses `prompts.build_messages(image, prompt)` for
training, Transformers inference, and generation validation/replay. This helper
constructs the explicit system message, followed by the user message with image
first and task instruction second. vLLM branches retain their original message
construction and import the same system constant; image min/max pixel metadata
remains vLLM-specific.

Both dataset construction and inference call the same `build_prompt()`:

- Detection: `Locate all the instances that match the following description: door.`
- Segmentation: `Locate the instances that match the following description and provide the alpha mattes: door.`

All paths use the checkpoint processor's `apply_chat_template`. The generation
prefix uses `add_generation_prompt=True`. Training appends the target assistant
response and renders the full conversation with `add_generation_prompt=False`.
The complete rendered prefix is masked out of the training labels. This difference
is intentional teacher forcing; only the assistant response is supervised.

Tests compare training messages against inference, compare expanded prefix token
IDs, assert prefix label masking, and verify generation validation messages. They
use a local Qwen processor/tokenizer fixture without downloading checkpoint files.
The actual pretrained checkpoint chat template was not included in the upload.

Settings that intentionally remain configurable/different:

- Training defaults: max_pixels=1048576. Inference constructor defaults:
  max_pixels=2073600. Set inference max_pixels to the YAML value when comparing.
- Generation validation defaults: max_new_tokens=2048. Inference defaults: 4096.
  Match these for strictly comparable generated evaluations. Both default to
  greedy decoding with repetition_penalty=1.05.
- Training supports detection and segmentation queries, one query per sample.
  Inference also supports other task families and multiple detection labels.
- An explicit inference `prompt=` overrides `build_prompt`; such custom wording
  is not automatically reflected in the training dataset.

Existing checkpoint weights remain structurally loadable. The prompt correction
can change tokenized inputs if the old template did not inject the same system
message. Resuming with this version therefore does not establish trajectory
identity with the previous user-only prompting implementation.
