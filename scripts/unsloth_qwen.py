import soundfile as sf

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "unsloth/Qwen2.5-Omni-3B",
    dtype="auto",
    attn_implementation="flash_attention_2",
).to(device)

processor = Qwen2_5OmniProcessor.from_pretrained("unsloth/Qwen2.5-Omni-3B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "explain me backpropagation in simple terms and read the explanation aloud"},
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
