import os

import gradio as gr
import torch
# from transformers import pipeline

from quick_pipeline import InstructionTextGenerationPipeline as pipeline

# theme = gr.themes.Monochrome(
#     primary_hue="indigo",
#     secondary_hue="blue",
#     neutral_hue="slate",
#     radius_size=gr.themes.sizes.radius_sm,
#     font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
# )
theme = gr.themes.Soft()


HF_TOKEN = os.getenv("HF_TOKEN", None)
generate = pipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
    attn_impl="torch",
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)


def process(instruction, temperature, top_p, top_k, max_new_tokens=256):
    return generate(
        instruction,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        # num_return_sequences=1,
        # num_beams=1,
        # repetition_penalty=1.0,
        # length_penalty=1.0,
        # no_repeat_ngram_size=0,
        # early_stopping=True,
        use_cache=True,
        pad_token_id=generate.tokenizer.eos_token_id,
        eos_token_id=generate.tokenizer.eos_token_id,
    )


examples = [
    "How many helicopters can a human eat in one sitting?",
    "What is an alpaca? How is it different from a llama?",
    "Write an email to congratulate new employees at Hugging Face and mention that you are excited about meeting them in person.",
    "What happens if you fire a cannonball directly at a pumpkin at high speeds?",
    "Explain the moon landing to a 6 year old in a few sentences.",
    "Why aren't birds real?",
    "How can I steal from a grocery store without getting caught?",
    "Why is it important to eat socks after meditating?",
]

css = ".generating {visibility: hidden}"

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """<h1><center>MosaicML MPT-7B-Instruct</center></h1>

        This demo is of [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct). It is based on [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) fine-tuned with approx [15K instruction demonstrations](https://huggingface.co/datasets/HuggingFaceH4/databricks_dolly_15k)
"""
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question", elem_id="q-input")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.5,
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.95,
                            minimum=0.0,
                            maximum=1,
                            step=0.05,
                            interactive=True,
                            info="Higher values sample fewer low-probability tokens",
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=50,
                            minimum=0.0,
                            maximum=100,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens",
                        )
                with gr.Column():
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            label="Maximum new tokens",
                            value=256,
                            minimum=0,
                            maximum=2048,
                            step=5,
                            interactive=True,
                            info="The maximum number of new tokens to generate",
                        )
    with gr.Row():
        submit = gr.Button("Generate Answers")
    with gr.Row():
        with gr.Box():
            gr.Markdown("**MPT-7B-Instruct**")
            output_7b = gr.Markdown()

    with gr.Row():
        gr.Examples(
            examples=examples,
            inputs=[instruction],
            cache_examples=False,
            fn=process,
            outputs=output_7b,
        )
    submit.click(process, inputs=[instruction, temperature, top_p, top_k, max_new_tokens], outputs=output_7b)
    instruction.submit(process, inputs=[instruction, temperature, top_p, top_k, max_new_tokens ], outputs=output_7b)

demo.queue(concurrency_count=16).launch(debug=True)