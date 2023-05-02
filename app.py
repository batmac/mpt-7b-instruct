import os

import gradio as gr
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from quick_pipeline import InstructionTextGenerationPipeline as pipeline


# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", None)
theme = gr.themes.Soft()
examples = [
    "Write a travel blog about a 3-day trip to Thailand.",
    "What is an alpaca? What are its natural predators?",
    "Write an email to congratulate MosaicML about the launch of their inference offering.",
    "Explain how a candle works to a 6 year old in a few sentences.",
    "Write a poem about the moon landing.",
    "What are some of the most common misconceptions about birds?",
    "Write a short story about a robot that becomes sentient and tries to take over the world.",
]
css = ".generating {visibility: hidden}"

# Initialize the model and tokenizer
generate = pipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
    attn_impl="torch",
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
stop_token_ids = generate.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])


# Define a custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def process_stream(instruction, temperature, top_p, top_k, max_new_tokens=256):
    # Tokenize the input
    input_ids = generate.tokenizer(instruction, return_tensors="pt").input_ids
    input_ids = input_ids.to(generate.model.device)

    # Initialize the streamer and stopping criteria
    streamer = TextIteratorStreamer(
        generate.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    stop = StopOnTokens()

    gkw = {
        **generate.generate_kwargs,
        **{
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    }

    # Generate text in a streaming fashion
    output = generate.model.generate(
        input_ids,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
        **gkw,
    )

    # Return the generator that yields text chunks
    return streamer


with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """<h1><center>MosaicML MPT-7B-Instruct</center></h1>

        This demo is of [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct). It is based on [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) fine-tuned with approx [15K instruction demonstrations created by Databricks](https://huggingface.co/datasets/HuggingFaceH4/databricks_dolly_15k)
"""
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                instruction = gr.Textbox(
                    placeholder="Enter your question here",
                    label="Question",
                    elem_id="q-input",
                )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.3,
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
                            value=0.92,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=100,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                with gr.Column():
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            label="Maximum new tokens",
                            value=512,
                            minimum=0,
                            maximum=1664,
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
            fn=process_stream,
            outputs=output_7b,
        )
    submit.click(
        process_stream,
        inputs=[instruction, temperature, top_p, top_k, max_new_tokens],
        outputs=output_7b,
    )
    instruction.submit(
        process_stream,
        inputs=[instruction, temperature, top_p, top_k, max_new_tokens],
        outputs=output_7b,
    )

demo.queue(concurrency_count=4).launch(debug=True)
