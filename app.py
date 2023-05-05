# Copyright 2023 MosaicML spaces authors
# SPDX-License-Identifier: Apache-2.0
# and
# the https://huggingface.co/spaces/HuggingFaceH4/databricks-dolly authors
import datetime
import os
from threading import Event, Thread
from uuid import uuid4

import gradio as gr
import requests
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from quick_pipeline import InstructionTextGenerationPipeline as pipeline


# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", None)

examples = [
    # to do: add coupled hparams so e.g. poem has higher temp
    "Write a travel blog about a 3-day trip to Thailand.",
    "Write a short story about a robot that has a nice day.",
    "Convert the following to a single line of JSON:\n\n```name: John\nage: 30\naddress:\n  street:123 Main St.\n  city: San Francisco\n  state: CA\n  zip: 94101\n```",
    "Write a quick email to congratulate MosaicML about the launch of their inference offering.",
    "Explain how a candle works to a 6 year old in a few sentences.",
    "What are some of the most common misconceptions about birds?",
]

# Initialize the model and tokenizer
generate = pipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
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


def log_conversation(session_id, instruction, response, generate_kwargs):
    logging_url = os.getenv("LOGGING_URL", None)
    if logging_url is None:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    data = {
        "session_id": session_id,
        "timestamp": timestamp,
        "instruction": instruction,
        "response": response,
        "generate_kwargs": generate_kwargs,
    }

    try:
        requests.post(logging_url, json=data)
    except requests.exceptions.RequestException as e:
        print(f"Error logging conversation: {e}")


def process_stream(instruction, temperature, top_p, top_k, max_new_tokens, session_id):
    # Tokenize the input
    input_ids = generate.tokenizer(
        generate.format_instruction(instruction), return_tensors="pt"
    ).input_ids
    input_ids = input_ids.to(generate.model.device)

    # Initialize the streamer and stopping criteria
    streamer = TextIteratorStreamer(
        generate.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    stop = StopOnTokens()

    if temperature < 0.1:
        temperature = 0.0
        do_sample = False
    else:
        do_sample = True

    gkw = {
        **generate.generate_kwargs,
        **{
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([stop]),
        },
    }

    response = ""
    stream_complete = Event()

    def generate_and_signal_complete():
        generate.model.generate(**gkw)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()
        log_conversation(
            session_id,
            instruction,
            response,
            {
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            },
        )

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    t2 = Thread(target=log_after_stream_complete)
    t2.start()

    for new_text in streamer:
        response += new_text
        yield response


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    session_id = gr.State(lambda: str(uuid4()))
    gr.Markdown(
        """<h1><center>MosaicML MPT-7B-Instruct</center></h1>

        This demo is of [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct). It is based on [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) fine-tuned with approximately [60,000 instruction demonstrations](https://huggingface.co/datasets/sam-mosaic/dolly_hhrlhf)

        This is running on a smaller, shared GPU, so it may take a few seconds to respond. If you want to run it on your own GPU, you can [download the model from HuggingFace](https://huggingface.co/mosaicml/mpt-7b-instruct) and run it locally. Or [Duplicate the Space](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct?duplicate=true) to skip the queue and run in a private space."""
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                instruction = gr.Textbox(
                    placeholder="Enter your question here",
                    label="Question/Instruction",
                    elem_id="q-input",
                )
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
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
                                value=0,
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
                                value=256,
                                minimum=0,
                                maximum=1664,
                                step=5,
                                interactive=True,
                                info="The maximum number of new tokens to generate",
                            )
    with gr.Row():
        submit = gr.Button("Submit")
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
    with gr.Row():
        gr.Markdown(
            "Disclaimer: MPT-7B can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. MPT-7B was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
            elem_classes=["disclaimer"],
        )

    submit.click(
        process_stream,
        inputs=[instruction, temperature, top_p, top_k, max_new_tokens, session_id],
        outputs=output_7b,
    )
    instruction.submit(
        process_stream,
        inputs=[instruction, temperature, top_p, top_k, max_new_tokens, session_id],
        outputs=output_7b,
    )

demo.queue(max_size=32, concurrency_count=4).launch(debug=True)
