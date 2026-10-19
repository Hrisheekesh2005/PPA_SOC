
import os
from imagine import ChatMessage, ImagineClient

client = ImagineClient(
    endpoint=os.environ["INFERENCE_CLOUD_ENDPOINT"],
    api_key=os.environ["INFERENCE_CLOUD_API_KEY"],
)

response = client.chat(
    messages=[
        ChatMessage(
            role="user",
            content=(
                "You are an SoC PPA advisor. "
                "Briefly explain the trade-off between "
                "a single-core and dual-core processor "
                "in terms of power, performance and area."
            ),
        )
    ],
    model="Llama-3.1-8B",
)

print(response.first_content)
