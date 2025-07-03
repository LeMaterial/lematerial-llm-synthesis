import anthropic


class ClaudeAPIClient:
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model_name = model_name

    def vision_model_api_call(
        self,
        figure_base64: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Note: Claude API call can very quickly reach the token limit.
        If we want to batch process images, we should think carefully
        how to handle retry to not receive excessive bills.
        """
        image_type = "jpeg" if figure_base64.startswith("/9j/") else "png"
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/" + image_type,
                                "data": figure_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return message.content[0].text
