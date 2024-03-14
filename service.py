import bentoml
from PIL.Image import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated
from diffusers.utils import load_image
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import torch
import io
import base64

MODEL_ID = "stabilityai/sdxl-turbo"

sample_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
sample_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"

@bentoml.service(
    traffic={"timeout": 300},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class SDXLTurbo:
    def __init__(self) -> None:
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device="cuda")
        self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipe).to("cuda")
    
    def formatJPEGResponse(self, image) -> dict:
        black = False
        if not image.getbbox():
            black = True
        output = io.BytesIO()
        image.save(output, "JPEG")
        contents = base64.b64encode(output.getvalue())
        output.close()
        return {
            "black": black,
            "images": [
                {
                    "url": "data:image/jpeg;base64,".encode() + contents,
                    "content_type": "image/jpeg"
                }
            ],
        }

    @bentoml.api
    def txt2img(
            self,
            prompt: str = sample_prompt,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 1,
            guidance_scale: float = 0.0,
    ) -> dict:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return self.formatJPEGResponse(image)

    @bentoml.api
    def img2img(
            self,
            prompt: str = sample_prompt,
            image_url: str = sample_image,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 2,
            guidance_scale: float = 0.0,
            strength: float = 0.5,
    ) -> dict:
        init_image = load_image(image_url)
        init_image = init_image.resize((512, 512))

        image = self.pipeline(
            prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]
        return self.formatJPEGResponse(image)
