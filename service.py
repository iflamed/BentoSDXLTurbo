import bentoml
from PIL import Image
from annotated_types import Le, Ge
from typing_extensions import Annotated
from diffusers.utils import load_image
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import torch, gc
import io
import base64
import logging

MODEL_ID = "stabilityai/sdxl-turbo"

sample_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
sample_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
auth_key = "Cws5ddVL9CD1UdpWLTK1MYO0LMdcew6B"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        logger.info("start to loading pipeline")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device="cuda")
        self.pipeline = AutoPipelineForImage2Image.from_pipe(self.pipe).to("cuda")
        logger.info("finish to load pipeline")
    
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
            key: str = "",
    ) -> dict:
        if key != auth_key:
            return {
                "code": 403
            }

        image = None
        try:
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        except Exception as err:
            print(err)
            gc.collect()
            torch.cuda.empty_cache()

        if image is None:
            return {
                "code": 500
            }

        return self.formatJPEGResponse(image)

    @bentoml.api
    def img2img(
            self,
            prompt: str = sample_prompt,
            image_url: str = sample_image,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 2,
            guidance_scale: float = 0.0,
            strength: float = 0.5,
            key: str = "",
    ) -> dict:
        if key != auth_key:
            return {
                "code": 403
            }
        
        init_image = None
        if image_url.startswith("http://") or image_url.startswith("https://"):
            init_image = load_image(image_url)
        else:
            imgStr = image_url.replace("data:image/jpeg;base64,", "", 1)
            imgStr = imgStr.replace("data:image/png;base64,", "", 1)
            init_image = Image.open(io.BytesIO(base64.b64decode(imgStr)))
        maxW = init_image.width
        maxH = init_image.height
        if maxW > 960:
            maxW = 960
        if maxH > 960:
            maxH = 960
        ratio = min(maxW/init_image.width, maxH/init_image.height)
        width = int(init_image.width * ratio)
        height = int(init_image.height * ratio)
        logger.info("resize input image to %d x %d with ratio %f", width, height, ratio)
        init_image = init_image.resize((width, height))

        image = None
        try:
            image = self.pipeline(
                prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]
        except Exception as err:
            print(err)
            gc.collect()
            torch.cuda.empty_cache()

        if image is None:
            return {
                "code": 500
            }

        return self.formatJPEGResponse(image)
