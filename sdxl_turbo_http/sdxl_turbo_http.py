from diffusers import AutoPipelineForText2Image
import torch, time
from fastapi import FastAPI
from fastapi.responses import Response
from io import BytesIO
from PIL import Image


pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

app = FastAPI()

@app.get("/process/")
async def process(prompt: str, negative: str = None):
    negative = negative if negative is not None else "cartoon, oil, (worst quality, bad quality:1.2)(easynegative), (worst quality:2), (low quality:2), (normal quality:2),watermark, signature, lowres, ((monochrome)), ((grayscale)), cropped, signature, watermark, framed, border, grain, dust, film grain"
    # You can replace this with your own logic to generate a binary image.
    # Here, we are creating a simple binary image.
    img = pipe(prompt=prompt, negative=negative, num_inference_steps=1, guidance_scale=0.0).images[0]
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Return the binary image as a response with appropriate content type.
    return Response(content=img_bytes.read(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

