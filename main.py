# Choose model: 'stable_diffusion' or 'dalle_mini'
MODEL = "stable_diffusion"  # or "dalle_mini"
PROMPT = "A futuristic cityscape in watercolor"
NUM_IMAGES = 3

def generate_sd(prompt, num_images):
    import keras_cv
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
    return model.text_to_image(prompt, batch_size=num_images)

def generate_dalle(prompt, num_images):
    from dalle_mini import DalleMini
    model = DalleMini()
    return model.generate_images(prompt, num_images=num_images)

if MODEL == "stable_diffusion":
    images = generate_sd(PROMPT, NUM_IMAGES)
elif MODEL == "dalle_mini":
    images = generate_dalle(PROMPT, NUM_IMAGES)
else:
    raise ValueError("Unknown model")
    
from utils import plot_images
plot_images(images)
