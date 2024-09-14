from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Inicjalizacja generatora tekstu
generator = pipeline('text-generation', model='gpt2')

# Generowanie odpowiedzi tekstowej
prompt = "Welcome to the AI-driven card game. What would you like to do next?"
text_response = generator(prompt, max_length=50, num_return_sequences=1)

# Wyświetlanie odpowiedzi AI
print("AI says:", text_response[0]['generated_text'])

# Inicjalizacja generatora obrazów (Stable Diffusion)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Jeśli masz GPU, użyj tej linii; w przeciwnym razie pomiń

# Generowanie obrazu na podstawie promptu
image_prompt = "fantasy card game illustration"
image = pipe(image_prompt).images[0]

# Zapisanie obrazu do pliku
image.save("generated_image.png")
