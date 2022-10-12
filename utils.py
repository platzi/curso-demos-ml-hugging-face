import numpy as np
import torch
from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN


## Cargamos el modelo desde el Hub de Hugging Face
def carga_modelo(model_name="ceyda/butterfly_cropped_uniq1K_512", model_version=None):
    gan = LightweightGAN.from_pretrained(model_name, version=model_version)
    gan.eval()
    return gan


## Usamos el modelo GAN para generar imágenes
def genera(gan, batch_size=1):
    with torch.no_grad():
        ims = gan.G(torch.randn(batch_size, gan.latent_dim)).clamp_(0.0, 1.0) * 255
        ims = ims.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    return ims
