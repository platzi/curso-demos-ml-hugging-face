import streamlit as st

from utils import carga_modelo, genera

## Página principal
st.title("Butterfly GAN (GAN de mariposas)")
st.write(
    "Modelo Light-GAN entrenado con 1000 imágenes de mariposas tomadas de la colección del Museo Smithsonian."
)

## Barra lateral
st.sidebar.subheader("¡Esta mariposa no existe! Ni en América Latina 🤯.")
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.caption(
    f"[Modelo](https://huggingface.co/ceyda/butterfly_cropped_uniq1K_512) y [Dataset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) usados."
)
st.sidebar.caption(f"*Disclaimers:*")
st.sidebar.caption(
    "* Este demo es una versión simplificada del creado por [Ceyda Cinarel](https://github.com/cceyda) y [Jonathan Whitaker](https://datasciencecastnet.home.blog/) ([link](https://huggingface.co/spaces/huggan/butterfly-gan)) durante el hackathon [HugGan](https://github.com/huggingface/community-events). Cualquier error se atribuye a [Omar Espejel](https://twitter.com/espejelomar)."
)
st.sidebar.caption(
    "* Modelo basado en el [paper](https://openreview.net/forum?id=1Fqg133qRaI) *Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis*."
)

## Cargamos modelo
repo_id = "ceyda/butterfly_cropped_uniq1K_512"
version_modelo = "57d36a15546909557d9f967f47713236c8288838"
modelo_gan = carga_modelo(repo_id, version_modelo)

## Generamos 4 mariposas
n_mariposas = 4

## Función que genera mariposas y lo guarda como un estado de la sesión
def corre():
    with st.spinner("Generando, espera un poco..."):
        ims = genera(modelo_gan, n_mariposas)
        st.session_state["ims"] = ims


## Si no hay una imagen generada entonces generala
if "ims" not in st.session_state:
    st.session_state["ims"] = None
    corre()

## ims contiene las imágenes generadas
ims = st.session_state["ims"]

## Si la usuaria da click en el botón entonces corremos la función genera()
corre_boton = st.button(
    "Genera mariposas, porfa.",
    on_click=corre,
    help="Estamos en pleno vuelo, puede tardar.",
)

if ims is not None:
    cols = st.columns(n_mariposas)
    for j, im in enumerate(ims):
        i = j % n_mariposas
        cols[i].image(im, use_column_width=True)
