import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import numpy as np

# ---------- CONFIGURACIÓN DE LA PÁGINA ----------
st.set_page_config(page_title="Agente RAG - Documentos", page_icon="📄")
st.title("📄 Agente de IA sobre tus documentos")
st.write("Subí uno o más PDFs y hacele preguntas sobre su contenido.")

# ---------- CONECTAR CON GOOGLE GEMINI ----------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Busca automáticamente un modelo de EMBEDDINGS disponible
@st.cache_resource
def obtener_modelo_embedding():
    available_models = genai.list_models()
    for m in available_models:
        if 'embedContent' in m.supported_generation_methods and 'embedding' in m.name:
            return m.name
    return None

# Busca automáticamente un modelo de GENERACIÓN disponible (con orden de preferencia)
@st.cache_resource
def obtener_modelo_generacion():
    available_models = genai.list_models()
    preferred_models = ['gemini-flash-latest', 'gemini-pro-latest', 'gemini-1.5-flash', 'gemini-pro']

    for preferred_name in preferred_models:
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods and preferred_name in m.name:
                return m.name

    # Si ninguno de los preferidos está disponible, usa cualquier modelo Gemini que sirva
    for m in available_models:
        if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
            return m.name
    return None

EMBEDDING_MODEL_NAME = obtener_modelo_embedding()
GENERATION_MODEL_NAME = obtener_modelo_generacion()

if not EMBEDDING_MODEL_NAME or not GENERATION_MODEL_NAME:
    st.error("⚠️ No se encontró un modelo de Gemini disponible. Revisá tu API Key en Secrets.")
    st.stop()

# ---------- FUNCIONES DEL PASO 1: LEER Y PROCESAR PDFs ----------
def extraer_texto_pdf(archivo_pdf):
    lector = PdfReader(archivo_pdf)
    texto_completo = ""
    for pagina in lector.pages:
        texto_pagina = pagina.extract_text()
        if texto_pagina:
            texto_completo += texto_pagina + "\n"
    return texto_completo

def dividir_en_fragmentos(texto, tamano_fragmento=500, superposicion=50):
    palabras = texto.split()
    fragmentos = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + tamano_fragmento
        fragmento = " ".join(palabras[inicio:fin])
        fragmentos.append(fragmento)
        inicio += tamano_fragmento - superposicion
    return fragmentos

# ---------- SUBIR Y PROCESAR PDFs ----------
archivos_subidos = st.file_uploader(
    "Subí tus PDFs", type="pdf", accept_multiple_files=True
)

if archivos_subidos and st.button("Procesar documentos"):
    with st.spinner("Leyendo y procesando tus documentos..."):
        documentos_procesados = []
        for archivo in archivos_subidos:
            texto = extraer_texto_pdf(archivo)
            fragmentos = dividir_en_fragmentos(texto)
            for i, fragmento in enumerate(fragmentos):
                documentos_procesados.append({
                    "archivo_origen": archivo.name,
                    "fragmento_numero": i,
                    "texto": fragmento
                })

        embeddings_modelo = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        textos = [doc["texto"] for doc in documentos_procesados]
        vectores_fragmentos = embeddings_modelo.embed_documents(textos)

        st.session_state["documentos_procesados"] = documentos_procesados
        st.session_state["vectores_fragmentos"] = vectores_fragmentos
        st.session_state["embeddings_modelo"] = embeddings_modelo

    st.success(f"✅ Listo. Se generaron {len(documentos_procesados)} fragmentos de {len(archivos_subidos)} documento(s).")

# ---------- FUNCIONES DEL PASO 2: EL AGENTE ----------
def buscar_fragmentos_relevantes(pregunta, top_k=2):
    embeddings_modelo = st.session_state["embeddings_modelo"]
    documentos_procesados = st.session_state["documentos_procesados"]
    vectores_fragmentos = st.session_state["vectores_fragmentos"]

    vector_pregunta = embeddings_modelo.embed_query(pregunta)
    similitudes = []
    for i, vector_fragmento in enumerate(vectores_fragmentos):
        similitud = np.dot(vector_pregunta, vector_fragmento) / (
            np.linalg.norm(vector_pregunta) * np.linalg.norm(vector_fragmento)
        )
        similitudes.append((similitud, i))
    similitudes.sort(reverse=True)

    return [documentos_procesados[i] for _, i in similitudes[:top_k]]

class EstadoAgente(TypedDict):
    pregunta: str
    fragmentos_encontrados: List[dict]
    respuesta: str

def nodo_buscar_contexto(estado: EstadoAgente) -> EstadoAgente:
    estado["fragmentos_encontrados"] = buscar_fragmentos_relevantes(estado["pregunta"])
    return estado

def nodo_generar_respuesta(estado: EstadoAgente) -> EstadoAgente:
    modelo_llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL_NAME, temperature=0)
    contexto = "\n\n".join([
        f"[Fuente: {frag['archivo_origen']}]\n{frag['texto']}"
        for frag in estado["fragmentos_encontrados"]
    ])
    prompt = f"""Respondé la siguiente pregunta usando ÚNICAMENTE la información del contexto de abajo.
Si la respuesta no está en el contexto, decí claramente que no la encontraste en los documentos.

CONTEXTO:
{contexto}

PREGUNTA: {estado['pregunta']}

RESPUESTA:"""
    respuesta = modelo_llm.invoke(prompt)
    estado["respuesta"] = respuesta.content
    return estado

@st.cache_resource
def construir_agente():
    grafo = StateGraph(EstadoAgente)
    grafo.add_node("buscar_contexto", nodo_buscar_contexto)
    grafo.add_node("generar_respuesta", nodo_generar_respuesta)
    grafo.add_edge(START, "buscar_contexto")
    grafo.add_edge("buscar_contexto", "generar_respuesta")
    grafo.add_edge("generar_respuesta", END)
    return grafo.compile()

# ---------- INTERFAZ DE CHAT ----------
if "documentos_procesados" in st.session_state:
    st.divider()
    st.subheader("💬 Preguntale al agente")
    pregunta = st.text_input("Escribí tu pregunta sobre los documentos:")

    if pregunta:
        agente = construir_agente()
        with st.spinner("Pensando..."):
            resultado = agente.invoke({
                "pregunta": pregunta,
                "fragmentos_encontrados": [],
                "respuesta": ""
            })
        st.write("**Respuesta:**")
        st.write(resultado["respuesta"])

        with st.expander("Ver fuentes usadas"):
            for frag in resultado["fragmentos_encontrados"]:
                st.caption(f"📄 {frag['archivo_origen']} (fragmento {frag['fragmento_numero']})")
else:
    st.info("👆 Subí tus PDFs y hacé clic en 'Procesar documentos' para empezar.")
