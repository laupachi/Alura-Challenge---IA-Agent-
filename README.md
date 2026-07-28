# Agent-OnlyForGamers
# Agente RAG - Consulta de Documentos con IA

Agente de inteligencia artificial que responde preguntas sobre el contenido de documentos PDF, desarrollado como challenge final del programa **ONE AI for Tech (Grupo 10) - Oracle Next Education**.

## 📋 Descripción general

Esta aplicación permite subir uno o más documentos PDF (por ejemplo, políticas internas, manuales de procesos o documentación técnica de una empresa) y hacerle preguntas en lenguaje natural. El agente busca la información relevante dentro del documento y genera una respuesta clara, basada únicamente en el contenido real del archivo — sin inventar información.

El caso de uso pensado para este proyecto es el de un **analista de negocio** que necesita consultar rápidamente documentación interna (por ejemplo, políticas de la empresa, reportes o documentación de herramientas) sin tener que leer el documento completo cada vez.

## 🏗️ Arquitectura de la solución

El proyecto sigue un patrón **RAG (Retrieval-Augmented Generation)** con tres etapas:

1. **Ingesta y procesamiento del documento**
   - Se extrae el texto del PDF con `pypdf`.
   - El texto se divide en fragmentos de ~500 palabras con superposición de 50 palabras, para no perder contexto en los bordes de cada fragmento.

2. **Búsqueda semántica (Retrieval)**
   - Cada fragmento se convierte en un vector numérico (embedding) usando el modelo de embeddings de Google Generative AI.
   - Cuando el usuario hace una pregunta, esta también se convierte en vector y se compara contra todos los fragmentos usando similitud coseno.
   - Se seleccionan los 2 fragmentos más relevantes para la pregunta.

3. **Generación de la respuesta (Generation)**
   - Los fragmentos relevantes + la pregunta se envían a un modelo Gemini (LLM).
   - El modelo genera la respuesta usando únicamente la información encontrada en el documento.

El flujo del agente está orquestado con **LangGraph**, mediante un grafo de 2 nodos:
START → [nodo_buscar_contexto] → [nodo_generar_respuesta] → END

## 🛠️ Tecnologías y herramientas utilizadas

- **Python 3**
- **Streamlit** — interfaz web y hosting del deploy
- **LangGraph** — orquestación del agente como grafo de estados
- **LangChain** — integración con el modelo de Google
- **Google Generative AI (Gemini)** — modelo de lenguaje (generación) y modelo de embeddings
- **pypdf** — extracción de texto de archivos PDF
- **NumPy** — cálculo de similitud coseno entre vectores

> **Nota sobre el deploy:** el programa recomienda Oracle Cloud Infrastructure (OCI), pero según la aclaración oficial del challenge, el uso de OCI no es obligatorio. Se optó por **Streamlit Community Cloud** como alternativa gratuita y de acceso público, cumpliendo el mismo requisito de tener la aplicación disponible mediante una URL pública.

## ▶️ Instrucciones para ejecutar el proyecto

### Opción 1: Usar la app ya desplegada (recomendado)
Accedé directamente a la URL pública: **https://agentonlyforgamers-5yvqteywthwuxblqfdi9jw.streamlit.app/**
<img width="772" height="546" alt="image" src="https://github.com/user-attachments/assets/d38434dd-1426-4201-9f57-e24a78a2815e" />


### Opción 2: Ejecutar localmente
```bash
# Clonar el repositorio
git clone https://github.com/[tu-usuario]/[tu-repositorio].git
cd [tu-repositorio]

# Instalar dependencias
pip install -r requirements.txt

# Configurar tu API Key de Google Gemini
# Crear archivo .streamlit/secrets.toml con:
# GOOGLE_API_KEY = "tu-clave-aqui"

# Ejecutar la app
streamlit run app.py
```

Podés obtener una API Key gratuita en [Google AI Studio](https://aistudio.google.com/app/apikey).

## 💬 Ejemplos de preguntas que el agente puede responder (dependiendo del documento adjuntado)

- [EJEMPLO 1: Que vende]
- [EJEMPLO 2: Cuales son los metodos de pago]
- [EJEMPLO 3: En cuanto tiempo procesan mi pedido]




