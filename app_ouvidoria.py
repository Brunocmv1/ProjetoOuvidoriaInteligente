# app_ouvidoria.py
# Entrega 4 — Buscador Semântico + App Streamlit
# Desafio: Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")
st.title("🏛️ Ouvidoria Inteligente")
st.markdown(
    "Protótipo de triagem semântica de manifestações cidadãs — compara palavras-chave com "
    "**busca vetorial**, capaz de encontrar duplicatas e temas relacionados mesmo com "
    "vocabulário diferente."
)

MODELOS_DISPONIVEIS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-pt-v1.5",
]

# ---------------------------------------------------------------------------
# Sidebar: configurações globais (modelo de embedding + top-k)
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Configurações")
modelo_nome = st.sidebar.selectbox("Modelo de Embedding", MODELOS_DISPONIVEIS)
top_k = st.sidebar.slider("Top-K resultados na busca", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption(
    "🟢 Similaridade > 0.7 · 🟡 > 0.5 · 🔴 abaixo de 0.5"
)


@st.cache_resource
def carregar_modelo(nome):
    return SentenceTransformer(nome)


@st.cache_data
def carregar_manifestacoes():
    with open("manifestacoes.json", encoding="utf-8") as f:
        dados = json.load(f)
    return pd.DataFrame(dados)


@st.cache_data
def gerar_embeddings(_modelo, textos, nome_modelo):
    # nome_modelo entra só pra invalidar o cache quando o modelo mudar
    return _modelo.encode(textos, show_progress_bar=False)


modelo = carregar_modelo(modelo_nome)
df = carregar_manifestacoes()
embeddings = gerar_embeddings(modelo, df["texto"].tolist(), modelo_nome)


def cor_score(score):
    if score > 0.7:
        return "🟢"
    if score > 0.5:
        return "🟡"
    return "🔴"


tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
)

# ---------------------------------------------------------------------------
# Aba 1 — Busca Semântica
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Buscar manifestações semelhantes")
    consulta = st.text_input(
        "Descreva o problema em suas próprias palavras:",
        value="asfalto quebrado com muitos buracos na rua",
    )

    if st.button("🚀 Buscar", key="buscar_semantico"):
        emb_consulta = modelo.encode([consulta])
        sims = cosine_similarity(emb_consulta, embeddings)[0]
        ranking = np.argsort(sims)[::-1][:top_k]

        st.markdown(f"### 🏆 Top {top_k} manifestações mais relevantes")
        for pos, idx in enumerate(ranking, 1):
            score = sims[idx]
            linha = df.iloc[idx]
            with st.expander(
                f"{cor_score(score)} #{pos} — {linha['id']} — score {score:.3f} — "
                f"{linha['texto'][:60]}..."
            ):
                st.write(f"**Categoria oficial:** {linha['categoria_oficial']}")
                st.write(f"**Data:** {linha['data']}")
                st.info(linha["texto"])

# ---------------------------------------------------------------------------
# Aba 2 — Base Completa
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Todas as manifestações")
    st.dataframe(df, use_container_width=True, height=400)

    if st.button("🔢 Gerar Matriz de Similaridade Completa"):
        with st.spinner("Calculando similaridades..."):
            sim_completa = cosine_similarity(embeddings)
            df_sim = pd.DataFrame(sim_completa, index=df["id"], columns=df["id"])

        st.markdown("#### Matriz de Similaridade de Cosseno (todas × todas)")
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(df_sim, cmap="Blues", vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": "Similaridade"})
        st.pyplot(fig)

        limiar = st.slider("Limiar para listar pares duplicados", 0.5, 0.99, 0.85, 0.01,
                            key="limiar_base_completa")
        pares = []
        n = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_completa[i, j] >= limiar:
                    pares.append((df.iloc[i]["id"], df.iloc[j]["id"], round(float(sim_completa[i, j]), 3)))
        pares.sort(key=lambda x: x[2], reverse=True)

        st.markdown(f"#### Pares acima do limiar {limiar:.2f} ({len(pares)} encontrados)")
        if pares:
            st.dataframe(pd.DataFrame(pares, columns=["Manifestação A", "Manifestação B", "Similaridade"]),
                        use_container_width=True)
        else:
            st.info("Nenhum par encontrado acima desse limiar.")

# ---------------------------------------------------------------------------
# Aba 3 — Espaço Vetorial
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Visualização do Espaço Semântico")
    metodo = st.radio("Método de redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)

    if metodo == "PCA":
        coords = PCA(n_components=2).fit_transform(embeddings)
    else:
        perp = min(30, len(df) - 1)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(embeddings)

    categorias = sorted(df["categoria_oficial"].unique())
    paleta = dict(zip(categorias, sns.color_palette("tab10", len(categorias))))

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat in categorias:
        mask = (df["categoria_oficial"] == cat).values
        ax.scatter(coords[mask, 0], coords[mask, 1], label=cat,
                   color=paleta[cat], s=100, edgecolor="black")
    for i, row in df.iterrows():
        ax.annotate(row["id"], (coords[i, 0], coords[i, 1]),
                   textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(f"Manifestações no Espaço Semântico ({metodo}) — coloridas por categoria oficial")
    ax.legend(title="Categoria oficial")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("#### 🧠 Os clusters semânticos coincidem com as categorias oficiais?")
    st.text_area(
        "Escreva aqui sua análise (o que você observa nos agrupamentos acima):",
        placeholder="Ex.: as manifestações de saúde formam um bloco compacto, mas duas "
                    "manifestações de segurança aparecem próximas do bloco de meio ambiente "
                    "porque ambas mencionam 'praça' e 'iluminação'...",
        height=120,
    )

# ---------------------------------------------------------------------------
# Aba 4 — Chunking
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Chunking de Manifestações Longas")
    st.markdown("Cole uma manifestação longa (ou escolha uma da base) para dividir em chunks.")

    usar_da_base = st.checkbox("Usar uma manifestação da base de dados", value=True)
    if usar_da_base:
        longas = df.assign(tamanho=df["texto"].str.len()).sort_values("tamanho", ascending=False)
        opcao = st.selectbox(
            "Manifestação (ordenadas da mais longa para a mais curta):",
            longas["id"] + " — " + longas["tamanho"].astype(str) + " caracteres",
        )
        id_escolhido = opcao.split(" — ")[0]
        texto_alvo = df.loc[df["id"] == id_escolhido, "texto"].iloc[0]
    else:
        texto_alvo = ""

    texto_input = st.text_area("Texto a ser dividido:", value=texto_alvo, height=200)

    col1, col2, col3 = st.columns(3)
    with col1:
        estrategia = st.selectbox("Estratégia", ["RecursiveCharacter", "Fixed-Size (Character)"])
    with col2:
        chunk_size = st.slider("Chunk Size", 100, 1000, 400, 50)
    with col3:
        chunk_overlap = st.slider("Overlap", 0, 300, 50, 10)

    if st.button("✂️ Gerar Chunks"):
        if not texto_input.strip():
            st.warning("Cole ou selecione um texto primeiro.")
        else:
            if estrategia == "RecursiveCharacter":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            else:
                from langchain_text_splitters import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ",
                )

            chunks = splitter.split_text(texto_input)
            st.success(f"{len(chunks)} chunks gerados.")

            for i, c in enumerate(chunks):
                with st.expander(f"Chunk {i + 1} ({len(c)} caracteres)"):
                    st.write(c)

            emb_chunks = modelo.encode(chunks)
            st.write(f"**Formato dos embeddings dos chunks:** {emb_chunks.shape}")

            if len(chunks) > 1:
                coords_chunks = PCA(n_components=2).fit_transform(emb_chunks)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(coords_chunks[:, 0], coords_chunks[:, 1], s=120,
                          c=range(len(chunks)), cmap="viridis", edgecolor="black")
                for i, (x, y) in enumerate(coords_chunks):
                    ax.annotate(f"c{i + 1}", (x, y), textcoords="offset points",
                               xytext=(5, 5), fontsize=9)
                ax.set_title("Chunks desta manifestação no espaço semântico (PCA)")
                st.pyplot(fig)
