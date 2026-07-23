import hashlib
import os
from datetime import date, timedelta
from urllib.parse import quote_plus

import chromadb
import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from plotly.subplots import make_subplots

import config

st.set_page_config(page_title="주가 분석 AI", layout="wide")


def huggingface_api_key() -> str | None:
    """Read a key from Streamlit secrets first, then local environment variables."""
    try:
        secrets_key = st.secrets.get("HUGGINGFACE_API_KEY")
    except Exception:
        secrets_key = None
    return secrets_key or config.HUGGINGFACE_API_KEY


@st.cache_resource
def huggingface_client(key: str) -> InferenceClient:
    return InferenceClient(token=key)


@st.cache_resource
def chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=config.CHROMA_PATH)


@st.cache_data(ttl="1h", max_entries=50, show_spinner=False)
def load_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl="1h", max_entries=50, show_spinner=False)
def load_financial_summary(ticker: str) -> dict[str, str]:
    info = yf.Ticker(ticker).info

    def number(value: object, suffix: str = "", multiplier: float = 1) -> str:
        return f"{value * multiplier:,.2f}{suffix}" if isinstance(value, (int, float)) else "N/A"

    def market_cap(value: object) -> str:
        if not isinstance(value, (int, float)):
            return "N/A"
        for divisor, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if value >= divisor:
                return f"{value / divisor:,.2f}{unit}"
        return f"{value:,.0f}"

    return {
        "종목명": str(info.get("longName", "N/A")),
        "섹터": str(info.get("sector", "N/A")),
        "산업": str(info.get("industry", "N/A")),
        "시가총액": market_cap(info.get("marketCap")),
        "PER": number(info.get("trailingPE")),
        "EPS": number(info.get("trailingEps")),
        "배당수익률": number(info.get("dividendYield"), "%", 100),
        "52주 최고가": number(info.get("fiftyTwoWeekHigh")),
        "52주 최저가": number(info.get("fiftyTwoWeekLow")),
    }


@st.cache_data(ttl="30m", max_entries=50, show_spinner=False)
def load_news(ticker: str) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    queries = ((f"{ticker} 주가", "ko", "ko", "KR"), (f"{ticker} stock", "en", "en-US", "US"))
    for query, language, locale, country in queries:
        url = (
            "https://news.google.com/rss/search?q="
            f"{quote_plus(query)}&hl={locale}&gl={country}&ceid={country}:{language}"
        )
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        for entry in feed.entries[:10]:
            documents.append(
                {
                    "text": f"제목: {entry.get('title', '')}\n요약: {entry.get('summary', '')}",
                    "source": str(entry.get("link", "")),
                    "published_date": str(entry.get("published", "")),
                    "type": "news",
                }
            )
    return documents


@st.cache_data(ttl="24h", max_entries=50, show_spinner=False)
def load_sec_filings(ticker: str) -> list[dict[str, str]]:
    """Fetch recent SEC filing metadata using the official submissions endpoint."""
    headers = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    tickers = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers,
        timeout=config.REQUEST_TIMEOUT,
    )
    tickers.raise_for_status()
    match = next(
        (item for item in tickers.json().values() if item["ticker"].upper() == ticker.upper()),
        None,
    )
    if match is None:
        return []

    cik = str(match["cik_str"]).zfill(10)
    submissions = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=headers,
        timeout=config.REQUEST_TIMEOUT,
    )
    submissions.raise_for_status()
    recent = submissions.json().get("filings", {}).get("recent", {})
    documents: list[dict[str, str]] = []
    for form, filed, report_date, accession, primary_document in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form not in {"10-K", "10-Q", "8-K"}:
            continue
        accession_number = accession.replace("-", "")
        source = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}"
        documents.append(
            {
                "text": f"SEC 공시: {form}\n제출일: {filed}\n보고 기준일: {report_date}",
                "source": source,
                "published_date": filed,
                "type": "filing",
            }
        )
        if len(documents) == 5:
            break
    return documents


def collection_name(ticker: str) -> str:
    digest = hashlib.sha256(ticker.upper().encode()).hexdigest()[:16]
    return f"ticker_{digest}"


def rebuild_vector_store(ticker: str, documents: list[dict[str, str]], key: str) -> int:
    if not documents:
        return 0
    client = chroma_client()
    name = collection_name(ticker)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"ticker": ticker.upper()})
    llm_client = huggingface_client(key)
    texts = [document["text"] for document in documents]
    embeddings = llm_client.feature_extraction(model=config.HUGGINGFACE_EMBEDDING_MODEL, text=texts)
    collection.add(
        ids=[f"{ticker.upper()}-{index}" for index in range(len(documents))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": item["source"], "published_date": item["published_date"], "type": item["type"]}
            for item in documents
        ],
    )
    return len(documents)


def answer_question(ticker: str, question: str, key: str) -> tuple[str, list[dict[str, str]]]:
    collection = chroma_client().get_collection(collection_name(ticker))
    llm_client = huggingface_client(key)
    question_embedding = llm_client.feature_extraction(
        model=config.HUGGINGFACE_EMBEDDING_MODEL,
        text=question,
    ).tolist()
    result = collection.query(query_embeddings=question_embedding, n_results=5)
    texts = result.get("documents", [[]])[0]
    metadata = result.get("metadatas", [[]])[0]
    sources = [{"text": text, **item} for text, item in zip(texts, metadata)]
    context = "\n\n".join(f"[{index + 1}] {text}" for index, text in enumerate(texts))
    base_prompt = (
        "당신은 금융 분석 보조자입니다. 제공된 문맥에만 근거해 한국어로 답하세요. "
        "투자 조언으로 단정하지 말고, 문맥이 부족하면 부족하다고 명시하세요.\n\n"
        f"문맥:\n{context}\n\n질문: {question}"
    )
    # Gemma instruction format
    prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
    response = llm_client.text_generation(
        model=config.HUGGINGFACE_CHAT_MODEL,
        prompt=prompt,
        max_new_tokens=1024,
        temperature=0.1,
    )
    return response or "응답을 생성하지 못했습니다.", sources


def stock_figure(data: pd.DataFrame, ticker: str) -> go.Figure:
    chart = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )
    processed = data.copy()
    processed["MA20"] = processed["Close"].rolling(20).mean()
    processed["MA60"] = processed["Close"].rolling(60).mean()
    chart.add_trace(
        go.Candlestick(
            x=processed.index,
            open=processed["Open"],
            high=processed["High"],
            low=processed["Low"],
            close=processed["Close"],
            name="주가",
        ),
        row=1,
        col=1,
    )
    chart.add_trace(
        go.Scatter(x=processed.index, y=processed["MA20"], name="MA20", line={"color": "orange"}),
        row=1,
        col=1,
    )
    chart.add_trace(
        go.Scatter(x=processed.index, y=processed["MA60"], name="MA60", line={"color": "purple"}),
        row=1,
        col=1,
    )
    chart.add_trace(
        go.Bar(x=processed.index, y=processed["Volume"], name="거래량", marker_color="gray"),
        row=2,
        col=1,
    )
    chart.update_layout(title=f"{ticker} 주가 분석", xaxis_rangeslider_visible=False, height=650)
    return chart


for state_key, default in {
    "ticker": "AAPL",
    "loaded_ticker": None,
    "analysis_result": None,
    "analysis_sources": [],
}.items():
    st.session_state.setdefault(state_key, default)

key = huggingface_api_key()
st.title("주가 분석 AI")
st.caption("시장 데이터, 뉴스, SEC 공시 메타데이터를 바탕으로 분석합니다.")
if not key:
    st.error("HUGGINGFACE_API_KEY를 Streamlit secrets 또는 환경 변수에 설정하세요.")
    st.stop()

with st.sidebar:
    st.header("분석 설정")
    with st.form("settings", border=False):
        ticker = st.text_input(
            "종목 코드",
            value=st.session_state.ticker,
            help="예: AAPL, GOOGL, 005930.KS",
        ).strip().upper()
        start_date = st.date_input("시작일", value=date.today() - timedelta(days=365))
        end_date = st.date_input("종료일", value=date.today())
        prepare = st.form_submit_button("데이터 로드 및 분석 준비", type="primary", width="stretch")

if prepare:
    if not ticker or start_date > end_date:
        st.sidebar.error("유효한 종목 코드와 날짜 범위를 입력하세요.")
    else:
        if ticker != st.session_state.ticker:
            st.session_state.analysis_result = None
            st.session_state.analysis_sources = []
            st.session_state.loaded_ticker = None
        st.session_state.ticker = ticker
        try:
            with st.spinner("뉴스와 공시를 수집하고 분석 인덱스를 준비하는 중입니다..."):
                documents = load_news(ticker) + load_sec_filings(ticker)
                count = rebuild_vector_store(ticker, documents, key)
            st.session_state.loaded_ticker = ticker if count else None
            st.sidebar.success(f"{count}개 문서를 준비했습니다.")
        except requests.RequestException as error:
            st.sidebar.error(f"데이터 수집에 실패했습니다: {error}")
        except (Exception, HfHubHTTPError) as error:
            st.sidebar.error(f"분석 준비에 실패했습니다: {error}")

view = st.segmented_control(
    "화면",
    ["AI 분석", "주가 차트", "원본 데이터"],
    default="AI 분석",
    required=True,
    width="stretch",
)

if view == "AI 분석":
    st.header("AI 기반 종합 분석")
    with st.form("question", border=False):
        question = st.text_input(
            "질문",
            placeholder="예: 최근 공시에서 확인된 주요 위험은 무엇인가요?",
        )
        ask = st.form_submit_button("분석 요청", width="content")
    if ask:
        if st.session_state.loaded_ticker != st.session_state.ticker:
            st.warning("먼저 사이드바에서 현재 종목의 데이터를 준비하세요.")
        elif not question.strip():
            st.warning("질문을 입력하세요.")
        else:
            try:
                with st.spinner("관련 문서를 검색하고 답변을 생성하는 중입니다..."):
                    answer, sources = answer_question(st.session_state.ticker, question, key)
                st.session_state.analysis_result = answer
                st.session_state.analysis_sources = sources
            except (Exception, HfHubHTTPError) as error:
                st.error(f"AI 분석에 실패했습니다: {error}")
    if st.session_state.analysis_result:
        st.subheader("분석 결과")
        st.markdown(st.session_state.analysis_result)
        if st.session_state.analysis_sources:
            with st.expander("참고 데이터 소스"):
                for index, source in enumerate(st.session_state.analysis_sources):
                    st.caption(source["text"])
                    st.link_button(
                        "원문 열기",
                        source["source"],
                        icon=":material/open_in_new:",
                        key=f"source-link-{index}",
                    )

elif view == "주가 차트":
    st.header("주가 차트 및 주요 지표")
    try:
        with st.spinner("시장 데이터를 불러오는 중입니다..."):
            stock_data = load_stock_data(st.session_state.ticker, start_date, end_date)
            summary = load_financial_summary(st.session_state.ticker)
        if stock_data.empty:
            st.warning("해당 조건의 주가 데이터가 없습니다.")
        else:
            for column, (label, value) in zip(st.columns(4), summary.items()):
                column.metric(label, value)
            st.plotly_chart(stock_figure(stock_data, st.session_state.ticker), width="stretch")
    except Exception as error:
        st.error(f"시장 데이터를 불러오지 못했습니다: {error}")

else:
    st.header("수집된 원본 데이터")
    if st.session_state.loaded_ticker != st.session_state.ticker:
        st.info("먼저 사이드바에서 현재 종목의 데이터를 준비하세요.")
    else:
        try:
            documents = load_news(st.session_state.ticker) + load_sec_filings(st.session_state.ticker)
            table = pd.DataFrame(documents).drop(columns="text")
            st.dataframe(
                table,
                column_config={"source": st.column_config.LinkColumn("원문")},
                width="stretch",
            )
        except Exception as error:
            st.error(f"원본 데이터를 불러오지 못했습니다: {error}")
