"""Korean stock market dashboard powered by yfinance and Streamlit."""

from datetime import date, timedelta
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import html
import io
import smtplib
import ssl
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from openai import OpenAI


st.set_page_config(page_title="Korea Stock Watch", page_icon="📈", layout="wide")

STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "POSCO홀딩스": "005490.KS",
    "셀트리온": "068270.KS",
}


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Download adjusted daily close prices in one yfinance request."""
    raw = yf.download(
        list(tickers), start=start, end=end + timedelta(days=1),
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.dropna(how="all")


def make_stock_context(prices: pd.DataFrame, summary: pd.DataFrame, start: date, end: date) -> str:
    """Create a compact, data-only context for the chatbot."""
    recent = prices.tail(10).round(0).astype("Int64").reset_index()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    return (
        f"조회 기간: {start} ~ {end}\n\n"
        "최신 요약 (최근 종가 KRW, 기간 수익률 %):\n"
        + summary.to_csv(index=False)
        + "\n최근 10거래일 종가 (KRW):\n"
        + recent.to_csv(index=False)
    )


def build_stock_report(
    summary: pd.DataFrame,
    news: pd.DataFrame,
    disclosures: pd.DataFrame,
    start: date,
    end: date,
) -> tuple[str, str]:
    """Create plain-text and HTML reports from the data shown in the dashboard."""
    best = summary.iloc[0]
    worst = summary.iloc[-1]
    lines = [
        "한국 주식 데이터·뉴스 보고서",
        f"조회 기간: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}",
        "",
        "[시장 요약]",
        f"기간 수익률 상위: {best['종목']} ({best['기간 수익률 (%)']:+.2f}%)",
        f"기간 수익률 하위: {worst['종목']} ({worst['기간 수익률 (%)']:+.2f}%)",
        "",
        "[종목별 종가 및 기간 수익률]",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"- {row[0]}: ₩{row[1]:,.0f} / {row[2]:+.2f}%")
    lines.extend(["", "[수집된 최신 뉴스]"])
    if news.empty:
        lines.append("- 수집된 뉴스가 없습니다.")
    else:
        for item in news.itertuples(index=False):
            source = f" · {item[2]}" if item[2] else ""
            lines.append(f"- [{item[0]}] {item[1]} ({item[3]}{source})\n  {item[4]}")
    lines.extend(["", "[최근 OpenDART 공시]"])
    if disclosures.empty:
        lines.append("- 조회된 공시가 없습니다. (OpenDART API 키 미입력 시 포함)")
    else:
        for item in disclosures.itertuples(index=False):
            lines.append(f"- [{item[0]}] {item[2]} ({item[1]})\n  {item[5]}")
    lines.extend(["", "본 보고서는 수집된 시세·뉴스·공시를 요약한 참고용 정보이며 투자 권유가 아닙니다."])
    text_report = "\n".join(lines)

    summary_rows = "".join(
        "<tr><td>{}</td><td>₩{:,.0f}</td><td>{:+.2f}%</td></tr>".format(
            html.escape(str(row[0])), row[1], row[2]
        )
        for row in summary.itertuples(index=False)
    )
    news_items = "<li>수집된 뉴스가 없습니다.</li>" if news.empty else "".join(
        '<li><strong>[{}]</strong> <a href="{}">{}</a><br><small>{} {}</small></li>'.format(
            html.escape(str(item[0])), html.escape(str(item[4]), quote=True),
            html.escape(str(item[1])), html.escape(str(item[3])), html.escape(str(item[2])),
        )
        for item in news.itertuples(index=False)
    )
    disclosure_items = "<li>조회된 공시가 없습니다.</li>" if disclosures.empty else "".join(
        '<li><strong>[{}]</strong> <a href="{}">{}</a> <small>({})</small></li>'.format(
            html.escape(str(item[0])), html.escape(str(item[5]), quote=True),
            html.escape(str(item[2])), html.escape(str(item[1])),
        )
        for item in disclosures.itertuples(index=False)
    )
    html_report = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;line-height:1.55;color:#202124">
    <h2>한국 주식 데이터·뉴스 보고서</h2><p>조회 기간: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}</p>
    <h3>시장 요약</h3><ul><li>기간 수익률 상위: <b>{html.escape(str(best['종목']))}</b> ({best['기간 수익률 (%)']:+.2f}%)</li>
    <li>기간 수익률 하위: <b>{html.escape(str(worst['종목']))}</b> ({worst['기간 수익률 (%)']:+.2f}%)</li></ul>
    <h3>종목별 종가 및 기간 수익률</h3><table style="border-collapse:collapse"><tr><th>종목</th><th>최근 종가</th><th>기간 수익률</th></tr>{summary_rows}</table>
    <h3>수집된 최신 뉴스</h3><ul>{news_items}</ul><h3>최근 OpenDART 공시</h3><ul>{disclosure_items}</ul>
    <hr><small>본 보고서는 수집된 시세·뉴스·공시를 요약한 참고용 정보이며 투자 권유가 아닙니다.</small></body></html>"""
    return text_report, html_report


def send_smtp_report(
    host: str, port: int, username: str, app_password: str, recipient: str,
    subject: str, text_report: str, html_report: str, use_ssl: bool,
) -> None:
    """Send a multipart report. Credentials remain only in this request's memory."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = username
    message["To"] = recipient
    message.set_content(text_report)
    message.add_alternative(html_report, subtype="html")
    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
            server.login(username, app_password)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(username, app_password)
            server.send_message(message)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_company_news(company_names: tuple[str, ...], limit_per_company: int = 5) -> pd.DataFrame:
    """Fetch recent Korean company news from Google News RSS."""
    rows: list[dict[str, str]] = []
    for company in company_names:
        url = "https://news.google.com/rss/search?q=" + quote_plus(company) + "&hl=ko&gl=KR&ceid=KR:ko"
        try:
            response = requests.get(url, headers={"User-Agent": "KoreaStockWatch/1.0"}, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall("./channel/item")[:limit_per_company]:
                title = html.unescape(item.findtext("title", default="")).strip()
                link = item.findtext("link", default="").strip()
                published = item.findtext("pubDate", default="").strip()
                source_node = item.find("source")
                source = html.unescape(source_node.text or "").strip() if source_node is not None else ""
                try:
                    published_label = parsedate_to_datetime(published).astimezone().strftime("%Y-%m-%d %H:%M")
                except (TypeError, ValueError, OverflowError):
                    published_label = published
                if title and link:
                    rows.append({"기업": company, "제목": title, "출처": source, "게시 시각": published_label, "링크": link})
        except (requests.RequestException, ET.ParseError):
            continue
    return pd.DataFrame(rows, columns=["기업", "제목", "출처", "게시 시각", "링크"])


@st.cache_data(ttl=86400, show_spinner=False)
def load_dart_corp_codes(api_key: str) -> pd.DataFrame:
    """Download and cache the OpenDART corporation-code master."""
    response = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": api_key}, timeout=30,
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        xml_bytes = archive.read("CORPCODE.xml")
    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.findall("list"):
        rows.append({
            "corp_name": item.findtext("corp_name", default="").strip(),
            "corp_code": item.findtext("corp_code", default="").strip(),
            "stock_code": item.findtext("stock_code", default="").strip(),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_dart_disclosures(api_key: str, company_names: tuple[str, ...], days: int = 90) -> pd.DataFrame:
    """Fetch recent filings for selected listed companies from OpenDART."""
    codes = load_dart_corp_codes(api_key)
    codes = codes[codes["stock_code"].str.strip() != ""].copy()
    rows: list[dict[str, str]] = []
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=days)
    dart_aliases = {"NAVER": "네이버", "POSCO홀딩스": "포스코홀딩스"}
    for company in company_names:
        lookup_name = dart_aliases.get(company, company)
        match = codes[codes["corp_name"].str.contains(lookup_name, case=False, na=False)]
        if match.empty:
            continue
        corp_code = match.iloc[0]["corp_code"]
        try:
            response = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params={
                    "crtfc_key": api_key, "corp_code": corp_code,
                    "bgn_de": start_dt.strftime("%Y%m%d"), "end_de": end_dt.strftime("%Y%m%d"),
                    "page_no": 1, "page_count": 10, "sort": "date", "sort_mth": "desc",
                }, timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") not in (None, "000"):
                continue
            for item in payload.get("list", []):
                rcept_no = item.get("rcept_no", "")
                rows.append({
                    "기업": company,
                    "접수일": item.get("rcept_dt", ""),
                    "공시 제목": item.get("report_nm", ""),
                    "제출인": item.get("flr_nm", ""),
                    "공시 유형": item.get("pblntf_ty_nm", ""),
                    "링크": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
                })
        except (requests.RequestException, ValueError):
            continue
    return pd.DataFrame(rows, columns=["기업", "접수일", "공시 제목", "제출인", "공시 유형", "링크"])


def answer_stock_question(
    api_key: str,
    question: str,
    context: str,
    history: list[dict],
    pdf_file_id: str | None = None,
) -> str:
    """Ask GPT-4.1 mini about the currently displayed data."""
    conversation = "\n".join(
        f"{'사용자' if item['role'] == 'user' else '어시스턴트'}: {item['content']}"
        for item in history[-6:]
    )
    instructions = (
        "당신은 한국 주식 데이터 대시보드의 분석 도우미입니다. 반드시 제공된 데이터만 근거로 "
        "한국어로 답하세요. 데이터에 없는 실시간 정보, 뉴스, 재무제표, 미래 가격을 사실처럼 말하지 마세요. "
        "계산이 필요하면 제공된 수치로 계산하고, 불확실하면 데이터에 없다고 밝히세요. "
        "투자 매수·매도 지시를 하지 말고 교육적·정보 제공 목적임을 짧게 덧붙이세요."
    )
    client = OpenAI(api_key=api_key)
    prompt = (
        "현재 화면의 주식 데이터:\n" + context
        + "\n이전 대화:\n" + (conversation or "(없음)")
        + "\n\n사용자 질문: " + question
    )
    input_content = [{"type": "input_text", "text": prompt}]
    if pdf_file_id:
        input_content.insert(0, {"type": "input_file", "file_id": pdf_file_id, "detail": "auto"})
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=[{"role": "user", "content": input_content}],
        store=False,
    )
    return response.output_text


st.title("📈 Korea Stock Watch")
st.caption("yfinance로 수집한 KOSPI 주요 10개 종목의 일별 종가 데이터입니다. 데이터는 15분 동안 캐시됩니다.")

with st.sidebar:
    st.header("조회 설정")
    selected_names = st.multiselect("종목", list(STOCKS), default=list(STOCKS))
    default_start = date.today() - timedelta(days=90)
    period = st.date_input("기간", value=(default_start, date.today()), max_value=date.today())
    if st.button("데이터 새로고침", use_container_width=True):
        load_prices.clear()
    st.divider()
    api_key = st.text_input("OpenAI API 키", type="password", help="키는 이 브라우저 세션에서만 사용되며 파일에 저장하지 않습니다.")
    uploaded_pdf = st.file_uploader("분석할 PDF (선택)", type=["pdf"], help="PDF는 질문 시 OpenAI API로 전달됩니다. 파일당 최대 50MB입니다.")
    dart_api_key = st.text_input("OpenDART API 키", type="password", help="공시 조회용 40자리 인증키입니다. 파일에 저장하지 않습니다.")

if not selected_names:
    st.info("왼쪽에서 하나 이상의 종목을 선택하세요.")
    st.stop()
if not isinstance(period, tuple) or len(period) != 2:
    st.warning("시작일과 종료일을 모두 선택하세요.")
    st.stop()

start_date, end_date = period
if start_date >= end_date:
    st.warning("종료일은 시작일보다 뒤여야 합니다.")
    st.stop()

ticker_to_name = {ticker: name for name, ticker in STOCKS.items()}
tickers = tuple(STOCKS[name] for name in selected_names)

with st.spinner("한국 주식 데이터를 가져오는 중..."):
    prices = load_prices(tickers, start_date, end_date)

if prices.empty:
    st.error("데이터를 가져오지 못했습니다. 네트워크 연결 또는 조회 기간을 확인해 주세요.")
    st.stop()

prices = prices.rename(columns=ticker_to_name)
latest = prices.ffill().iloc[-1]
first = prices.bfill().iloc[0]
change = (latest / first - 1) * 100

summary = pd.DataFrame({
    "종목": latest.index,
    "최근 종가 (KRW)": latest.values,
    "기간 수익률 (%)": change.values,
}).sort_values("기간 수익률 (%)", ascending=False)
summary["최근 종가 (KRW)"] = summary["최근 종가 (KRW)"].round(0).astype("Int64")
summary["기간 수익률 (%)"] = summary["기간 수익률 (%)"].round(2)

cols = st.columns(min(5, len(summary)))
for col, row in zip(cols, summary.itertuples(index=False)):
    col.metric(row[0], f"₩{row[1]:,}", f"{row[2]:+.2f}%")

st.subheader("기간 수익률 비교")
fig_return = px.bar(
    summary, x="종목", y="기간 수익률 (%)", color="기간 수익률 (%)",
    color_continuous_scale=["#D75452", "#F3F4F6", "#128277"],
)
fig_return.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig_return, use_container_width=True)

st.subheader("종가 추이")
chart_data = prices.reset_index().melt(id_vars="Date", var_name="종목", value_name="종가")
fig_price = px.line(chart_data, x="Date", y="종가", color="종목", template="plotly_white")
fig_price.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("📰 기업 관련 뉴스")
st.caption("Google News RSS에서 선택된 기업의 최신 기사를 가져옵니다. 기사를 클릭하면 원문으로 이동합니다.")
news = fetch_company_news(tuple(selected_names))
if news.empty:
    st.info("현재 뉴스를 가져오지 못했습니다. 잠시 후 다시 시도하거나 네트워크 연결을 확인해 주세요.")
else:
    for company in selected_names:
        company_news = news[news["기업"] == company]
        if company_news.empty:
            continue
        with st.expander(f"{company} ({len(company_news)}건)", expanded=company == selected_names[0]):
            for item in company_news.itertuples(index=False):
                source = f" · {item[2]}" if item[2] else ""
                st.markdown(f"- [{item[1]}]({item[4]})  \n  `{item[3]}{source}`")

st.subheader("📑 OpenDART 공시")
if not dart_api_key:
    st.info("공시 정보를 보려면 왼쪽 사이드바에 OpenDART API 키를 입력하세요.")
    disclosures = pd.DataFrame(columns=["기업", "접수일", "공시 제목", "제출인", "공시 유형", "링크"])
else:
    with st.spinner("OpenDART 공시를 조회하는 중..."):
        try:
            disclosures = fetch_dart_disclosures(dart_api_key, tuple(selected_names))
        except (requests.RequestException, zipfile.BadZipFile, ET.ParseError, ValueError) as error:
            disclosures = pd.DataFrame(columns=["기업", "접수일", "공시 제목", "제출인", "공시 유형", "링크"])
            st.error(f"OpenDART 조회에 실패했습니다. API 키와 네트워크를 확인해 주세요: {error}")
    if disclosures.empty:
        st.info("선택한 기간(최근 90일)에 공시가 없거나 회사명을 찾지 못했습니다.")
    else:
        for company in selected_names:
            company_disclosures = disclosures[disclosures["기업"] == company]
            if company_disclosures.empty:
                continue
            with st.expander(f"{company} 공시 ({len(company_disclosures)}건)"):
                st.dataframe(
                    company_disclosures.drop(columns=["기업", "링크"]),
                    use_container_width=True, hide_index=True,
                )
                for item in company_disclosures.itertuples(index=False):
                    st.markdown(f"[{item[2]}]({item[5]}) · {item[1]}")

report_text, report_html = build_stock_report(summary, news, disclosures, start_date, end_date)
st.subheader("📝 데이터·뉴스 보고서")
st.caption("현재 화면에서 수집된 시세, 뉴스와 OpenDART 공시를 하나의 보고서로 만듭니다.")
with st.expander("보고서 미리보기", expanded=False):
    st.text(report_text)
report_filename = f"korea_stock_report_{end_date:%Y%m%d}.txt"
st.download_button("텍스트 보고서 다운로드", report_text.encode("utf-8-sig"), report_filename, "text/plain")

with st.expander("✉️ SMTP로 보고서 메일 보내기", expanded=False):
    st.caption("SMTP 계정과 앱 비밀번호는 이 발송 요청에만 사용되며 앱 파일·세션에 저장하지 않습니다. Gmail은 2단계 인증 후 생성한 앱 비밀번호를 사용하세요.")
    with st.form("smtp_report_form", clear_on_submit=True):
        smtp_host = st.text_input("SMTP 서버", value="smtp.gmail.com", placeholder="예: smtp.gmail.com")
        transport = st.radio("암호화 방식", ["SSL/TLS (권장, 465)", "STARTTLS (587)"], horizontal=True)
        default_port = 465 if transport.startswith("SSL") else 587
        smtp_port = st.number_input("포트", min_value=1, max_value=65535, value=default_port, step=1)
        smtp_username = st.text_input("발신 계정", placeholder="name@example.com")
        smtp_password = st.text_input("앱 비밀번호", type="password")
        recipient = st.text_input("수신자 이메일", placeholder="recipient@example.com")
        mail_subject = st.text_input("메일 제목", value=f"한국 주식 데이터·뉴스 보고서 ({end_date:%Y-%m-%d})")
        send_report = st.form_submit_button("보고서 메일 보내기", use_container_width=True)
    if send_report:
        if not all([smtp_host.strip(), smtp_username.strip(), smtp_password, recipient.strip(), mail_subject.strip()]):
            st.error("SMTP 서버, 발신 계정, 앱 비밀번호, 수신자 이메일, 제목을 모두 입력해 주세요.")
        elif "@" not in smtp_username or "@" not in recipient:
            st.error("발신 계정과 수신자 이메일 주소 형식을 확인해 주세요.")
        else:
            try:
                with st.spinner("SMTP 서버로 보고서를 전송하는 중..."):
                    send_smtp_report(
                        smtp_host.strip(), int(smtp_port), smtp_username.strip(), smtp_password,
                        recipient.strip(), mail_subject.strip(), report_text, report_html,
                        use_ssl=transport.startswith("SSL"),
                    )
                st.success(f"보고서를 {recipient.strip()} 주소로 전송했습니다.")
            except (OSError, smtplib.SMTPException) as error:
                st.error(f"메일 전송에 실패했습니다. SMTP 설정과 앱 비밀번호를 확인해 주세요: {error}")

st.subheader("최신 요약")
st.dataframe(
    summary.style.format({"최근 종가 (KRW)": "₩{:,.0f}", "기간 수익률 (%)": "{:+.2f}%"}),
    use_container_width=True,
    hide_index=True,
)

csv = prices.reset_index().to_csv(index=False).encode("utf-8-sig")
st.download_button("CSV 다운로드", csv, "korean_stock_prices.csv", "text/csv")

st.divider()
st.subheader("💬 주식 데이터 챗봇")
st.caption("현재 화면에 표시된 가격 데이터 기준으로 답변합니다. 모델: gpt-4.1-mini")

pdf_file_id = None
if uploaded_pdf:
    if uploaded_pdf.size > 50 * 1024 * 1024:
        st.error("PDF 파일은 50MB 이하만 업로드할 수 있습니다.")
    elif not api_key:
        st.info("PDF 내용을 질문하려면 먼저 왼쪽 사이드바에 OpenAI API 키를 입력하세요.")
    else:
        pdf_signature = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
        if st.session_state.get("pdf_signature") != pdf_signature:
            try:
                with st.spinner("PDF를 분석할 수 있도록 업로드하는 중..."):
                    client = OpenAI(api_key=api_key)
                    uploaded = client.files.create(
                        file=(uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf"),
                        purpose="user_data",
                    )
                st.session_state.pdf_signature = pdf_signature
                st.session_state.pdf_file_id = uploaded.id
            except Exception as error:
                st.error(f"PDF 업로드에 실패했습니다: {error}")
        pdf_file_id = st.session_state.get("pdf_file_id")
        if pdf_file_id:
            st.success(f"PDF 준비 완료: {uploaded_pdf.name}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("예: 기간 중 가장 많이 오른 종목은 무엇인가요?")
if question:
    if not api_key:
        st.warning("왼쪽 사이드바에 OpenAI API 키를 입력한 뒤 질문해 주세요.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("데이터를 분석하는 중..."):
                try:
                    answer = answer_stock_question(
                        api_key,
                        question,
                        make_stock_context(prices, summary, start_date, end_date)
                        + "\n수집된 기업 뉴스:\n"
                        + (news.drop(columns=["링크"]).to_csv(index=False) if not news.empty else "(없음)")
                        + "\nOpenDART 공시:\n"
                        + (disclosures.drop(columns=["링크"]).to_csv(index=False) if not disclosures.empty else "(API 키 미입력 또는 없음)"),
                        st.session_state.chat_history[:-1],
                        pdf_file_id=pdf_file_id,
                    )
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as error:
                    st.error(f"OpenAI API 호출에 실패했습니다: {error}")

st.caption("투자 참고용 정보이며, 투자 권유가 아닙니다. yfinance 데이터는 거래소 사정에 따라 지연되거나 누락될 수 있습니다.")
