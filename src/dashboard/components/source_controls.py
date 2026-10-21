"""Common source selection and date controls."""

from datetime import UTC, date, datetime, time, timedelta
import streamlit as st

def _utc_date(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _tickers(text: str, provider: str = "yahoo") -> list[str]:
    items = text.replace(",", " ").split()
    return list(dict.fromkeys(item.upper() if provider == "yahoo" else item for item in items))


def _provider_choice(names, *, key, result_key=None, memory=None):
    names = sorted(set(names))
    if not names:
        st.info("No providers are available here. Fetching requires a registered provider; browsing and backtesting require local datasets.")
        return None
    provider = (memory.widget(st.selectbox, "Provider", options=names, key=key) if memory else
                st.selectbox("Provider", names, key=key))
    if result_key and st.session_state.get(f"{key}-previous") != provider:
        st.session_state.pop(result_key, None)
    st.session_state[f"{key}-previous"] = provider
    return provider


def _date_fields(prefix: str, *, first: date | None = None, last: date | None = None,
                 intraday: bool = False, memory=None):
    left, right = st.columns(2)
    first = first or date.today() - timedelta(days=30)
    last = last or date.today()
    if memory:
        start = memory.widget(left.date_input, "Start date (UTC, inclusive)", value=first, key=f"{prefix}-start")
        end = memory.widget(right.date_input, "End date (UTC, exclusive)", value=last, key=f"{prefix}-end")
    else:
        start = left.date_input("Start date (UTC, inclusive)", first, key=f"{prefix}-start")
        end = right.date_input("End date (UTC, exclusive)", last, key=f"{prefix}-end")
    if intraday:
        start_time = (memory.widget(left.time_input, "Start time (UTC)", value=time.min, step=60, key=f"{prefix}-start-time") if memory else
                      left.time_input("Start time (UTC)", time.min, step=60, key=f"{prefix}-start-time"))
        end_time = (memory.widget(right.time_input, "End time (UTC)", value=time.min, step=60, key=f"{prefix}-end-time") if memory else
                    right.time_input("End time (UTC)", time.min, step=60, key=f"{prefix}-end-time"))
        return datetime.combine(start, start_time, UTC), datetime.combine(end, end_time, UTC)
    return _utc_date(start), _utc_date(end)


