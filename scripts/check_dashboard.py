from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"


def main() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    if app.exception:
        for exception in app.exception:
            print(exception)
        raise SystemExit(1)

    print("Dashboard check OK")
    print(f"Tabs: {[tab.label for tab in app.tabs]}")
    print(f"Metrics: {len(app.metric)}")
    print(f"Date filters: {len(app.date_input)}")
    print(f"Multiselect filters: {len(app.multiselect)}")
    print(f"Dataframes: {len(app.dataframe)}")


if __name__ == "__main__":
    main()
