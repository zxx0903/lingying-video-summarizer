import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = ROOT / "lingying" / "app.py"
    at = AppTest.from_file(path, default_timeout=180)
    at.run()
    print("initial exceptions:", [str(e) for e in at.exception])
    print("sidebar buttons:", [b.label for b in at.sidebar.button])
    if at.exception:
        sys.exit(1)

    buttons = at.sidebar.button
    run_button = next((button for button in buttons if button.label == "开始生成摘要"), None)
    if run_button is not None:
        run_button.click()
        at.run()
        print("after click exceptions:", [str(e) for e in at.exception])
        print("tabs:", [t.label for t in at.tabs])
        print("metrics:", [(m.label, m.value) for m in at.metric])
        if at.exception or not at.tabs or not at.metric:
            sys.exit(1)


if __name__ == "__main__":
    main()
