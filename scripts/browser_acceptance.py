from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import ConsoleMessage, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_server(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"acceptance server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/api/v1/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("acceptance server did not become healthy")


def assert_layout(page: Page, mobile: bool) -> None:
    overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth")
    assert not overflow, "page has horizontal overflow"
    assert page.locator("#board .square").count() == 64
    assert page.locator('#board .square[tabindex="0"]').count() == 1
    if mobile:
        positions = page.evaluate(
            """() => ['board-pane','result-pane','controls-pane'].map(
              name => document.querySelector('.' + name).getBoundingClientRect().top)"""
        )
        assert positions == sorted(positions), "mobile order must be board, review, controls"
    else:
        positions = page.evaluate(
            """() => ['board-pane','result-pane','controls-pane'].map(
              name => document.querySelector('.' + name).getBoundingClientRect().left)"""
        )
        assert positions == sorted(positions), "desktop order must be board, review, controls"


def session_id(page: Page) -> str:
    value = page.evaluate("JSON.parse(localStorage.getItem('gemmafischer.session.v2')).sessionId")
    assert isinstance(value, str)
    return value


def run_flow(page: Page, url: str) -> None:
    errors: list[str] = []

    def record_console(message: ConsoleMessage) -> None:
        if message.type == "error":
            errors.append(message.text)

    page.on("console", record_console)
    page.goto(url, wait_until="networkidle")
    page.locator("#status").wait_for(state="visible")
    assert_layout(page, mobile=False)
    live_fen = page.locator("#fen").input_value()
    page.locator("#analyze").click()
    page.locator("#practice").wait_for(state="visible", timeout=60_000)
    page.locator("#practice").click()
    page.locator("#practice-banner").wait_for(state="visible")
    page.locator("#tutor-hint-button").click()
    page.locator("#tutor-hint").wait_for(state="visible")
    page.screenshot(
        path=ROOT / "output" / "playwright" / "public-alpha-tutor-active-desktop.png",
        full_page=True,
    )

    page.reload(wait_until="networkidle")
    page.locator("#practice-banner").wait_for(state="visible")
    page.locator("#tutor-hint").wait_for(state="visible")
    assert "restored" in page.locator("#status").inner_text().lower()
    page.locator("#end-practice").click()
    page.locator("#practice-banner").wait_for(state="hidden")

    current_session = session_id(page)
    dismissed = page.request.get(f"{url}/api/v1/sessions/{current_session}/tutor").json()
    assert dismissed["items"][0]["status"] == "dismissed"
    page.locator("#practice").click()
    page.locator("#practice-banner").wait_for(state="visible")

    tutors = page.request.get(f"{url}/api/v1/sessions/{current_session}/tutor").json()
    interaction = tutors["items"][0]
    question_fen = interaction["question"]["fen"]
    source = next(square for square in page.locator("#board .square").evaluate_all(
        "nodes => nodes.map(node => node.dataset.square)"
    ) if page.request.get(
        f"{url}/api/v1/sessions/{current_session}/tutor/{interaction['interaction_id']}"
        f"/legal-moves?from_square={square}"
    ).json()["moves_uci"])
    legal = page.request.get(
        f"{url}/api/v1/sessions/{current_session}/tutor/{interaction['interaction_id']}"
        f"/legal-moves?from_square={source}"
    ).json()["moves_uci"][0]
    page.locator(f'[data-square="{legal[:2]}"]').click()
    page.locator(f'[data-square="{legal[2:4]}"]').click()
    page.locator("#tutor-feedback").wait_for(state="visible", timeout=60_000)
    page.locator("#tutor-follow-up button").first.click()
    page.get_by_role("button", name="Return to game").last.click()
    page.locator("#practice-banner").wait_for(state="hidden")
    assert page.locator("#fen").input_value() == live_fen
    assert page.locator("#practice-banner").is_hidden()
    assert question_fen != ""

    page.set_viewport_size({"width": 390, "height": 844})
    assert_layout(page, mobile=True)
    page.screenshot(
        path=ROOT / "output" / "playwright" / "public-alpha-tutor-mobile.png",
        full_page=True,
    )
    assert not errors, "browser console errors: " + json.dumps(errors)


def main() -> int:
    (ROOT / "output" / "playwright").mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="gemmafischer-browser-") as directory:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "acceptance_server.py"),
                str(port),
                str(Path(directory) / "history.sqlite3"),
            ],
            cwd=ROOT,
        )
        try:
            wait_for_server(url, process)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                run_flow(page, url)
                browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    print(
        "Browser acceptance passed: live analysis, persisted tutor restore/dismiss, "
        "cited practice, desktop/mobile layout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
