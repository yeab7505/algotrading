import os
import time
import threading
import logging
from queue import Queue, Empty
from typing import Optional, Callable
import requests


class TelegramReporter:
    """
    Production-oriented Telegram reporter.

    - Non-blocking: messages are enqueued and sent by a background worker
    - Retry with exponential backoff on network errors and 429/5xx responses
    - Small memory footprint and graceful shutdown
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        max_queue_size: int = 512,
        request_timeout_sec: int = 10,
        max_retries: int = 5,
        backoff_base_sec: float = 0.5,
        formatter: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._bot_token = bot_token.strip()
        self._chat_id = chat_id.strip()
        self._timeout = request_timeout_sec
        self._max_retries = max_retries
        self._backoff_base = backoff_base_sec
        self._queue: Queue[str] = Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="TelegramReporterWorker", daemon=True)
        self._session = requests.Session()
        self._formatter = formatter
        self._thread.start()

    @staticmethod
    def from_env() -> Optional["TelegramReporter"]:
        token = os.getenv("8238483052:AAHr4CQqHIcASPgEK6txHyKOI0Kb0_CTBgg")
        chat_id = os.getenv("746595758")
        if token and chat_id:
            return TelegramReporter(token, chat_id)
        return None

    def stop(self, *, drain: bool = True) -> None:
        if drain:
            # Wait briefly for queue to drain, without blocking forever
            deadline = time.time() + 3.0
            while not self._queue.empty() and time.time() < deadline:
                time.sleep(0.05)
        self._stop_event.set()
        # Give worker time to exit
        self._thread.join(timeout=2.0)

    def send(self, message: str, *, parse_mode: str = "HTML", disable_web_page_preview: bool = True) -> None:
        formatted = self._formatter(message) if self._formatter else message
        try:
            self._queue.put_nowait((formatted, parse_mode, disable_web_page_preview))
        except Exception:
            # If queue is full, drop the oldest one to keep latest signals
            try:
                _ = self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait((formatted, parse_mode, disable_web_page_preview))
            except Exception:
                # Give up silently to avoid crashing trading loop
                pass

    # Internal
    def _worker(self) -> None:
        api_url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue

            message, parse_mode, disable_web_page_preview = item
            attempt = 0
            while attempt <= self._max_retries and not self._stop_event.is_set():
                try:
                    resp = self._session.post(
                        api_url,
                        timeout=self._timeout,
                        data={
                            "chat_id": self._chat_id,
                            "text": message,
                            "parse_mode": parse_mode,
                            "disable_web_page_preview": disable_web_page_preview,
                        },
                    )
                    if resp.status_code == 200:
                        break
                    # Handle rate limiting per Telegram docs
                    if resp.status_code == 429:
                        retry_after = resp.json().get("parameters", {}).get("retry_after", 1)
                        time.sleep(float(retry_after))
                    elif 500 <= resp.status_code < 600:
                        time.sleep(self._backoff_base * (2 ** attempt))
                    else:
                        # Client errors - do not retry indefinitely
                        break
                except Exception:
                    time.sleep(self._backoff_base * (2 ** attempt))
                finally:
                    attempt += 1
            self._queue.task_done()


class TelegramLogHandler(logging.Handler):
    """
    Logging handler that forwards WARNING+ records to Telegram via TelegramReporter.
    Keeps formatting compact and avoids sending overly verbose stack traces unless ERROR.
    """

    def __init__(self, reporter: TelegramReporter, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Build concise message
            logger_name = record.name
            level_name = record.levelname
            msg = self.format(record)

            if record.exc_info:
                # Keep exception concise
                message = (
                    f"<b>[{level_name}]</b> <code>{logger_name}</code>\n"
                    f"{msg}"
                )
            else:
                message = f"<b>[{level_name}]</b> <code>{logger_name}</code> — {msg}"

            self._reporter.send(message)
        except Exception:
            # Never raise from emit
            pass


def setup_telegram_logging_from_env(default_level: int = logging.WARNING) -> Optional[TelegramReporter]:
    """
    Helper to create a TelegramReporter and attach a TelegramLogHandler to the root logger
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set. Returns the reporter instance if active.
    """
    reporter = TelegramReporter.from_env()
    if not reporter:
        return None

    handler = TelegramLogHandler(reporter, level=default_level)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    return reporter


