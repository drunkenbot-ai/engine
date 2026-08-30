from __future__ import annotations

import json
import logging
import smtplib
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional


LOGGER = logging.getLogger(__name__)
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_TELEGRAM_TEXT = 3900


@dataclass
class TelegramNotifierConfig:
    """Telegram Bot API settings for progress notifications."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class EmailNotifierConfig:
    """SMTP settings for email progress notifications."""

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""
    to_email: str = ""
    use_tls: bool = True


@dataclass
class NotifierConfig:
    """Notification manager settings stored in notifier_config.json."""

    progress_interval_seconds: int = 60
    telegram: TelegramNotifierConfig = field(default_factory=TelegramNotifierConfig)
    email: EmailNotifierConfig = field(default_factory=EmailNotifierConfig)


@dataclass(frozen=True)
class _PendingNotification:
    kind: str
    stage_key: str
    subject: str
    text: str


def default_notifier_config_path(project_dir: Optional[Path] = None) -> Path:
    """Return the notifier config path for a project or user profile.

    Args:
        project_dir: Optional project root directory.

    Returns:
        Path where notifier_config.json should live.
    """

    if project_dir is not None:
        return project_dir / "notifier_config.json"
    return Path.home() / ".drunkenbot_ide" / "notifier_config.json"


def ensure_notifier_config(path: Path) -> Path:
    """Create notifier_config.json with disabled defaults when missing.

    Args:
        path: Desired configuration file path.

    Returns:
        The same configuration path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(asdict(NotifierConfig()), indent=2), encoding="utf-8")
        LOGGER.info("Created notifier config: %s", path)
    return path


def load_notifier_config(path: Path) -> NotifierConfig:
    """Load notifier settings from JSON.

    Args:
        path: Configuration file path.

    Returns:
        Parsed notifier configuration.
    """

    ensure_notifier_config(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        LOGGER.error("Notifier config is invalid JSON: %s", exc)
        return NotifierConfig()
    telegram = TelegramNotifierConfig(**dict(raw.get("telegram", {})))
    email = EmailNotifierConfig(**dict(raw.get("email", {})))
    return NotifierConfig(
        progress_interval_seconds=int(raw.get("progress_interval_seconds", 60) or 60),
        telegram=telegram,
        email=email,
    )


class NotificationManager:
    """Send throttled progress and completion notifications.

    Telegram progress notifications are edited in-place when possible. Email
    notifications are sent as periodic snapshots because email messages cannot
    be edited after delivery.
    """

    def __init__(
        self,
        config_path: Path,
        max_pending: int = 32,
        config_check_interval_seconds: float = 2.0,
        minimum_progress_interval_seconds: float = 10.0,
    ) -> None:
        """Initialize the notification manager.

        Args:
            config_path: Path to notifier_config.json.
        """

        self.config_path = ensure_notifier_config(config_path)
        self.config = load_notifier_config(self.config_path)
        self._last_progress_at: dict[str, float] = {}
        self._telegram_message_ids: dict[str, int] = {}
        self._condition = threading.Condition(threading.RLock())
        self._delivery_lock = threading.RLock()
        self._pending_progress: OrderedDict[str, _PendingNotification] = OrderedDict()
        self._terminal_notifications: deque[_PendingNotification] = deque()
        self._max_pending = max(1, int(max_pending))
        self._config_check_interval_seconds = max(0.0, float(config_check_interval_seconds))
        self._minimum_progress_interval_seconds = max(
            0.0,
            float(minimum_progress_interval_seconds),
        )
        self._last_config_check_at = time.monotonic()
        self._config_mtime_ns = self._read_config_mtime()
        self._accepting = True
        self._disabled_warning_logged = False
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="training-notifier",
            daemon=True,
        )
        self._dispatcher.start()

    @property
    def enabled(self) -> bool:
        """Whether any notification channel is configured and enabled."""

        return self.telegram_enabled or self.email_enabled

    @property
    def telegram_enabled(self) -> bool:
        """Whether Telegram notifications can be sent."""

        telegram = self.config.telegram
        return telegram.enabled and bool(telegram.bot_token.strip()) and bool(telegram.chat_id.strip())

    @property
    def email_enabled(self) -> bool:
        """Whether email notifications can be sent."""

        email = self.config.email
        return (
            email.enabled
            and bool(email.smtp_host.strip())
            and bool(email.from_email.strip())
            and bool(email.to_email.strip())
        )

    def reload(self) -> None:
        """Reload notifier_config.json from disk."""

        with self._delivery_lock:
            self.config = load_notifier_config(self.config_path)
            self._config_mtime_ns = self._read_config_mtime()
            self._last_config_check_at = time.monotonic()
            self._disabled_warning_logged = False

    def notify_progress(self, stage_key: str, title: str, lines: list[str], percent: Optional[int] = None) -> None:
        """Send or edit a throttled progress notification.

        Args:
            stage_key: Stable task key such as dataset, training, or fine_tune.
            title: User-facing notification title.
            lines: Body lines.
            percent: Optional progress percent.
        """

        self._reload_if_changed()
        if not self.enabled:
            self._log_disabled()
            return
        text = self._format_message(title, lines, percent)
        notification = _PendingNotification("progress", stage_key, f"{title} progress", text)
        with self._condition:
            if not self._accepting:
                return
            if stage_key in self._pending_progress:
                self._pending_progress[stage_key] = notification
                self._pending_progress.move_to_end(stage_key)
            elif self._pending_count_locked() < self._max_pending:
                self._pending_progress[stage_key] = notification
            elif self._pending_progress:
                self._pending_progress.popitem(last=False)
                self._pending_progress[stage_key] = notification
            self._condition.notify()

    def notify_complete(self, stage_key: str, title: str, lines: list[str]) -> None:
        """Send a completion summary immediately.

        Args:
            stage_key: Stable task key such as dataset, training, or fine_tune.
            title: User-facing notification title.
            lines: Body lines.
        """

        self._reload_if_changed()
        if not self.enabled:
            self._log_disabled()
            return
        text = self._format_message(title, lines, 100)
        self._enqueue_terminal(_PendingNotification("complete", stage_key, title, text))

    def notify_failure(self, stage_key: str, title: str, message: str) -> None:
        """Send a failure summary immediately.

        Args:
            stage_key: Stable task key such as dataset, training, or fine_tune.
            title: User-facing notification title.
            message: Failure message.
        """

        self._reload_if_changed()
        if not self.enabled:
            self._log_disabled()
            return
        text = self._format_message(title, [message], None)
        self._enqueue_terminal(_PendingNotification("failure", stage_key, title, text))

    def close(self, timeout: Optional[float] = 5.0) -> None:
        """Stop accepting work and wait for queued notifications to finish."""

        with self._condition:
            self._accepting = False
            self._condition.notify_all()
        self._dispatcher.join(timeout=timeout)

    @property
    def pending_count(self) -> int:
        """Return queued progress and terminal notification count."""

        with self._condition:
            return self._pending_count_locked()

    def _send_progress(self, stage_key: str, subject: str, text: str) -> None:
        """Dispatch a progress notification to enabled channels."""

        if self.telegram_enabled:
            self._send_or_edit_telegram(stage_key, text)
        if self.email_enabled:
            self._send_email(subject, text)

    def _send_completion(self, stage_key: str, subject: str, text: str) -> None:
        """Dispatch a final notification to enabled channels."""

        if self.telegram_enabled:
            self._send_or_edit_telegram(stage_key, text)
        if self.email_enabled:
            self._send_email(subject, text)

    def _send_or_edit_telegram(self, stage_key: str, text: str) -> None:
        """Send a new Telegram message or edit the existing stage message."""

        with self._delivery_lock:
            message_id = self._telegram_message_ids.get(stage_key)
            if message_id is None:
                response = self._telegram_post(
                    "sendMessage",
                    {
                        "chat_id": self.config.telegram.chat_id,
                        "text": self._truncate(text),
                        "disable_web_page_preview": "true",
                    },
                )
                result = response.get("result", {}) if isinstance(response, dict) else {}
                new_id = result.get("message_id")
                if isinstance(new_id, int):
                    self._telegram_message_ids[stage_key] = new_id
                return
            try:
                self._telegram_post(
                    "editMessageText",
                    {
                        "chat_id": self.config.telegram.chat_id,
                        "message_id": str(message_id),
                        "text": self._truncate(text),
                        "disable_web_page_preview": "true",
                    },
                )
            except Exception as exc:
                LOGGER.warning("Telegram edit failed, sending a new message: %s", exc)
                self._telegram_message_ids.pop(stage_key, None)
                self._send_or_edit_telegram(stage_key, text)

    def _telegram_post(self, method: str, payload: dict[str, str]) -> dict[str, Any]:
        """Call the Telegram Bot API.

        Args:
            method: Telegram method name.
            payload: Form data.

        Returns:
            Parsed JSON response.
        """

        token = self.config.telegram.bot_token.strip()
        url = TELEGRAM_API.format(token=urllib.parse.quote(token), method=method)
        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def _send_email(self, subject: str, text: str) -> None:
        """Send an email notification through SMTP."""

        email = self.config.email
        message = EmailMessage()
        message["Subject"] = f"Micro LLM Creator - {subject}"
        message["From"] = email.from_email
        message["To"] = email.to_email
        message.set_content(text)
        if email.use_tls:
            smtp = smtplib.SMTP(email.smtp_host, email.smtp_port, timeout=20)
            try:
                smtp.starttls()
                if email.username:
                    smtp.login(email.username, email.password)
                smtp.send_message(message)
            finally:
                smtp.quit()
        else:
            with smtplib.SMTP_SSL(email.smtp_host, email.smtp_port, timeout=20) as smtp_ssl:
                if email.username:
                    smtp_ssl.login(email.username, email.password)
                smtp_ssl.send_message(message)

    def _enqueue_terminal(self, notification: _PendingNotification) -> None:
        """Queue a completion or failure without exceeding the delivery bound."""

        with self._condition:
            if not self._accepting:
                return
            self._pending_progress.pop(notification.stage_key, None)
            while self._pending_count_locked() >= self._max_pending and self._pending_progress:
                self._pending_progress.popitem(last=False)
            while self._pending_count_locked() >= self._max_pending and self._accepting:
                self._condition.wait(timeout=0.1)
            if not self._accepting:
                return
            self._terminal_notifications.append(notification)
            self._condition.notify()

    def _dispatch_loop(self) -> None:
        """Deliver queued notifications on one bounded background worker."""

        while True:
            with self._condition:
                notification = self._next_notification_locked()
                while notification is None:
                    if not self._accepting and self._pending_count_locked() == 0:
                        return
                    self._condition.wait(timeout=self._next_progress_wait_locked())
                    notification = self._next_notification_locked()
            try:
                if notification.kind == "progress":
                    self._send_progress(
                        notification.stage_key,
                        notification.subject,
                        notification.text,
                    )
                else:
                    self._send_completion(
                        notification.stage_key,
                        notification.subject,
                        notification.text,
                    )
            except Exception:
                LOGGER.exception("Notification delivery failed")
            finally:
                with self._condition:
                    self._condition.notify_all()

    def _next_notification_locked(self) -> Optional[_PendingNotification]:
        if self._terminal_notifications:
            return self._terminal_notifications.popleft()
        now = time.monotonic()
        interval = max(
            self._minimum_progress_interval_seconds,
            float(self.config.progress_interval_seconds),
        )
        for stage_key, notification in list(self._pending_progress.items()):
            last = self._last_progress_at.get(stage_key)
            if last is None or now - last >= interval:
                self._pending_progress.pop(stage_key)
                self._last_progress_at[stage_key] = now
                return notification
        return None

    def _next_progress_wait_locked(self) -> float:
        if not self._pending_progress:
            return 1.0
        now = time.monotonic()
        interval = max(
            self._minimum_progress_interval_seconds,
            float(self.config.progress_interval_seconds),
        )
        waits = [
            max(0.01, interval - (now - self._last_progress_at.get(stage_key, 0.0)))
            for stage_key in self._pending_progress
            if stage_key in self._last_progress_at
        ]
        return min(waits, default=0.01)

    def _pending_count_locked(self) -> int:
        return len(self._pending_progress) + len(self._terminal_notifications)

    def _reload_if_changed(self) -> None:
        now = time.monotonic()
        with self._condition:
            if now - self._last_config_check_at < self._config_check_interval_seconds:
                return
            self._last_config_check_at = now
            mtime_ns = self._read_config_mtime()
            if mtime_ns == self._config_mtime_ns:
                return
        self.reload()

    def _read_config_mtime(self) -> Optional[int]:
        try:
            return self.config_path.stat().st_mtime_ns
        except OSError:
            return None

    def _log_disabled(self) -> None:
        """Log a one-time hint when notifications are configured off."""

        if self._disabled_warning_logged:
            return
        self._disabled_warning_logged = True
        telegram = self.config.telegram
        email = self.config.email
        LOGGER.warning(
            "Notifications skipped because no channel is enabled. "
            "Config=%s telegram_enabled=%s telegram_token_set=%s telegram_chat_id_set=%s "
            "email_enabled=%s email_host_set=%s email_to_set=%s",
            self.config_path,
            telegram.enabled,
            bool(telegram.bot_token.strip()),
            bool(telegram.chat_id.strip()),
            email.enabled,
            bool(email.smtp_host.strip()),
            bool(email.to_email.strip()),
        )

    @staticmethod
    def _format_message(title: str, lines: list[str], percent: Optional[int]) -> str:
        """Build a compact plain-text notification body."""

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        body = [title, f"Time: {now}"]
        if percent is not None:
            body.append(f"Progress: {max(0, min(100, int(percent)))}%")
        body.extend(line for line in lines if line)
        return "\n".join(body)

    @staticmethod
    def _truncate(text: str) -> str:
        """Keep Telegram messages within the API text limit."""

        if len(text) <= MAX_TELEGRAM_TEXT:
            return text
        return text[: MAX_TELEGRAM_TEXT - 40] + "\n... truncated ..."
