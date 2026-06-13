import json

from ai import config
from ai.agent import notifier


def test_slack_default_statuses_include_limit_events():
    assert "recommended" in config.SLACK_NOTIFY_STATUSES
    assert "applied" in config.SLACK_NOTIFY_STATUSES
    assert "failed" in config.SLACK_NOTIFY_STATUSES
    assert "rolled_back" in config.SLACK_NOTIFY_STATUSES
    assert "rollback_failed" in config.SLACK_NOTIFY_STATUSES
    assert "settings_updated" not in config.SLACK_NOTIFY_STATUSES
    assert len(config.SLACK_NOTIFY_STATUSES) == len(set(config.SLACK_NOTIFY_STATUSES))


def test_routine_noop_recommendations_are_suppressed(monkeypatch):
    monkeypatch.setattr(config, "SLACK_NOTIFY_ONLY_CHANGES", True)

    assert notifier._is_routine_noop_recommendation({
        "status": "recommended",
        "reason": "recommended limits already applied; no docker update",
        "applied_limits": None,
    })


def test_noop_recommendations_can_be_allowed(monkeypatch):
    monkeypatch.setattr(config, "SLACK_NOTIFY_ONLY_CHANGES", False)

    assert not notifier._is_routine_noop_recommendation({
        "status": "recommended",
        "reason": "recommended limits already applied; no docker update",
        "applied_limits": None,
    })


def test_real_limit_recommendations_are_not_suppressed(monkeypatch):
    monkeypatch.setattr(config, "SLACK_NOTIFY_ONLY_CHANGES", True)

    assert not notifier._is_routine_noop_recommendation({
        "status": "recommended",
        "reason": "advisory policy; no docker update",
        "applied_limits": None,
    })
    assert not notifier._is_routine_noop_recommendation({
        "status": "applied",
        "reason": None,
        "applied_limits": {"cpu_quota": 0.5, "memory_bytes": 128},
    })


def test_notify_action_disabled_does_not_post(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(config, "NOTIFICATION_LOG_PATH", str(tmp_path / "notifications.jsonl"))
    monkeypatch.setattr(
        notifier,
        "get_slack_settings",
        lambda: {"enabled": False, "webhook_url": ""},
    )
    monkeypatch.setattr(notifier.requests, "post", lambda *args, **kwargs: calls.append(args))

    notifier.notify_action({"id": "a1", "container": "demo", "status": "applied"})

    assert calls == []
    entries = _read_notification_entries(tmp_path / "notifications.jsonl")
    assert entries[-1]["notification_status"] == "disabled"


def test_notify_action_without_webhook_does_not_post(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(config, "NOTIFICATION_LOG_PATH", str(tmp_path / "notifications.jsonl"))
    monkeypatch.setattr(
        notifier,
        "get_slack_settings",
        lambda: {"enabled": True, "webhook_url": ""},
    )
    monkeypatch.setattr(notifier.requests, "post", lambda *args, **kwargs: calls.append(args))

    notifier.notify_action({"id": "a1", "container": "demo", "status": "applied"})

    assert calls == []
    entries = _read_notification_entries(tmp_path / "notifications.jsonl")
    assert entries[-1]["notification_status"] == "disabled"


def test_notify_action_posts_payload_when_enabled(monkeypatch, tmp_path):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(config, "NOTIFICATION_LOG_PATH", str(tmp_path / "notifications.jsonl"))
    monkeypatch.setattr(
        notifier,
        "get_slack_settings",
        lambda: {"enabled": True, "webhook_url": "https://hooks.slack.com/services/test"},
    )
    monkeypatch.setattr(notifier.requests, "post", fake_post)

    notifier.notify_action({
        "id": "a1",
        "container": "demo-mixed-wave",
        "policy": "auto",
        "status": "applied",
        "recommended_limits": {"cpu_quota": 0.1, "memory_bytes": 1610612736},
        "applied_limits": {"cpu_quota": 0.1, "memory_bytes": 1610612736},
        "previous_limits": {"cpu_quota": 11000, "memory_bytes": 1632087572},
    })

    assert len(calls) == 1
    assert calls[0]["timeout"] == config.SLACK_TIMEOUT_SEC
    assert "demo-mixed-wave" in calls[0]["json"]["text"]
    block_text = json.dumps(calls[0]["json"], ensure_ascii=False)
    assert "Applied CPU" in block_text
    assert "Previous Memory Limit" in block_text
    assert "Docker limit updated successfully" in block_text
    entries = _read_notification_entries(tmp_path / "notifications.jsonl")
    assert entries[-1]["notification_status"] == "sent"


def test_notify_action_records_failed_post_without_raising(monkeypatch, tmp_path):
    def fake_post(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(config, "NOTIFICATION_LOG_PATH", str(tmp_path / "notifications.jsonl"))
    monkeypatch.setattr(
        notifier,
        "get_slack_settings",
        lambda: {"enabled": True, "webhook_url": "https://hooks.slack.com/services/test"},
    )
    monkeypatch.setattr(notifier.requests, "post", fake_post)

    notifier.notify_action({"id": "a1", "container": "demo", "status": "applied"})

    entries = _read_notification_entries(tmp_path / "notifications.jsonl")
    assert entries[-1]["notification_status"] == "failed"
    assert "network down" in entries[-1]["detail"]


def test_noop_event_is_suppressed_without_post(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(config, "SLACK_NOTIFY_ONLY_CHANGES", True)
    monkeypatch.setattr(config, "NOTIFICATION_LOG_PATH", str(tmp_path / "notifications.jsonl"))
    monkeypatch.setattr(
        notifier,
        "get_slack_settings",
        lambda: {"enabled": True, "webhook_url": "https://hooks.slack.com/services/test"},
    )
    monkeypatch.setattr(notifier.requests, "post", lambda *args, **kwargs: calls.append(args))

    notifier.notify_action({
        "id": "a1",
        "container": "demo",
        "status": "recommended",
        "reason": "recommended limits already applied; no docker update",
        "applied_limits": None,
    })

    assert calls == []
    entries = _read_notification_entries(tmp_path / "notifications.jsonl")
    assert entries[-1]["notification_status"] == "suppressed"


def _read_notification_entries(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
