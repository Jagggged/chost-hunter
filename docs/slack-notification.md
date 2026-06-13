# Slack Notification Demo

Chost Hunter can send Slack Incoming Webhook messages when the AI agent creates
resource recommendations, applies Docker limits, or rolls back a risky change.

## Create A Slack Webhook

1. In Slack, create or open an app for the workspace.
2. Enable Incoming Webhooks.
3. Add a webhook to the channel used for the demo.
4. Copy the webhook URL into a local `.env` file. Do not commit `.env`.

## Local Environment

Create `.env` in the repository root:

```env
SLACK_NOTIFY_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_NOTIFY_ONLY_CHANGES=true
```

`SLACK_NOTIFY_ONLY_CHANGES=true` suppresses repeated no-op recommendations when
the recommended limits are already applied.

## Run The Demo Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d
```

Check the AI agent logs:

```bash
docker logs ai-agent --tail=100
```

Notification delivery attempts are also recorded locally:

```bash
tail -n 20 logs/notifications.jsonl
```

## Events Sent To Slack

- Advisory recommendations: recommendation only, no Docker update.
- Auto apply success: Docker CPU/memory limits updated.
- Rollback success: previous limits restored.
- Rollback failure: rollback attempted but failed.
- Manual Slack test from the dashboard or Control API.

Settings changes are intentionally not sent to Slack by default.

## Presentation Flow

1. Start the demo workload with the compose command above.
2. Open Grafana and confirm CPU/Memory movement for the demo containers.
3. Wait for the AI Agent to create a recommendation or auto apply event.
4. Confirm the Slack channel receives the Chost Hunter notification.
5. Use `docker logs ai-agent --tail=100` if you need to narrate the event source.
