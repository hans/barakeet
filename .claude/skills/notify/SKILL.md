---
name: notify
description: Send a push notification via ntfy.sh. Use this to ping the user when a long-running task completes or when something needs their attention.
argument-hint: [message]
allowed-tools: Bash(curl *)
---

Send a notification to ntfy.sh using curl. The topic for this project is `labctl-jon-ucsf`.

Use the following command:

```
curl -s -d "$ARGUMENTS" ntfy.sh/labctl-jon-ucsf
```

If no message argument is provided, send a default message: "Task complete".

After sending, confirm to the user that the notification was sent.
