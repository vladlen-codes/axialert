# ThingsBoard Setup

## 1. Create Device In ThingsBoard

1. Open your ThingsBoard cloud instance.
2. Create a new device (for example: `axialert-device`).
3. Copy the device access token.

## 2. Configure Environment

From backend folder:

```bash
cp .env.thingsboard.example .env.thingsboard
# edit .env.thingsboard and set TB_ACCESS_TOKEN
source .env.thingsboard
```

## 3. Test One Sync

```bash
/Users/vlad/Documents/GitHub/axialert/.venv/bin/python thingsboard_bridge.py --once
```

Expected output includes `Synced vector #...`.

## 4. Run With Stack

When `TB_ACCESS_TOKEN` is exported, `run_stack.sh` auto-starts bridge.

Logs:

- `logs/thingsboard_bridge.log`

## 5. Verify

```bash
./verify_stack.sh
```

You should see `PASS thingsboard_bridge.py` when enabled.

## 6. End-To-End Smoke Test

This command posts one fresh vector to ingestion and confirms bridge sync to ThingsBoard:

```bash
./tb_smoke_test.sh
```

## Notes

- Core local pipeline continues to work even if ThingsBoard is unavailable.
- Bridge retries on errors and keeps running.
