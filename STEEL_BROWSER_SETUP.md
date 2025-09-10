# Steel Browser Integration

Run Steel Browser first, then connect the UI to it.

## 1. Start Steel Browser

Create `docker-compose.yml`:
```yaml
services:
  api:
    image: ghcr.io/steel-dev/steel-browser-api:latest
    ports:
      - "3000:3000"
      - "9223:9223"
    environment:
      - DOMAIN=${DOMAIN:-localhost:3000}
      - CDP_DOMAIN=${CDP_DOMAIN:-localhost:9223}
    volumes:
      - ./.cache:/app/.cache
    networks:
      - steel-network

  ui:
    image: ghcr.io/steel-dev/steel-browser-ui:latest
    ports:
      - "5173:80"
    environment:
      - API_URL=${API_URL:-http://api:3000}
    depends_on:
      - api
    networks:
      - steel-network

networks:
  steel-network:
    name: steel-network
    driver: bridge
```

Start it:
```bash
docker-compose up -d
```

## 2. Verify Steel Browser is Running

Check that CDP is available:
```bash
curl http://localhost:9223/json/version
```

## 3. Start Browser-Use UI

```bash
uv run python -m browser_use.cli recorder ui --steel-cdp ws://localhost:9223
```

Or directly:
```bash
uv run python -m browser_use.ui.server
```

## 4. Open UI

Navigate to: http://localhost:8080

The UI will automatically connect to Steel Browser via CDP at `ws://localhost:9223`.
