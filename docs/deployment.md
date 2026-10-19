# Deployment (Docker, then AWS EC2)

The app is three containers: `web` (nginx serving the React build and proxying `/api` and `/health`),
`backend` (FastAPI plus the CPU model stack) and `ollama` (the language model). No API keys are
needed: every model runs on the machine. Data lives on named volumes (`storage`, `chroma_db`,
`hf_cache`, `ollama_models`), so it survives restarts and rebuilds.

## Run locally (tested)

Reuses the Ollama already installed on the PC, so the model is not downloaded twice.

```
copy .env.docker.example .env
docker compose up -d --build
```

Open http://localhost:8080. Stop with `docker compose down` (volumes are kept; add `-v` to wipe them).

What was verified on a Windows PC with Docker Desktop: images build, `/health` and `/api/videos`
answer through nginx, a video uploads through the proxy, transcription, chunking, embedding and
indexing run in the container, search returns precise `jump_time` values, a chat question is
answered through the host's Ollama, and the uploaded video is still listed after
`docker compose down` and `up`. The first transcription is slow because Whisper, MiniLM and the
reranker download on first use (cached on the `hf_cache` volume afterwards).

## Run on an AWS EC2 server

1. Launch Ubuntu 22.04, 16 GB RAM (for example `m6i.xlarge` or `t3.xlarge`), 60 GB gp3 disk.
   Security group: open 22 and 80 (and 443 if you add HTTPS) to your own IP only. Never open
   8000 or 11434. The app has no login, so anyone who can reach it can upload and delete videos.
2. Install Docker and the compose plugin (https://docs.docker.com/engine/install/ubuntu/), then
   `sudo usermod -aG docker ubuntu` and log in again.
3. Copy the project to the server (`git clone ...` or `scp`).
4. Start with the Ollama container (do not create a `.env` that points at a host Ollama):

```
docker compose --profile ollama up -d --build
docker compose exec ollama ollama pull qwen2.5:3b
```

5. Open `http://<public-ip>:8080` (or set `WEB_PORT=80` in `.env` to use port 80).
6. Optional HTTPS: point a domain at an Elastic IP and put Caddy or certbot in front of port 8080.

## Known limits

- YouTube downloads from AWS addresses are often blocked by YouTube's bot checks. Upload files instead.
- Transcription is a long synchronous request (there is no job queue yet); nginx timeouts are 30 minutes.
- Docker Desktop on a low-memory PC can be slow: the model stack needs several GB of RAM.

## Free demo: frontend on Vercel, backend in Docker on your own PC

The page is a static build, so it can live on Vercel while the models run on your PC. A Cloudflare
Quick Tunnel gives the PC a public https address (no account needed, testing use only).

1. Push the repo to GitHub, then on vercel.com import it: root directory `frontend`, Vite preset
   (build `npm run build`, output `dist`). No environment variable is required.
2. Put the backend in demo mode so strangers cannot upload or delete: add `READ_ONLY=true` to the
   project-root `.env`, then `docker compose up -d`. Prepare your videos first with `READ_ONLY` unset
   (or set to `false`), because processing is disabled in demo mode.
3. Start the tunnel: `cloudflared tunnel --url http://localhost:8080`. Copy the printed
   `https://<random>.trycloudflare.com` address.
4. Open your Vercel link. It shows "backend offline" with an address box the first time: paste the
   tunnel address and press "Use this address". The browser remembers it. The tunnel address changes
   each time the tunnel restarts, so repeat this step (no redeploy needed).

What works through the tunnel: browsing, playback, transcript search, visual and fused search,
questions and answers, and saved summaries. Transcription and other long processing should be done
locally: a proxy such as Cloudflare typically gives up on a request after about 100 seconds (not tested
here). The PC, Docker and the tunnel must stay running for the site to work.

`READ_ONLY=true` blocks every POST except search and chat, all deletes, uploads and summary generation
(HTTP 403), and the interface hides those controls. The app still has no login: only share the link
with people you trust and stop the tunnel afterwards.
