# Moviebot HeimdaLLM demo

Install requirements (note: only direct dependencies are frozen):

```
pip install requirements.txt
```

Create `.env` file with the following env vars:

- `DISCORD_TOKEN`
- `OPENAI_API_SECRET`
- `POSTGRES_PASSWORD`

Edit `movies.make_conn()` to have the correct db connection info.

Then run:

```
python bot.py
```
