#!/usr/bin/env python3
"""
Zammad Live Chat AI Bot — Zammad 7.0
Connects as an agent via WebSocket and responds using an LLM.
"""

import asyncio
import json
import logging
import re
import os
import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

ZAMMAD_URL    = "https://zammad.hnee.de"
ZAMMAD_WS_URL = "wss://zammad.hnee.de/ws"
ZAMMAD_TOKEN  = os.environ.get("ZAMMAD_BOT_TOKEN", "")
ZAMMAD_USER   = os.environ.get("ZAMMAD_USER", "")
ZAMMAD_PASS   = os.environ.get("ZAMMAD_PASS", "")
ZAMMAD_SESSION_COOKIE = os.environ.get("ZAMMAD_SESSION_COOKIE", "")

LOCAL_AI_URL   = "http://10.1.2.239:1234/v1"
LOCAL_AI_MODEL = "google/gemma-4-26b-a4b"

ai_client = AsyncOpenAI(api_key="not-needed", base_url=LOCAL_AI_URL)

# Track conversation history and customer names per session
sessions: dict[str, list[dict]] = {}
session_names: dict[str, str] = {}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("zammad-bot")

SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Kundensupport-Mitarbeiter des ITSZ der HNEE. "
    "Antworte immer auf Deutsch und so präzise wie möglich. "
    "Versuche jede Frage selbst zu beantworten. "
    "Verwende den Tag [TICKET] NUR wenn: der Kunde explizit einen Mitarbeiter verlangt, "
    "oder du die Frage wirklich nicht beantworten kannst und ein Mensch eingreifen muss. "
    "Für normale Fragen, Begrüßungen und allgemeine Probleme antworte einfach direkt — KEIN [TICKET]."
)



async def create_zammad_ticket(session_id: str, customer_name: str, history: list[dict]) -> bool:
    """Create a Zammad ticket from the chat history via REST API."""
    transcript = "\n".join(
        f"{'Kunde' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in history
    )
    payload = {
        "title": f"Chat-Eskalation: {customer_name or 'Anonym'}",
        "group": "Allgemeiner Support",
        "customer_id": 16694,  # pbutest — fallback since chat guest has no Zammad account
        "article": {
            "subject": "Chat-Protokoll",
            "body": transcript,
            "type": "note",
            "internal": False,
        },
    }
    headers = {
        "Authorization": f"Token token={ZAMMAD_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{ZAMMAD_URL}/api/v1/tickets",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    log.info(f"Ticket #{data.get('id')} created for session {session_id}")
                    return True
                else:
                    log.error(f"Ticket creation failed: {resp.status} {await resp.text()}")
                    return False
    except Exception as e:
        log.error(f"Ticket creation error: {e}")
        return False


async def get_ai_reply(session_id: str, user_message: str) -> tuple[str, bool]:
    """Call the LLM and return (reply, needs_ticket)."""
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})

    response = await ai_client.chat.completions.create(
        model=LOCAL_AI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
        ],
    )
    reply = (response.choices[0].message.content or "").strip()

    needs_ticket = "[TICKET]" in reply
    reply = reply.replace("[TICKET]", "").strip()

    history.append({"role": "assistant", "content": reply})
    return reply, needs_ticket




async def bot_main():
    """
    Keeps a Playwright browser alive so Zammad's in-memory WebSocket session stays
    authenticated. Our bot WebSocket then connects with the same session cookie.
    """
    from playwright.async_api import async_playwright

    log.info("Playwright: Browser wird gestartet...")
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(ignore_https_errors=True)
    page = await context.new_page()

    try:
        # Queue to pass WebSocket events from sync Playwright callbacks into async code
        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_websocket(ws):
            log.info(f"Playwright: SPA WebSocket → {ws.url}")
            def on_frame(raw):
                try:
                    evts = json.loads(raw) if isinstance(raw, str) else []
                    if not isinstance(evts, list):
                        evts = [evts]
                    for e in evts:
                        if isinstance(e, dict):
                            loop.call_soon_threadsafe(event_queue.put_nowait, e)
                except Exception:
                    pass
            ws.on("framereceived", on_frame)

        page.on("websocket", on_websocket)

        log.info(f"Playwright: Navigiere zu {ZAMMAD_URL}")
        await page.goto(ZAMMAD_URL, wait_until="networkidle")

        # Login
        user_field = page.locator("input[name='username']").or_(page.locator("#username")).first
        pass_field = page.locator("input[name='password']").or_(page.locator("#password")).first
        await user_field.click(timeout=8000)
        await user_field.type(ZAMMAD_USER, delay=50)
        await page.keyboard.press("Tab")
        await pass_field.type(ZAMMAD_PASS, delay=50)
        await page.keyboard.press("Enter")

        await page.wait_for_url(lambda url: "#login" not in url, timeout=15000)
        log.info(f"Playwright: Login erfolgreich — URL: {page.url}")

        # Enable chat availability through the SPA's authenticated WebSocket
        await asyncio.sleep(3)
        result = await page.evaluate("""() => {
            try {
                App.WebSocket.send({event: 'chat_agent_state', data: {active: true}});
                return 'OK';
            } catch(e) { return 'FEHLER: ' + e.message; }
        }""")
        log.info(f"chat_agent_state gesendet → {result}")
        log.info("Bot ist online — wartet auf Chats via SPA-WebSocket...")

        accepted: set[str] = set()

        async def spa_send(event_name: str, data: dict):
            await page.evaluate(
                "([e, d]) => App.WebSocket.send({event: e, data: d})",
                [event_name, data],
            )

        while True:
            try:
                evt = await asyncio.wait_for(event_queue.get(), timeout=60)
            except asyncio.TimeoutError:
                # Keep session alive with a ping
                await page.evaluate("() => App.WebSocket.send({event: 'ping'})")
                continue

            event_name = evt.get("event", "")
            data = evt.get("data", {})

            if event_name not in ("chat_status_agent", "pong"):
                log.info(f"WS-Event: {event_name} | data: {str(data)[:200]}")

            if event_name == "chat_status_agent":
                waiting = data.get("waiting_chat_session_list", [])
                active_ids = data.get("active_agent_ids", [])
                log.info(f"chat_status_agent: wartend={len(waiting)}, aktive_agenten={active_ids}")
                for chat in waiting:
                    sid = chat.get("session_id")
                    if sid and sid not in accepted:
                        accepted.add(sid)
                        log.info(f"Akzeptiere Chat {sid[:8]}...")
                        await spa_send("chat_session_start", {"session_id": sid})
                        await asyncio.sleep(0.3)
                        await spa_send("chat_session_message", {
                            "session_id": sid,
                            "content": "Hallo! Ich bin der virtuelle Assistent. Wie kann ich Ihnen helfen?",
                            "type": "message",
                        })
                        log.info(f"Chat {sid[:8]} akzeptiert und begrüßt")

            elif event_name == "chat_session_message":
                sid = data.get("session_id")
                message = data.get("message", {})
                content = re.sub(r"<[^>]+>", " ", message.get("content", "")).strip()
                created_by_id = message.get("created_by_id")  # None = customer, int = agent
                if sid and content and created_by_id is None:
                    log.info(f"Kunde [{sid[:8]}]: {content[:80]}")
                    reply, needs_ticket = await get_ai_reply(sid, content)
                    await spa_send("chat_session_message", {
                        "session_id": sid, "content": reply, "type": "message"
                    })
                    log.info(f"Bot [{sid[:8]}]: {reply[:80]}")

                    if needs_ticket:
                        ok = await create_zammad_ticket(sid, session_names.get(sid, ""), sessions.get(sid, []))
                        msg = (
                            "Ich habe ein Support-Ticket für Sie erstellt. "
                            "Ein Mitarbeiter wird sich so bald wie möglich bei Ihnen melden."
                            if ok else
                            "Leider konnte ich kein Ticket erstellen. "
                            "Bitte nutzen Sie den 'Ticket erstellen'-Button im Chat."
                        )
                        await spa_send("chat_session_message", {"session_id": sid, "content": msg, "type": "message"})
                        await spa_send("chat_session_close", {"session_id": sid})

            elif event_name == "chat_session_closed":
                sid = data.get("session_id")
                if sid:
                    sessions.pop(sid, None)
                    session_names.pop(sid, None)
                    accepted.discard(sid)
                    log.info(f"Session {sid[:8]} geschlossen")
    finally:
        log.info("Playwright: Browser wird geschlossen")
        await browser.close()
        await playwright.stop()


if __name__ == "__main__":
    asyncio.run(bot_main())
