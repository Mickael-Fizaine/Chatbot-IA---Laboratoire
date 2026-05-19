import os
import traceback
from dotenv import load_dotenv

load_dotenv()

from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
from bot.teams_bot import LabChatBot

settings = BotFrameworkAdapterSettings(
    app_id=os.getenv("MICROSOFT_APP_ID", ""),
    app_password=os.getenv("MICROSOFT_APP_PASSWORD", ""),
)
adapter = BotFrameworkAdapter(settings)


async def on_error(context: TurnContext, error: Exception):
    print(f"\n[on_turn_error] {type(error).__name__}: {error}")
    traceback.print_exc()
    await context.send_activity("Une erreur interne est survenue. Veuillez reessayer.")


adapter.on_turn_error = on_error

bot = LabChatBot()


async def messages(req: web.Request) -> web.Response:
    if "application/json" not in req.headers.get("Content-Type", ""):
        return web.Response(status=415, text="Content-Type must be application/json")

    try:
        body = await req.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body")

    try:
        activity = Activity().deserialize(body)
    except Exception as exc:
        print(f"[deserialize error] {type(exc).__name__}: {exc}")
        return web.Response(status=400, text=f"Cannot deserialize activity: {exc}")

    auth_header = req.headers.get("Authorization", "")
    print(f"[messages] type={body.get('type')} auth={'yes' if auth_header else 'no'}")

    try:
        invoke_response = await adapter.process_activity(activity, auth_header, bot.on_turn)
        if invoke_response:
            return web.json_response(data=invoke_response.body, status=invoke_response.status)
        return web.Response(status=201)
    except PermissionError as exc:
        print(f"[auth error] {exc}")
        return web.Response(status=401, text="Unauthorized")
    except Exception as exc:
        print(f"[messages handler error] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return web.Response(status=500, text=str(exc))


async def health(req: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "lab-chatbot"})


app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/health", health)

if __name__ == "__main__":
    print("=" * 50)
    print(f"MICROSOFT_APP_ID  : '{os.getenv('MICROSOFT_APP_ID', '')}'")
    print(f"MICROSOFT_APP_PWD : '{'***' if os.getenv('MICROSOFT_APP_PASSWORD') else ''}'")
    print("=" * 50)
    web.run_app(app, host="0.0.0.0", port=3978)
