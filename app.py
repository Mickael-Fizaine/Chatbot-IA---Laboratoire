import os
from dotenv import load_dotenv

load_dotenv()

from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity
from bot.teams_bot import LabChatBot

settings = BotFrameworkAdapterSettings(
    app_id=os.getenv("MICROSOFT_APP_ID", ""),
    app_password=os.getenv("MICROSOFT_APP_PASSWORD", ""),
)
adapter = BotFrameworkAdapter(settings)
bot = LabChatBot()


async def messages(req: web.Request) -> web.Response:
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")
    await adapter.process_activity(activity, auth_header, bot.on_turn)
    return web.Response(status=200)


async def health(req: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "lab-chatbot"})


app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/health", health)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=3978)
