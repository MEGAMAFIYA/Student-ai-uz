"""
🤖 Multi-AI Telegram Bot
Gemini asosiy + DeepSeek + Groq + Pollinations
"""

import os, re, asyncio, logging
from io import BytesIO
from urllib.parse import quote

from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

import httpx
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER

load_dotenv()

# =================== SOZLAMALAR ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

# =================== AI FUNKSIYALAR ===================

async def ai_gemini(prompt: str, system: str = "") -> str | None:
    if not gemini_model:
        return None
    try:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = await asyncio.to_thread(
            gemini_model.generate_content, full_prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini xato: {e}")
        return None


async def ai_deepseek(prompt: str) -> str | None:
    if not DEEPSEEK_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"DeepSeek xato: {e}")
        return None


async def ai_groq(prompt: str, system: str = "") -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system or "Siz foydali yordamchisiz. O'zbek tilida javob bering."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq xato: {e}")
        return None


async def ai_pollinations_image(prompt: str) -> bytes | None:
    try:
        url = (
            f"https://image.pollinations.ai/prompt/{quote(prompt)}"
            f"?width=1024&height=1024&nologo=true&model=flux"
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.content
    except Exception as e:
        logger.error(f"Pollinations xato: {e}")
        return None

# =================== INTENT ===================

INTENT_KEYWORDS = {
    "image": [
        "rasm chiz", "rasm yarat", "surat chiz", "surat yarat",
        "chizib ber", "rasmini chiz", "rasmini yarat",
        "draw", "create image", "generate image", "paint", "picture of"
    ],
    "code": [
        "kod yoz", "kod tuz", "dastur yoz", "skript yoz",
        "code", "python", "javascript", "function",
        "def ", "import ", "class ", "html", "css"
    ],
    "pdf": [
        "pdf", "darslik yarat", "kitob", "qollanma",
        "tutorial", "darslik tayyorla", "pdf qil", "pdf yarat"
    ],
    "math": [
        "hisobla", "qancha bo'ladi", "formula", "tenglama",
        "masala yech", "yechimini top"
    ],
    "translate": [
        "tarjima", "tarjima qil", "translate",
        "ingliz tiliga", "rus tiliga", "o'zbek tiliga"
    ],
}


def detect_intent(text: str) -> str:
    t = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return intent
    return "chat"

# =================== PDF ===================

def make_pdf(title: str, content: str) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'],
        fontSize=22, alignment=TA_CENTER, spaceAfter=20,
        textColor="#1a5490"
    )
    body_style = ParagraphStyle(
        'BodyStyle', parent=styles['Normal'],
        fontSize=12, leading=20, alignment=TA_LEFT
    )
    h2_style = ParagraphStyle(
        'H2Style', parent=styles['Heading2'],
        fontSize=14, spaceBefore=12, spaceAfter=8,
        textColor="#2a6fb0"
    )

    story = [Paragraph(title, title_style), Spacer(1, 1*cm)]

    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        block_safe = (block.replace("&", "&amp;")
                          .replace("<", "&lt;")
                          .replace(">", "&gt;"))
        if block_safe.startswith("#"):
            story.append(Paragraph(block_safe.lstrip("# ").strip(), h2_style))
        else:
            story.append(Paragraph(block_safe.replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 0.4*cm))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =================== HANDLERLAR ===================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Multi-AI Bot ga xush kelibsiz!**\n\n"
        "🧠 **Gemini** — asosiy suhbat\n"
        "💎 **DeepSeek** — kod, matematika\n"
        "⚡ **Groq/Llama 3.3** — backup\n"
        "🎨 **Pollinations** — bepul rasm\n"
        "📄 **PDF** — darslik yasash\n\n"
        "Oddiy matn yozing — qaysi AI kerakligini o'zim aniqlayman!",
        parse_mode=ParseMode.MARKDOWN
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 **Foydalanish**\n\n"
        "• `rasm chiz: kosmik kema`\n"
        "• `Python da bot kodini yoz`\n"
        "• `pdf: Matematika formulalari`\n"
        "• `2x + 5 = 15 yech`\n"
        "• `tarjima: Hello world`\n"
        "• 📷 Rasm yuboring — tahlil",
        parse_mode=ParseMode.MARKDOWN
    )


async def smart_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    intent = detect_intent(user_text)

    await context.bot.send_chat_action(
        update.effective_chat.id,
        ChatAction.UPLOAD_PHOTO if intent == "image" else ChatAction.TYPING
    )

    # ====== RASM ======
    if intent == "image":
        prompt = re.sub(
            r"^(rasm chiz|rasm yarat|surat chiz|surat yarat|chizib ber)[\s:]*",
            "", user_text, flags=re.IGNORECASE
        ).strip()
        if not prompt:
            await update.message.reply_text(
                "❓ Nima chizay? Masalan:\n`rasm chiz: quyosh botayotgan tog'lar`"
            )
            return

        msg = await update.message.reply_text(
            f"🎨 Chizayapman: *{prompt[:100]}*...",
            parse_mode=ParseMode.MARKDOWN
        )
        img = await ai_pollinations_image(prompt)
        if img:
            await update.message.reply_photo(
                photo=InputFile(BytesIO(img), filename="image.png"),
                caption=f"✨ **Pollinations.ai**\n📝 {prompt[:200]}"
            )
            await msg.delete()
        else:
            await msg.edit_text("❌ Rasm yaratib bo'lmadi.")
        return

    # ====== PDF ======
    if intent == "pdf":
        topic = re.sub(
            r"^(pdf|darslik|kitob)[\s:]*",
            "", user_text, flags=re.IGNORECASE
        ).strip()
        if not topic:
            await update.message.reply_text(
                "❓ Qaysi mavzu? Masalan:\n`pdf: Python asoslari`"
            )
            return

        msg = await update.message.reply_text(
            f"📚 PDF tayyorlayapman: *{topic}*...",
            parse_mode=ParseMode.MARKDOWN
        )
        system = (
            "O'qituvchi. Darslik yozing. "
            "Tuzilishi: #Kirish, #Asosiy tushunchalar, "
            "#Misol, #Xulosa. O'zbek tilida, aniq."
        )
        content = await ai_gemini(
            f"'{topic}' mavzusida qisqa darslik yoz.", system
        )
        if not content and GROQ_API_KEY:
            content = await ai_groq(
                f"'{topic}' mavzusida qisqa darslik yoz.",
                "O'qituvchi. 3-5 bet. O'zbek tilida. # bilan bo'limlarga ajrating."
            )

        if not content:
            await msg.edit_text("❌ AI dan javob olib bo'lmadi.")
            return

        try:
            pdf_buf = make_pdf(topic.title(), content)
            await update.message.reply_document(
                document=InputFile(pdf_buf, filename=f"{topic[:30]}.pdf"),
                caption=f"📄 {topic}\n\n✨ Gemini + ReportLab"
            )
            await msg.delete()
        except Exception as e:
            await msg.edit_text(f"❌ PDF xato: {e}\n\n📝 Matn:\n{content[:3000]}")
        return

    # ====== MATNLI JAVOB ======
    response = None
    used_ai = "Gemini"

    if intent in ("code", "math"):
        response = await ai_deepseek(user_text)
        if response:
            used_ai = "💎 DeepSeek"
        else:
            response = await ai_gemini(user_text)
            used_ai = "🧠 Gemini"

    elif intent == "translate":
        response = await ai_gemini(
            user_text,
            "Professional tarjimon. Aniq va ravon tarjima qiling. "
            "Manba tilini saqlang, faqat tarjimasini bering."
        )
        used_ai = "🧠 Gemini"

    else:
        response = await ai_gemini(user_text)
        if not response and GROQ_API_KEY:
            response = await ai_groq(user_text)
            used_ai = "⚡ Groq/Llama"
        else:
            used_ai = "🧠 Gemini"

    if not response:
        await update.message.reply_text(
            "❌ Barcha AI lar ishlamadi. API key larni tekshiring."
        )
        return

    if len(response) > 4000:
        bio = BytesIO(response.encode("utf-8"))
        await update.message.reply_document(
            document=InputFile(bio, filename="javob.txt"),
            caption=f"💡 Javob faylga yozildi\n🤖 {used_ai}"
        )
    else:
        await update.message.reply_text(
            f"💡 **{used_ai}**\n\n{response}",
            parse_mode=ParseMode.MARKDOWN
        )


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo or not gemini_model:
        return
    await context.bot.send_chat_action(
        update.effective_chat.id, ChatAction.TYPING
    )

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)

    import PIL.Image
    img = PIL.Image.open(bio)
    caption = update.message.caption or "Bu rasmda nima bor?"

    try:
        response = await asyncio.to_thread(
            gemini_model.generate_content, [caption, img]
        )
        await update.message.reply_text(
            f"🖼 **Gemini Vision**\n\n{response.text}",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Tahlil xatosi: {e}")


# =================== ISHGA TUSHIRISH ===================

def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN o'rnatilmagan!")
        return
    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY yo'q")

    print("🤖 Bot ishga tushmoqda...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, smart_handler)
    )

    print("✅ Bot tayyor! /start yuboring.")
    app.run_polling()


if __name__ == "__main__":
    main()