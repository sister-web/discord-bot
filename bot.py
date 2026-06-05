import discord
from discord import app_commands
from google import genai
import re
import io
import json
import os
import asyncio
import aiohttp
import time as _time

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

histories = {}
autoreply_guilds = set()
last_bets = {}
autoreply_histories = {}
_ai_client = None

def get_ai_client():
    global _ai_client
    if _ai_client is None:
        _ai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _ai_client

def get_settings_file():
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        return "/data/settings.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

SETTINGS_FILE = get_settings_file()
_settings_cache = None

def load_settings():
    global _settings_cache, SETTINGS_FILE
    if _settings_cache is not None:
        return _settings_cache
    SETTINGS_FILE = get_settings_file()
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            _settings_cache = json.load(f)
    else:
        _settings_cache = {}
    return _settings_cache

def save_settings(settings):
    global _settings_cache
    _settings_cache = settings
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

def can_use_bot(message):
    settings = load_settings()
    guild_id = str(message.guild.id) if message.guild else None
    if not guild_id:
        return False
    allowed = settings.get(guild_id, {}).get("allowed_roles", [])
    if not allowed:
        return False
    user_role_ids = [str(r.id) for r in message.author.roles]
    return any(r in user_role_ids for r in allowed)

def extract_code(text):
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        lang, code = matches[0]
        ext = {"python": "py", "lua": "lua", "js": "js", "javascript": "js",
               "ts": "ts", "html": "html", "css": "css", "java": "java",
               "cpp": "cpp", "c": "c", "bash": "sh", "sh": "sh"}.get(lang.lower(), "txt")
        return code.strip(), ext
    return None, None

def remove_code_blocks(text):
    return re.sub(r"```(\w+)?\n.*?```", "[コードはファイルを参照]", text, flags=re.DOTALL).strip()

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_THINKING = "minimal"

def get_user_settings(user_id):
    settings = load_settings()
    return settings.get("users", {}).get(str(user_id), {})

async def ask_gemini_stream(channel_id, query, reply_msg, new_conversation=False, user_id=None):
    from google.genai import types
    ai = genai.Client(api_key=GEMINI_API_KEY)

    if new_conversation or channel_id not in histories:
        histories[channel_id] = []

    histories[channel_id].append({"role": "user", "parts": [{"text": query}]})

    user_cfg = get_user_settings(user_id) if user_id else {}
    model = user_cfg.get("model", DEFAULT_MODEL)
    thinking = user_cfg.get("thinking", DEFAULT_THINKING)

    full_text = ""
    last_edit = 0

    for _attempt in range(3):
        try:
            stream = await ai.aio.models.generate_content_stream(
                model=model,
                contents=histories[channel_id],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=thinking)
                )
            )
            async for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    now = asyncio.get_event_loop().time()
                    if now - last_edit >= 0.7:
                        display = full_text[:1900] + "..." if len(full_text) > 1900 else full_text + " ▌"
                        try:
                            await reply_msg.edit(content=f"🤖 {display}")
                            last_edit = now
                        except:
                            pass
            break
        except Exception as e:
            if "503" in str(e) or "unavailable" in str(e).lower():
                await asyncio.sleep(3)
                continue
            raise

    histories[channel_id].append({"role": "model", "parts": [{"text": full_text}]})
    if len(histories[channel_id]) > 20:
        histories[channel_id] = histories[channel_id][-20:]

    return full_text

# ==================== スラッシュコマンド ====================

@tree.command(name="mod", description="自分のモデルと思考レベルを変更します")
@app_commands.describe(
    モデル="使用するモデルを選んでください（省略可）",
    思考レベル="思考レベルを選んでください（省略可）"
)
@app_commands.choices(モデル=[
    app_commands.Choice(name="3.5 Flash（バランス・速め）",    value="gemini-3.5-flash"),
    app_commands.Choice(name="3.1 Pro（高精度・遅め）",        value="gemini-3.1-pro-preview"),
    app_commands.Choice(name="3.1 Flash-Lite（最速・軽め）",  value="gemini-3.1-flash-lite"),
])
@app_commands.choices(思考レベル=[
    app_commands.Choice(name="minimal（最速）",  value="minimal"),
    app_commands.Choice(name="low（速め）",      value="low"),
    app_commands.Choice(name="medium（普通）",   value="medium"),
    app_commands.Choice(name="high（最高精度）", value="high"),
])
async def set_mod(interaction: discord.Interaction, モデル: str = None, 思考レベル: str = None):
    settings = load_settings()
    uid = str(interaction.user.id)
    if "users" not in settings:
        settings["users"] = {}
    if uid not in settings["users"]:
        settings["users"][uid] = {}
    if モデル:
        settings["users"][uid]["model"] = モデル
    if 思考レベル:
        settings["users"][uid]["thinking"] = 思考レベル
    save_settings(settings)
    cfg = settings["users"][uid]
    m = cfg.get("model", DEFAULT_MODEL)
    t = cfg.get("thinking", DEFAULT_THINKING)
    await interaction.response.send_message(
        f"✅ 設定を更新しました。\nモデル: `{m}`\n思考レベル: `{t}`",
        ephemeral=True
    )

@tree.command(name="setrole", description="BOTを使えるロールを追加します（管理者のみ）")
@app_commands.describe(role="使用を許可するロール")
@app_commands.checks.has_permissions(administrator=True)
async def setrole(interaction: discord.Interaction, role: discord.Role):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if guild_id not in settings:
        settings[guild_id] = {"allowed_roles": []}
    if str(role.id) not in settings[guild_id]["allowed_roles"]:
        settings[guild_id]["allowed_roles"].append(str(role.id))
    save_settings(settings)
    await interaction.response.send_message(f"✅ `{role.name}` をBOT使用可能ロールに追加しました。", ephemeral=True)

@tree.command(name="removerole", description="BOT使用ロールを削除します（管理者のみ）")
@app_commands.describe(role="削除するロール")
@app_commands.checks.has_permissions(administrator=True)
async def removerole(interaction: discord.Interaction, role: discord.Role):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if guild_id in settings and str(role.id) in settings[guild_id]["allowed_roles"]:
        settings[guild_id]["allowed_roles"].remove(str(role.id))
        save_settings(settings)
        await interaction.response.send_message(f"✅ `{role.name}` を削除しました。", ephemeral=True)
    else:
        await interaction.response.send_message(f"⚠️ `{role.name}` は設定されていません。", ephemeral=True)

@setrole.error
@removerole.error
async def role_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="dm", description="指定したユーザーにBOTからDMを送ります（管理者のみ）")
@app_commands.describe(ユーザー="DMを送る相手", メッセージ="送るメッセージ内容", 回数="送る回数（デフォルト1）")
@app_commands.checks.has_permissions(administrator=True)
async def send_dm(interaction: discord.Interaction, ユーザー: discord.Member, メッセージ: str, 回数: int = 1):
    回数 = max(1, min(回数, 3000))
    await interaction.response.send_message(f"📨 {ユーザー.mention} に{回数}回送信中...", ephemeral=True)
    try:
        sent = 0
        for i in range(回数):
            try:
                await ユーザー.send(メッセージ)
                sent += 1
                if 回数 > 1:
                    await asyncio.sleep(0.3)
            except discord.HTTPException as e:
                if e.status == 429:
                    await asyncio.sleep(1)
                    await ユーザー.send(メッセージ)
                    sent += 1
        await interaction.edit_original_response(content=f"✅ {ユーザー.mention} に{sent}回送信しました。")
    except discord.Forbidden:
        await interaction.edit_original_response(content=f"⚠️ {ユーザー.mention} はDMを受け取れない設定になっています。")
    except Exception as e:
        await interaction.edit_original_response(content=f"⚠️ エラー: `{e}`")

@send_dm.error
async def dm_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="mes", description="特定のキーワードへの自動返信を設定します（管理者のみ）")
@app_commands.describe(
    キーワード="反応するキーワード",
    返信="返信内容（複数ある場合は | で区切る。例: おはよう|やあ|おっす）",
    削除="このキーワードの設定を削除する場合はTrue"
)
@app_commands.checks.has_permissions(administrator=True)
async def set_mes(interaction: discord.Interaction, キーワード: str, 返信: str = None, 削除: bool = False):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "mes" not in settings:
        settings["mes"] = {}
    if guild_id not in settings["mes"]:
        settings["mes"][guild_id] = {}

    if 削除:
        if キーワード in settings["mes"][guild_id]:
            del settings["mes"][guild_id][キーワード]
            save_settings(settings)
            await interaction.response.send_message(f"✅ `{キーワード}` の自動返信を削除しました。", ephemeral=True)
        else:
            await interaction.response.send_message(f"⚠️ `{キーワード}` は設定されていません。", ephemeral=True)
        return

    if not 返信:
        await interaction.response.send_message("返信内容を入力してください。", ephemeral=True)
        return

    replies = [r.strip() for r in 返信.split("|") if r.strip()]
    settings["mes"][guild_id][キーワード] = replies
    save_settings(settings)
    preview = " / ".join(f"`{r}`" for r in replies)
    await interaction.response.send_message(f"✅ `{キーワード}` → {preview} に設定しました。", ephemeral=True)

@set_mes.error
async def mes_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="yak", description="特定ユーザーへの自動返信スタイルを設定します（管理者のみ）")
@app_commands.describe(
    ユーザー="スタイルを設定するユーザー",
    スタイル="返信スタイル（例: メスガキ、馴れ馴れしい、丁寧、ツンデレ など。リセットは「なし」）"
)
@app_commands.checks.has_permissions(administrator=True)
async def set_yak(interaction: discord.Interaction, ユーザー: discord.Member, スタイル: str):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "yak" not in settings:
        settings["yak"] = {}
    if guild_id not in settings["yak"]:
        settings["yak"][guild_id] = {}

    if スタイル in ("なし", "reset", "リセット"):
        settings["yak"][guild_id].pop(str(ユーザー.id), None)
        save_settings(settings)
        await interaction.response.send_message(f"✅ {ユーザー.mention} のスタイルをリセットしました。", ephemeral=True)
    else:
        settings["yak"][guild_id][str(ユーザー.id)] = スタイル
        save_settings(settings)
        await interaction.response.send_message(f"✅ {ユーザー.mention} のスタイルを **{スタイル}** に設定しました。", ephemeral=True)

@set_yak.error
async def yak_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

# ==================== ギブアウェイ ====================

class GiveawayPanelView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="🎉", style=discord.ButtonStyle.primary, custom_id="giveaway_enter")
    async def enter(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        msg_id = str(interaction.message.id)
        giveaways = settings.get("giveaways", {}).get(guild_id, {})
        gw = giveaways.get(msg_id)

        if not gw:
            await interaction.response.send_message("⚠️ このギブアウェイは存在しません。", ephemeral=True)
            return
        if gw.get("ended"):
            await interaction.response.send_message("⚠️ このギブアウェイはすでに終了しています。", ephemeral=True)
            return
        if _time.time() > gw["end_time"]:
            await interaction.response.send_message("⚠️ このギブアウェイはすでに終了しています。", ephemeral=True)
            return

        uid = str(interaction.user.id)
        if uid in gw.get("entries", []):
            gw["entries"].remove(uid)
            save_settings(settings)
            await interaction.response.send_message("❌ 参加を取り消しました。", ephemeral=True)
        else:
            gw.setdefault("entries", []).append(uid)
            save_settings(settings)
            await interaction.response.send_message("✅ 参加しました！当選を待ってね🎉", ephemeral=True)

        await _update_giveaway_message(interaction.message, gw, interaction.guild)


async def _update_giveaway_message(message, gw, guild):
    discord_ts = f"<t:{int(gw['end_time'])}:R>"
    discord_ts_full = f"<t:{int(gw['end_time'])}:f>"
    host = guild.get_member(int(gw["host_id"]))
    host_display = host.mention if host else f"<@{gw['host_id']}>"
    entries = len(gw.get("entries", []))
    discord_ts = f"<t:{int(gw['end_time'])}:R>"
    discord_ts_full = f"<t:{int(gw['end_time'])}:f>"

    base_desc = (gw.get("description", "") + "\n\n") if gw.get("description") else ""
    desc = base_desc
    desc += f"**Ends:** {discord_ts} ({discord_ts_full})\n"
    desc += f"**Hosted by:** {host_display}\n**Entries:** {entries}\n**Winners:** {gw['winners']}"

    embed = discord.Embed(
        title=gw["prize"],
        description=desc,
        color=discord.Color.blurple()
    )

    try:
        await message.edit(embed=embed)
    except Exception:
        pass


async def _end_giveaway(guild, channel_id, msg_id, gw, settings):
    import random
    guild_id = str(guild.id)
    channel = guild.get_channel(int(channel_id))
    if not channel:
        return

    try:
        msg = await channel.fetch_message(int(msg_id))
    except Exception:
        return

    entries = gw.get("entries", [])
    winner_count = min(gw["winners"], len(entries))

    host = guild.get_member(int(gw["host_id"]))
    host_display = host.mention if host else f"<@{gw['host_id']}>"
    base_desc = (gw.get("description", "") + "\n\n") if gw.get("description") else ""
    end_desc = base_desc
    end_desc += f"**終了時刻:** <t:{int(gw['end_time'])}:f>\n"
    end_desc += f"**Hosted by:** {host_display}\n**Entries:** {len(entries)}\n**Winners:** {gw['winners']}"

    embed = discord.Embed(
        title=gw["prize"],
        description=end_desc,
        color=discord.Color.red()
    )
    embed.set_footer(text="🎊 ギブアウェイ終了")

    view = discord.ui.View()
    btn = discord.ui.Button(label="🎉", style=discord.ButtonStyle.primary, disabled=True)
    view.add_item(btn)

    try:
        await msg.edit(embed=embed, view=view)
    except Exception:
        pass

    gw["ended"] = True
    if winner_count == 0:
        await channel.send("😢 参加者がいなかったため、当選者なしでギブアウェイが終了しました。")
        gw["winner_ids"] = []
    else:
        winner_ids = random.sample(entries, winner_count)
        gw["winner_ids"] = winner_ids
        mentions = " ".join(f"<@{w}>" for w in winner_ids)
        await channel.send(
            f"🎊 **ギブアウェイ終了！**\n"
            f"**{gw['prize']}** の当選者: {mentions}\nおめでとうございます！🎉"
        )

    settings["giveaways"][guild_id][msg_id] = gw
    save_settings(settings)


async def giveaway_timer_task():
    await client.wait_until_ready()
    while not client.is_closed():
        try:
            settings = load_settings()
            now = _time.time()
            for guild_id, giveaways in settings.get("giveaways", {}).items():
                guild = client.get_guild(int(guild_id))
                if not guild:
                    continue
                for msg_id, gw in list(giveaways.items()):
                    if gw.get("ended"):
                        continue
                    if now >= gw["end_time"]:
                        ch_id = gw.get("channel_id")
                        if ch_id:
                            await _end_giveaway(guild, ch_id, msg_id, gw, settings)
        except Exception:
            pass
        await asyncio.sleep(15)


class GiveawayModal(discord.ui.Modal, title="ギブアウェイを作成"):
    duration = discord.ui.TextInput(
        label="Duration（例: 1h, 30m, 1d）",
        placeholder="Ex: 1h",
        required=True
    )
    winners = discord.ui.TextInput(
        label="Number of Winners",
        placeholder="1",
        default="1",
        required=True
    )
    prize = discord.ui.TextInput(
        label="Prize（景品名）",
        placeholder="例: Robux 1000",
        required=True
    )
    description = discord.ui.TextInput(
        label="Description（説明・省略可）",
        placeholder="例: 条件: サーバーメンバーであること",
        required=False,
        style=discord.TextStyle.paragraph,
        max_length=1000
    )

    async def on_submit(self, interaction: discord.Interaction):
        dur = self.duration.value.strip().lower()
        seconds = 0
        try:
            if dur.endswith("m"):
                seconds = int(dur[:-1]) * 60
            elif dur.endswith("h"):
                seconds = int(dur[:-1]) * 3600
            elif dur.endswith("d"):
                seconds = int(dur[:-1]) * 86400
            else:
                await interaction.response.send_message("⚠️ 時間形式が正しくありません（例: 1h, 30m, 1d）", ephemeral=True)
                return
        except ValueError:
            await interaction.response.send_message("⚠️ 時間の数値が正しくありません。", ephemeral=True)
            return

        try:
            winner_count = max(1, int(self.winners.value.strip()))
        except ValueError:
            await interaction.response.send_message("⚠️ 当選人数は整数で入力してください。", ephemeral=True)
            return

        end_time = _time.time() + seconds
        discord_ts = f"<t:{int(end_time)}:R>"
        discord_ts_full = f"<t:{int(end_time)}:f>"
        host = interaction.user

        desc = (self.description.value + "\n\n") if self.description.value else ""
        desc += f"**Ends:** {discord_ts} ({discord_ts_full})\n"
        desc += f"**Hosted by:** {host.mention}\n**Entries:** 0\n**Winners:** {winner_count}"
        embed = discord.Embed(
            title=self.prize.value,
            description=desc,
            color=discord.Color.blurple()
        )

        await interaction.response.send_message("✅ ギブアウェイを作成しました！", ephemeral=True)
        msg = await interaction.channel.send(embed=embed, view=GiveawayPanelView())

        settings = load_settings()
        guild_id = str(interaction.guild.id)
        if "giveaways" not in settings:
            settings["giveaways"] = {}
        if guild_id not in settings["giveaways"]:
            settings["giveaways"][guild_id] = {}
        settings["giveaways"][guild_id][str(msg.id)] = {
            "prize": self.prize.value,
            "description": self.description.value or "",
            "winners": winner_count,
            "end_time": end_time,
            "host_id": str(host.id),
            "channel_id": str(interaction.channel.id),
            "entries": [],
            "ended": False
        }
        save_settings(settings)


@tree.command(name="giv", description="ギブアウェイを作成します")
async def create_giveaway(interaction: discord.Interaction):
    await interaction.response.send_modal(GiveawayModal())

# ==================== チケットクローズ確認ビュー ====================

class CloseConfirmView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="🗑️ チケットを削除する", style=discord.ButtonStyle.danger, custom_id="close_confirm_btn")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
        owner_id = next((uid for uid, cid in open_tickets.items() if cid == str(interaction.channel.id)), None)
        if str(interaction.user.id) != owner_id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("⚠️ チケット作成者または管理者のみ閉じられます。", ephemeral=True)
            return
        await interaction.response.send_message("🔒 チケットを閉じます...")
        await send_ticket_log(interaction.channel, interaction.guild, guild_id, owner_id, settings)
        if owner_id and owner_id in open_tickets:
            del open_tickets[owner_id]
            settings["ticket"][guild_id]["open_tickets"] = open_tickets
            save_settings(settings)
        await asyncio.sleep(2)
        await interaction.channel.delete()

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary, custom_id="close_cancel_btn")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.message.delete()
        await interaction.response.send_message("キャンセルしました。", ephemeral=True)

# ==================== 偽パネル ====================

class NisePanelView(discord.ui.View):
    def __init__(self, button_label: str = "受け取る", message: str = "受け取りました！"):
        super().__init__(timeout=None)
        self.button_label = button_label
        self.message = message
        self.add_item(NisePanelButton(button_label, message))

class NisePanelButton(discord.ui.Button):
    def __init__(self, label: str = "受け取る", message: str = "受け取りました！"):
        super().__init__(label=label, style=discord.ButtonStyle.success, custom_id="nise_panel_btn_main")
        self.msg = message

    async def callback(self, interaction: discord.Interaction):
        settings = load_settings()
        msg = settings.get("nise_messages", {}).get(str(interaction.guild.id), self.msg)
        await interaction.response.send_message(msg, ephemeral=True)

# ==================== カジノ ====================

import random as _casino_random

SLOT_SYMBOLS = ["🍒", "🍋", "🍇", "⭐", "💎", "7️⃣"]
SLOT_PAYOUTS = {"💎💎💎": 2000, "7️⃣7️⃣7️⃣": 5000, "⭐⭐⭐": 1500,
                "🍇🍇🍇": 800, "🍋🍋🍋": 600, "🍒🍒🍒": 400}

async def update_kjnpn(guild):
    settings = load_settings()
    guild_id = str(guild.id)
    ch_id = settings.get("kjnpn", {}).get(guild_id)
    msg_id = settings.get("kjnpn_msg", {}).get(guild_id)
    if not ch_id or not msg_id:
        return
    ch = guild.get_channel(int(ch_id))
    if not ch:
        return
    try:
        msg = await ch.fetch_message(int(msg_id))
    except Exception:
        return

    role_id = settings.get("kjn_role", {}).get(guild_id)
    balances = settings.get("casino", {}).get(guild_id, {})
    entries = []
    for uid, bal in sorted(balances.items(), key=lambda x: -x[1]):
        member = guild.get_member(int(uid))
        if not member:
            continue
        if role_id and not any(str(r.id) == role_id for r in member.roles):
            continue
        entries.append(member.display_name + "\n``" + str(bal) + "円``")
    body = "\n\n".join(entries) if entries else "データなし"
    await msg.edit(content=f"💰 **残高一覧**\n\n{body}")

def get_balance(settings, guild_id, user_id):
    return settings.get("casino", {}).get(guild_id, {}).get(str(user_id), 1500)

def set_balance(settings, guild_id, user_id, amount):
    if "casino" not in settings:
        settings["casino"] = {}
    if guild_id not in settings["casino"]:
        settings["casino"][guild_id] = {}
    settings["casino"][guild_id][str(user_id)] = max(-2000, amount)
    save_settings(settings)

def can_use_casino(message, settings):
    guild_id = str(message.guild.id)
    role_id = settings.get("kjn_role", {}).get(guild_id)
    if not role_id:
        return False
    return any(str(r.id) == role_id for r in message.author.roles)


class BetModal(discord.ui.Modal, title="賭け金を入力"):
    amount = discord.ui.TextInput(label="賭け金 (1以上の整数)", placeholder="例: 200", required=True)

    async def on_submit(self, interaction: discord.Interaction):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        try:
            bet = int(self.amount.value)
            if bet <= 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("❌ 正しい金額を入力してください。", ephemeral=True)
            return

        bal = get_balance(settings, guild_id, interaction.user.id)
        if bal - bet < -2000:
            await interaction.response.send_message(f"💸 残高不足！現在の残高: **{bal}円**", ephemeral=True)
            return

        base_symbols = ["🍒", "🍋", "🍇", "⭐", "💎", "7️⃣"]
        weights = [60, 55, 50, 45, 35, 30]

        reel1 = _casino_random.choices(base_symbols, weights=weights)[0]
        idx = base_symbols.index(reel1)
        if bet >= 300000:
            match_chance = 0.74
        elif bet >= 10000:
            match_chance = 0.66
        elif bet >= 5000:
            match_chance = 0.61
        elif bet >= 2000:
            match_chance = 0.56
        elif bet >= 1000:
            match_chance = 0.54
        elif bet >= 500:
            match_chance = 0.48
        elif bet >= 200:
            match_chance = 0.42
        else:
            match_chance = 0.38

        if _casino_random.random() < match_chance:
            reels = [reel1, reel1, reel1]
        else:
            reels = [reel1]
            for _ in range(2):
                others = [s for s in base_symbols if s != reel1]
                reels.append(_casino_random.choice(others))
            _casino_random.shuffle(reels)

        result = "".join(reels)

        sym_multipliers = {
            "💎💎💎": 6.0,
            "7️⃣7️⃣7️⃣": 5.0,
            "⭐⭐⭐": 4.0,
            "🍇🍇🍇": 3.0,
            "🍋🍋🍋": 2.0,
            "🍒🍒🍒": 1.5,
        }
        sym_mult = sym_multipliers.get(result, 0)
        payout = int(bet * sym_mult) if sym_mult > 0 else 0
        new_bal = bal - bet + payout
        set_balance(settings, guild_id, interaction.user.id, new_bal)
        last_bets[interaction.user.id] = bet
        asyncio.ensure_future(update_kjnpn(interaction.guild))

        if payout > 0:
            txt = f"**{reels[0]} {reels[1]} {reels[2]}**\n🎉 当たり！ **+{payout}円** ({sym_mult}×)\n残高: **{new_bal}円**"
        else:
            txt = f"**{reels[0]} {reels[1]} {reels[2]}**\n😢 ハズレ… -{bet}円\n残高: **{new_bal}円**"
        try:
            await interaction.response.send_message(txt, ephemeral=True)
        except Exception:
            pass


class CasinoPanelView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="🎰 スロットを回す", style=discord.ButtonStyle.primary, custom_id="casino_play")
    async def play(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        role_id = settings.get("kjn_role", {}).get(guild_id)
        if role_id and not any(str(r.id) == role_id for r in interaction.user.roles):
            await interaction.response.send_message("⚠️ このコマンドを使用できるロールがありません。", ephemeral=True)
            return
        bal = get_balance(settings, guild_id, interaction.user.id)
        txt2 = "💰 現在の残高: **" + str(bal) + "円**\n貭け金を入力してください。"
        await interaction.response.send_message(txt2, ephemeral=True, view=BetButtonView())

    @discord.ui.button(label="💰 残高確認", style=discord.ButtonStyle.secondary, custom_id="casino_balance")
    async def balance(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        bal = get_balance(settings, str(interaction.guild.id), interaction.user.id)
        await interaction.response.send_message(f"💰 あなたの残高: **{bal}円**", ephemeral=True)


class BetButtonView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="金額を入力して賭ける", style=discord.ButtonStyle.success, custom_id="casino_bet_input")
    async def bet(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(BetModal())
        except Exception:
            pass


class GifView(discord.ui.View):
    def __init__(self, amount: int = 500):
        super().__init__(timeout=60)
        self.participants = []
        self.amount = amount

    @discord.ui.button(label="🎁 参加する", style=discord.ButtonStyle.success, custom_id="gif_join")
    async def join(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id in self.participants:
            await interaction.response.send_message("すでに参加しています！", ephemeral=True)
            return
        self.participants.append(interaction.user.id)
        await interaction.response.send_message(f"✅ 参加しました！現在 {len(self.participants)} 人", ephemeral=True)

    @discord.ui.button(label="🎰 抽選する", style=discord.ButtonStyle.primary, custom_id="gif_draw")
    async def draw(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("管理者のみ抽選できます。", ephemeral=True)
            return
        if not self.participants:
            await interaction.response.send_message("参加者がいません！", ephemeral=True)
            return
        winner_id = _casino_random.choice(self.participants)
        winner = interaction.guild.get_member(winner_id)
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        bal = get_balance(settings, guild_id, winner_id)
        set_balance(settings, guild_id, winner_id, bal + self.amount)
        asyncio.ensure_future(update_kjnpn(interaction.guild))
        self.stop()
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(
            content="\U0001f389 **抽選結果！**\n" + winner.mention + " が当選！500円GET！\n残高: " + str(bal+500) + "円",
        )

# ==================== チケット ====================

async def send_ticket_log(channel, guild, guild_id, owner_id, settings):
    from datetime import timezone
    log_channel_id = settings.get("ticket", {}).get(guild_id, {}).get("log_channel_id")
    if not log_channel_id:
        return
    log_channel = guild.get_channel(int(log_channel_id))
    if not log_channel:
        return

    lines = [f"=== チケットログ: #{channel.name} ==="]
    owner = guild.get_member(int(owner_id)) if owner_id else None
    lines.append(f"作成者: {owner.display_name if owner else owner_id}")
    roblox_id = settings.get("ticket", {}).get(guild_id, {}).get("ticket_data", {}).get(str(owner_id), {}).get("roblox_id", "未入力")
    lines.append(f"Roblox ID: {roblox_id}")
    lines.append("")

    async for msg in channel.history(limit=500, oldest_first=True):
        if msg.author.bot and msg.content in ("🔒 チケットを閉じます...",):
            continue
        ts = msg.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        attachments = " [添付ファイル]" if msg.attachments else ""
        lines.append(f"[{ts}] {msg.author.display_name}: {msg.content}{attachments}")

    log_text = "\n".join(lines)
    import re as _re
    safe_name = _re.sub(r"[^a-zA-Z0-9_-]", "_", channel.name)
    file = discord.File(io.BytesIO(log_text.encode("utf-8-sig")), filename=f"ticket-{safe_name}.txt")
    await log_channel.send(f"\U0001f4cb チケットが閉じられました: `{channel.name}`", file=file)


class TicketCreateButton(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="🎫 チケットを作成", style=discord.ButtonStyle.primary, custom_id="ticket_create")
    async def create_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        ticket_cfg = settings.get("ticket", {}).get(guild_id, {})
        category_id = ticket_cfg.get("category_id")
        mention_role_id = ticket_cfg.get("mention_role_id")

        existing = ticket_cfg.get("open_tickets", {}).get(str(interaction.user.id))
        if existing:
            ch = interaction.guild.get_channel(int(existing))
            if ch:
                await interaction.response.send_message(f"⚠️ すでにチケットがあります: {ch.mention}", ephemeral=True)
                return

        category = interaction.guild.get_channel(int(category_id)) if category_id else None
        overwrites = {
            interaction.guild.default_role: discord.PermissionOverwrite(view_channel=False),
            interaction.user: discord.PermissionOverwrite(view_channel=True, send_messages=True),
            interaction.guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True),
        }
        if mention_role_id:
            role = interaction.guild.get_role(int(mention_role_id))
            if role:
                overwrites[role] = discord.PermissionOverwrite(view_channel=True, send_messages=True)

        ch = await interaction.guild.create_text_channel(
            f"ticket-{interaction.user.name}",
            category=category,
            overwrites=overwrites
        )

        if "ticket" not in settings:
            settings["ticket"] = {}
        if guild_id not in settings["ticket"]:
            settings["ticket"][guild_id] = {}
        if "open_tickets" not in settings["ticket"][guild_id]:
            settings["ticket"][guild_id]["open_tickets"] = {}
        settings["ticket"][guild_id]["open_tickets"][str(interaction.user.id)] = str(ch.id)
        save_settings(settings)

        await interaction.response.send_message(f"✅ チケットを作成しました: {ch.mention}", ephemeral=True)

        mention_txt = ""
        if mention_role_id:
            role = interaction.guild.get_role(int(mention_role_id))
            if role:
                mention_txt = role.mention

        desc = f"{interaction.user.mention} のチケットです。\nまずRoblox IDを入力してください。"
        embed = discord.Embed(title="🎫 チケット", description=desc, color=discord.Color.blue())
        await ch.send(content=mention_txt if mention_txt else None, embed=embed, view=TicketPanelView(str(interaction.user.id)))


class RobloxIDModal(discord.ui.Modal, title="Roblox IDを入力"):
    roblox_id = discord.ui.TextInput(label="Roblox ID", placeholder="hanakuso", required=True)

    def __init__(self, user_id: str):
        super().__init__()
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        if "ticket" not in settings:
            settings["ticket"] = {}
        if guild_id not in settings["ticket"]:
            settings["ticket"][guild_id] = {}
        if "ticket_data" not in settings["ticket"][guild_id]:
            settings["ticket"][guild_id]["ticket_data"] = {}
        settings["ticket"][guild_id]["ticket_data"][self.user_id] = {"roblox_id": self.roblox_id.value}
        save_settings(settings)

        await interaction.response.defer()

        username = self.roblox_id.value.strip()
        avatar_url = None
        display_name = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://users.roblox.com/v1/usernames/users",
                    json={"usernames": [username], "excludeBannedUsers": False}
                ) as resp:
                    data = await resp.json()
                    users = data.get("data", [])
                    if users:
                        roblox_user_id = users[0]["id"]
                        display_name = users[0].get("displayName", username)
                        async with session.get(
                            f"https://thumbnails.roblox.com/v1/users/avatar-headshot?userIds={roblox_user_id}&size=150x150&format=Png"
                        ) as thumb_resp:
                            thumb_data = await thumb_resp.json()
                            thumb_list = thumb_data.get("data", [])
                            if thumb_list:
                                avatar_url = thumb_list[0].get("imageUrl")
        except Exception:
            pass

        embed = discord.Embed(title="商品を選択してください", color=discord.Color.green())
        embed.add_field(name="Roblox ID", value=username)
        if display_name:
            embed.add_field(name="表示名", value=display_name)
        if avatar_url:
            embed.set_thumbnail(url=avatar_url)
        await interaction.edit_original_response(embed=embed, view=ProductSelectView(self.user_id, username))


class LuaPurchaseView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="🛒 ゲームパスを購入", url="https://www.roblox.com/game-pass/1833026954", style=discord.ButtonStyle.link, row=0))

    @discord.ui.button(label="🔒 チケットを閉じる", style=discord.ButtonStyle.danger, custom_id="lua_ticket_close", row=1)
    async def close_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
        owner_id = next((uid for uid, cid in open_tickets.items() if cid == str(interaction.channel.id)), None)
        if str(interaction.user.id) != owner_id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("⚠️ チケット作成者または管理者のみ閉じられます。", ephemeral=True)
            return
        await interaction.response.send_message("🔒 チケットを閉じます...")
        await send_ticket_log(interaction.channel, interaction.guild, guild_id, owner_id, settings)
        if owner_id and owner_id in open_tickets:
            del open_tickets[owner_id]
            settings["ticket"][guild_id]["open_tickets"] = open_tickets
            save_settings(settings)
        await asyncio.sleep(2)
        await interaction.channel.delete()

def get_roblox_id_from_channel(interaction: discord.Interaction):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
    owner_id = next((uid for uid, cid in open_tickets.items() if cid == str(interaction.channel.id)), None)
    if owner_id:
        return settings.get("ticket", {}).get(guild_id, {}).get("ticket_data", {}).get(owner_id, {}).get("roblox_id", "不明")
    return "不明"


class HubPriceView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        plans = [
            ("Lifetime  3000R$", "https://www.roblox.com/game-pass/1852936342", "hub_lifetime"),
            ("3 Year  2500R$",   "https://www.roblox.com/game-pass/1854598312", "hub_3year"),
            ("1 Year  2000R$",   "https://www.roblox.com/game-pass/1852702354", "hub_1year"),
            ("5 Month  1500R$",  "https://www.roblox.com/game-pass/1853974313", "hub_5month"),
            ("2 Month  1200R$",  "https://www.roblox.com/game-pass/1852672344", "hub_2month"),
        ]
        for label, url, cid in plans:
            self.add_item(discord.ui.Button(label=label, url=url, style=discord.ButtonStyle.link))
        self.add_item(discord.ui.Button(label="🔒 チケットを閉じる", style=discord.ButtonStyle.danger, custom_id="hub_close"))

    async def interaction_check(self, interaction: discord.Interaction):
        if interaction.data.get("custom_id") == "hub_close":
            settings = load_settings()
            guild_id = str(interaction.guild.id)
            open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
            owner_id = next((uid for uid, cid in open_tickets.items() if cid == str(interaction.channel.id)), None)
            if str(interaction.user.id) != owner_id and not interaction.user.guild_permissions.administrator:
                await interaction.response.send_message("⚠️ チケット作成者または管理者のみ閉じられます。", ephemeral=True)
                return False
            await interaction.response.send_message("🔒 チケットを閉じます...")
            await send_ticket_log(interaction.channel, interaction.guild, guild_id, owner_id, settings)
            if owner_id and owner_id in open_tickets:
                del open_tickets[owner_id]
                settings["ticket"][guild_id]["open_tickets"] = open_tickets
                save_settings(settings)
            await asyncio.sleep(2)
            await interaction.channel.delete()
            return False
        return True


class LuaPriceView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="1 Month  400R$", url="https://www.roblox.com/game-pass/1853116337", style=discord.ButtonStyle.link, row=0))
        self.add_item(discord.ui.Button(label="2 Month  500R$", url="https://www.roblox.com/game-pass/1833026954", style=discord.ButtonStyle.link, row=0))
        self.add_item(discord.ui.Button(label="🔒 チケットを閉じる", style=discord.ButtonStyle.danger, custom_id="lua_price_close", row=1))

    async def interaction_check(self, interaction: discord.Interaction):
        cid = interaction.data.get("custom_id")
        if cid == "lua_price_close":
            settings = load_settings()
            guild_id = str(interaction.guild.id)
            open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
            owner_id = next((uid for uid, cid2 in open_tickets.items() if cid2 == str(interaction.channel.id)), None)
            if str(interaction.user.id) != owner_id and not interaction.user.guild_permissions.administrator:
                await interaction.response.send_message("⚠️ チケット作成者または管理者のみ閉じられます。", ephemeral=True)
                return False
            await interaction.response.send_message("🔒 チケットを閉じます...")
            await send_ticket_log(interaction.channel, interaction.guild, guild_id, owner_id, settings)
            if owner_id and owner_id in open_tickets:
                del open_tickets[owner_id]
                settings["ticket"][guild_id]["open_tickets"] = open_tickets
                save_settings(settings)
            await asyncio.sleep(2)
            await interaction.channel.delete()
            return False
        return True


class ProductSelectView(discord.ui.View):
    def __init__(self, user_id: str = None, roblox_id: str = None):
        super().__init__(timeout=None)
        self.user_id = user_id
        self.roblox_id = roblox_id

    @discord.ui.button(label="DoDo HUB", style=discord.ButtonStyle.secondary, custom_id="product_hub")
    async def select_hub(self, interaction: discord.Interaction, button: discord.ui.Button):
        rid = self.roblox_id or get_roblox_id_from_channel(interaction)
        embed = discord.Embed(title="DoDo HUB  プランを選択してください", color=discord.Color.green())
        embed.add_field(name="Roblox ID", value=rid)
        embed.add_field(name="商品", value="DoDo HUB")
        embed.add_field(name="​", value="購入したら写真を送ってください。", inline=False)
        await interaction.response.edit_message(embed=embed, view=HubPriceView())

    @discord.ui.button(label="DoDo HUB lua", style=discord.ButtonStyle.primary, custom_id="product_lua")
    async def select_lua(self, interaction: discord.Interaction, button: discord.ui.Button):
        rid = self.roblox_id or get_roblox_id_from_channel(interaction)
        embed = discord.Embed(title="DoDo HUB lua  プランを選択してください", color=discord.Color.blue())
        embed.add_field(name="Roblox ID", value=rid)
        embed.add_field(name="商品", value="DoDo HUB lua")
        embed.add_field(name="​", value="購入したら写真を送ってください。", inline=False)
        await interaction.response.edit_message(embed=embed, view=LuaPriceView())


class TicketPanelView(discord.ui.View):
    def __init__(self, user_id: str = None):
        super().__init__(timeout=None)
        self.user_id = user_id

    def get_owner(self, interaction: discord.Interaction):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
        return next((uid for uid, cid in open_tickets.items() if cid == str(interaction.channel.id)), None)

    @discord.ui.button(label="📝 Roblox IDを入力", style=discord.ButtonStyle.primary, custom_id="ticket_input_roblox")
    async def input_roblox(self, interaction: discord.Interaction, button: discord.ui.Button):
        owner_id = self.get_owner(interaction)
        if str(interaction.user.id) != owner_id:
            await interaction.response.send_message("⚠️ チケット作成者のみ入力できます。", ephemeral=True)
            return
        await interaction.response.send_modal(RobloxIDModal(owner_id))

    @discord.ui.button(label="🔒 チケットを閉じる", style=discord.ButtonStyle.danger, custom_id="ticket_close")
    async def close_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        settings = load_settings()
        guild_id = str(interaction.guild.id)
        open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
        owner_id = self.get_owner(interaction)
        if str(interaction.user.id) != owner_id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("⚠️ チケット作成者または管理者のみ閉じられます。", ephemeral=True)
            return
        if owner_id and owner_id in open_tickets:
            del open_tickets[owner_id]
            settings["ticket"][guild_id]["open_tickets"] = open_tickets
            save_settings(settings)
        await interaction.response.send_message("🔒 チケットを閉じます...")
        await asyncio.sleep(3)
        await interaction.channel.delete()


@tree.command(name="ticket", description="チケット作成パネルを設置します（管理者のみ）")
@app_commands.describe(
    カテゴリー="チケットチャンネルを作成するカテゴリー",
    メンションロール="チケット作成時にメンションするロール（省略可）",
    ログチャンネル="閉じた時のログを送るチャンネル（省略可）"
)
@app_commands.checks.has_permissions(administrator=True)
async def setup_ticket(interaction: discord.Interaction, カテゴリー: discord.CategoryChannel, メンションロール: discord.Role = None, ログチャンネル: discord.TextChannel = None):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "ticket" not in settings:
        settings["ticket"] = {}
    if guild_id not in settings["ticket"]:
        settings["ticket"][guild_id] = {}
    settings["ticket"][guild_id]["category_id"] = str(カテゴリー.id)
    if メンションロール:
        settings["ticket"][guild_id]["mention_role_id"] = str(メンションロール.id)
    if ログチャンネル:
        settings["ticket"][guild_id]["log_channel_id"] = str(ログチャンネル.id)
    save_settings(settings)

    embed = discord.Embed(
        title="🎫 チケット",
        description="下のボタンを押してチケットを作成してください。",
        color=discord.Color.blue()
    )
    await interaction.channel.send(embed=embed, view=TicketCreateButton())
    await interaction.response.send_message("✅ チケットパネルを設置しました。", ephemeral=True)

@setup_ticket.error
async def ticket_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="bl", description="自動返信しないチャンネルを設定します（管理者のみ）")
@app_commands.describe(
    チャンネル="対象チャンネル",
    解除="設定を解除する場合はTrue"
)
@app_commands.checks.has_permissions(administrator=True)
async def set_bl(interaction: discord.Interaction, チャンネル: discord.TextChannel, 解除: bool = False):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "bl_channels" not in settings:
        settings["bl_channels"] = {}
    if guild_id not in settings["bl_channels"]:
        settings["bl_channels"][guild_id] = []

    cid = str(チャンネル.id)
    if 解除:
        if cid in settings["bl_channels"][guild_id]:
            settings["bl_channels"][guild_id].remove(cid)
            save_settings(settings)
            await interaction.response.send_message(f"✅ {チャンネル.mention} のブロックを解除しました。", ephemeral=True)
        else:
            await interaction.response.send_message(f"⚠️ {チャンネル.mention} は設定されていません。", ephemeral=True)
    else:
        if cid not in settings["bl_channels"][guild_id]:
            settings["bl_channels"][guild_id].append(cid)
        save_settings(settings)
        await interaction.response.send_message(f"✅ {チャンネル.mention} を自動返信しないチャンネルに設定しました。", ephemeral=True)

@set_bl.error
async def bl_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="kjn", description="カジノを使えるロールを設定します（管理者のみ）")
@app_commands.describe(ロール="カジノ使用を許可するロール")
@app_commands.checks.has_permissions(administrator=True)
async def set_kjn_role(interaction: discord.Interaction, ロール: discord.Role):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "kjn_role" not in settings:
        settings["kjn_role"] = {}
    settings["kjn_role"][guild_id] = str(ロール.id)
    save_settings(settings)
    await interaction.response.send_message(f"✅ `{ロール.name}` をカジノ使用ロールに設定しました。", ephemeral=True)

@tree.command(name="zn", description="全員の残高を確認します（自分にのみ表示）")
async def show_balances(interaction: discord.Interaction):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    role_id = settings.get("kjn_role", {}).get(guild_id)
    balances = settings.get("casino", {}).get(guild_id, {})

    lines = []
    for uid, bal in sorted(balances.items(), key=lambda x: -x[1]):
        member = interaction.guild.get_member(int(uid))
        if not member:
            continue
        if role_id and not any(str(r.id) == role_id for r in member.roles):
            continue
        lines.append(f"{member.display_name}: **{bal}円**")

    if not lines:
        await interaction.response.send_message("残高データがありません。", ephemeral=True)
        return
    txt3 = "💰 **残高一覧**\n" + "\n".join(lines)
    await interaction.response.send_message(txt3, ephemeral=True)

@tree.command(name="bi", description="スロットの倍率を変更します（管理者のみ）")
@app_commands.describe(倍率="掛け金に掛ける倍率（例: 1, 1.5, 2）")
@app_commands.checks.has_permissions(administrator=True)
async def set_mult(interaction: discord.Interaction, 倍率: str):
    try:
        m = float(倍率)
        if m <= 0:
            raise ValueError
    except ValueError:
        await interaction.response.send_message("⚠️ 正しい数値を入力してください。", ephemeral=True)
        return
    settings = load_settings()
    if "slot_mult" not in settings:
        settings["slot_mult"] = {}
    settings["slot_mult"][str(interaction.guild.id)] = m
    save_settings(settings)
    await interaction.response.send_message(f"✅ 倍率を **{m}倍** に設定しました。", ephemeral=True)

@set_mult.error
async def bi_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="kjnpn", description="残高をリアルタイム表示するチャンネルを設定します（管理者のみ）")
@app_commands.describe(チャンネル="表示するチャンネル")
@app_commands.checks.has_permissions(administrator=True)
async def set_kjnpn(interaction: discord.Interaction, チャンネル: discord.TextChannel):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "kjnpn" not in settings:
        settings["kjnpn"] = {}
    settings["kjnpn"][guild_id] = str(チャンネル.id)
    save_settings(settings)

    msg = await チャンネル.send("💰 **残高一覧** (更新中...)")
    settings["kjnpn_msg"] = settings.get("kjnpn_msg", {})
    settings["kjnpn_msg"][guild_id] = str(msg.id)
    save_settings(settings)
    await interaction.response.send_message(f"✅ {チャンネル.mention} に残高パネルを設置しました。", ephemeral=True)
    await update_kjnpn(interaction.guild)

@set_kjnpn.error
async def kjnpn_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="kjnn", description="カジノパネルを設置します（管理者のみ）")
@app_commands.checks.has_permissions(administrator=True)
async def setup_casino(interaction: discord.Interaction):
    desc = "ボタンを押してスロットを回そう！多く賭けるほど当たりやすくなります！"
    embed = discord.Embed(title="🎰 カジノ", description=desc, color=discord.Color.gold())
    embed.add_field(name="💰 賭け金×倍率", value="💎💎💎 ×6.0 | 7️⃣7️⃣7️⃣ ×5.0 | ⭐⭐⭐ ×4.0 | 🍇🍇🍇 ×3.0 | 🍋🍋🍋 ×2.0 | 🍒🍒🍒 ×1.5", inline=False)
    embed.add_field(name="🎲 当たり確率", value="~199円:38% | 200~:42% | 500~:48% | 1000~:54% | 2000~:56% | 5000~:61% | 10000~:66% | 300000~:74%", inline=False)
    embed.add_field(name="🎰 シンボル確率", value="🍒 60% | 🍋 55% | 🍇 50% | ⭐ 45% | 7️⃣ 35% | 💎 30%", inline=False)
    await interaction.channel.send(embed=embed, view=CasinoPanelView())
    await interaction.response.send_message("✅ カジノパネルを設置しました。", ephemeral=True)

@setup_casino.error
async def kjnn_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@set_kjn_role.error
async def kjn_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="sn", description="メンバー参加通知チャンネルを設定します（管理者のみ）")
@app_commands.describe(チャンネル="通知を送るチャンネル", メッセージ="通知メッセージ（{user}で名前に置換）")
@app_commands.checks.has_permissions(administrator=True)
async def set_join_notify(interaction: discord.Interaction, チャンネル: discord.TextChannel, メッセージ: str = "{user}が来たぞ！"):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "notify" not in settings:
        settings["notify"] = {}
    if guild_id not in settings["notify"]:
        settings["notify"][guild_id] = {}
    settings["notify"][guild_id]["join_channel"] = str(チャンネル.id)
    settings["notify"][guild_id]["join_msg"] = メッセージ
    save_settings(settings)
    await interaction.response.send_message("✅ 参加通知を " + チャンネル.mention + " に設定しました。\nメッセージ: `" + メッセージ + "`", ephemeral=True)

@set_join_notify.error
async def sn_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="ts", description="メンバー退出通知チャンネルを設定します（管理者のみ）")
@app_commands.describe(チャンネル="通知を送るチャンネル", メッセージ="通知メッセージ（{user}で名前に置換）")
@app_commands.checks.has_permissions(administrator=True)
async def set_leave_notify(interaction: discord.Interaction, チャンネル: discord.TextChannel, メッセージ: str = "{user}が退出しました！"):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "notify" not in settings:
        settings["notify"] = {}
    if guild_id not in settings["notify"]:
        settings["notify"][guild_id] = {}
    settings["notify"][guild_id]["leave_channel"] = str(チャンネル.id)
    settings["notify"][guild_id]["leave_msg"] = メッセージ
    save_settings(settings)
    await interaction.response.send_message("✅ 退出通知を " + チャンネル.mention + " に設定しました。\nメッセージ: `" + メッセージ + "`", ephemeral=True)

@set_leave_notify.error
async def ts_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="by", description="URLをbypassします")
@app_commands.describe(url="bypassするURL")
async def bypass_url(interaction: discord.Interaction, url: str):
    await interaction.response.defer()
    import urllib.parse, json as _json
    encoded = urllib.parse.quote(url, safe="")

    async def try_get(session, api_url):
        async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            text = await resp.text()
            try:
                data = _json.loads(text)
            except Exception:
                return None
            return data.get("destination") or data.get("result") or data.get("url") or data.get("bypassed") or data.get("link")

    async def try_post(session, api_url, payload, headers=None):
        async with session.post(api_url, json=payload, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            text = await resp.text()
            try:
                data = _json.loads(text)
            except Exception:
                return None
            return data.get("destination") or data.get("result") or data.get("url") or data.get("bypassed") or data.get("link") or data.get("bypassed_url")

    async with aiohttp.ClientSession() as session:
        def valid(r):
            return r and "discord" not in r.lower() and "shut down" not in r.lower() and r.startswith("http")

        endpoints = [
            ("GET", "https://dlr.kys.gay/api/free/bypass?url=" + encoded, None, None),
            ("GET", "https://bypass.bot.nu/api/bypass?url=" + encoded, None, None),
            ("GET", "https://api.bypass.city/bypass?url=" + encoded, None, None),
        ]
        bypass_key = os.environ.get("BYPASS_API_KEY", "")
        if bypass_key:
            endpoints.append(("POST", "https://api.bypass.tools/api/v1/bypass/direct", {"url": url}, {"x-api-key": bypass_key}))

        for method, ep, payload, headers in endpoints:
            try:
                if method == "GET":
                    r = await try_get(session, ep)
                else:
                    r = await try_post(session, ep, payload, headers)
                if valid(r):
                    await interaction.followup.send("✅ Bypass成功！\n" + r)
                    return
            except Exception:
                continue

    await interaction.followup.send("⚠️ Bypass失敗しました。このURLは対応していない可能性があります。")

@tree.command(name="nisepanel", description="偽物のパネルを作成します")
@app_commands.describe(
    名前="貰えるものの名前",
    詳細="貰えるものの詳細",
    ボタン名="ボタンの名前",
    メッセージ="ボタンを押した人にだけ表示するメッセージ"
)
async def nise_panel(interaction: discord.Interaction, 名前: str, 詳細: str, ボタン名: str, メッセージ: str):
    settings = load_settings()
    if "nise_messages" not in settings:
        settings["nise_messages"] = {}
    settings["nise_messages"][str(interaction.guild.id)] = メッセージ
    save_settings(settings)
    embed = discord.Embed(title=名前, description=詳細, color=discord.Color.green())
    view = NisePanelView(ボタン名, メッセージ)
    await interaction.channel.send(embed=embed, view=view)
    await interaction.response.send_message("✅ パネルを作成しました。", ephemeral=True)

@tree.command(name="kes", description="自動削除キーワードを設定します（管理者のみ）")
@app_commands.describe(
    キーワード="含まれていたら削除するキーワード",
    削除="このキーワードの設定を削除する場合はTrue"
)
@app_commands.checks.has_permissions(administrator=True)
async def set_kes(interaction: discord.Interaction, キーワード: str, 削除: bool = False):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "kes" not in settings:
        settings["kes"] = {}
    if guild_id not in settings["kes"]:
        settings["kes"][guild_id] = []

    if 削除:
        if キーワード in settings["kes"][guild_id]:
            settings["kes"][guild_id].remove(キーワード)
            save_settings(settings)
            await interaction.response.send_message(f"✅ `{キーワード}` を自動削除リストから削除しました。", ephemeral=True)
        else:
            await interaction.response.send_message(f"⚠️ `{キーワード}` は設定されていません。", ephemeral=True)
        return

    if キーワード not in settings["kes"][guild_id]:
        settings["kes"][guild_id].append(キーワード)
    save_settings(settings)
    await interaction.response.send_message(f"✅ `{キーワード}` を自動削除キーワードに追加しました。", ephemeral=True)

@set_kes.error
async def kes_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="keslist", description="自動削除キーワード一覧を表示します（管理者のみ）")
@app_commands.checks.has_permissions(administrator=True)
async def kes_list(interaction: discord.Interaction):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    keywords = settings.get("kes", {}).get(guild_id, [])
    if not keywords:
        await interaction.response.send_message("自動削除キーワードは設定されていません。", ephemeral=True)
        return
    text = "\n".join(f"・`{kw}`" for kw in keywords)
    await interaction.response.send_message(f"🗑️ **自動削除キーワード一覧**\n{text}", ephemeral=True)

@kes_list.error
async def keslist_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

@tree.command(name="jisin", description="地震速報を送るチャンネルを設定します（管理者のみ）")
@app_commands.describe(チャンネル="通知を送るチャンネル")
@app_commands.checks.has_permissions(administrator=True)
async def set_jisin(interaction: discord.Interaction, チャンネル: discord.TextChannel):
    settings = load_settings()
    guild_id = str(interaction.guild.id)
    if "jisin" not in settings:
        settings["jisin"] = {}
    settings["jisin"][guild_id] = str(チャンネル.id)
    save_settings(settings)
    await interaction.response.send_message(f"✅ 地震速報を {チャンネル.mention} に設定しました。", ephemeral=True)

@set_jisin.error
async def jisin_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("⚠️ このコマンドは管理者のみ使えます。", ephemeral=True)

# ==================== プレフィックスコマンド ====================

_last_jisin_id = set()
_jisin_initialized = False

def _parse_jisin(quake: dict):
    """p2pquake APIから地震情報をembedで返す"""
    eq = quake.get("earthquake", {})
    hypo = eq.get("hypocenter", {})
    place = hypo.get("name", "不明")
    mag_val = hypo.get("magnitude", -1)
    depth_val = hypo.get("depth", -1)
    shindo_str = eq.get("maxScale", -1)
    time_str = eq.get("time", "")
    domestic_tsunami = eq.get("domesticTsunami", "None")

    shindo_map = {
        10: "1", 20: "2", 30: "3", 40: "4",
        45: "5弱", 50: "5強", 55: "6弱", 60: "6強", 70: "7"
    }
    shindo_disp = shindo_map.get(shindo_str, "不明")
    mag_disp = str(mag_val) if mag_val not in (-1, None) else "不明"
    depth_disp = f"{depth_val}km" if depth_val not in (-1, None) else "不明"

    # 津波情報
    tsunami_map = {
        "None": "この地震による津波の心配はありません。",
        "Unknown": "津波の有無は不明です。",
        "Checking": "津波の有無を確認中です。",
        "NonEffective": "若干の海面変動があるかもしれません。",
        "Watch": "⚠️ 津波注意報が発表されています。",
        "Warning": "🚨 津波警報が発表されています！",
    }
    tsunami_disp = tsunami_map.get(domestic_tsunami, "")

    # 震度で色を変える
    if shindo_str is not None and shindo_str >= 55:
        color = discord.Color.red()
    elif shindo_str is not None and shindo_str >= 40:
        color = discord.Color.orange()
    else:
        color = discord.Color.blue()

    # 発生時刻を整形
    title_time = time_str.replace("-", "/") if time_str else ""

    embed = discord.Embed(
        title="地震情報",
        description=f"{title_time}頃、地震がありました。\n\n{tsunami_disp}",
        color=color
    )
    embed.add_field(name="震央", value=place, inline=True)
    embed.add_field(name="深さ", value=depth_disp, inline=True)
    embed.add_field(name="マグニチュード", value=f"M{mag_disp}", inline=True)
    embed.add_field(name="最大震度", value=shindo_disp, inline=False)
    embed.set_footer(text="気象庁")

    # 震源地の緯度経度で地図画像URL（国土地理院タイル）
    lat = hypo.get("latitude", None)
    lon = hypo.get("longitude", None)
    if lat and lon and lat != -200 and lon != -200:
        map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=7&size=600x300&markers=color:red%7C{lat},{lon}"
        # Google Maps APIキー不要の代替: OpenStreetMap
        map_url = f"https://static-maps.yandex.ru/1.x/?lang=en_US&ll={lon},{lat}&z=7&l=map&size=600,300&pt={lon},{lat},pm2rdm"
        embed.set_image(url=map_url)

    return embed

async def jisin_task():
    global _last_jisin_id, _jisin_initialized
    await client.wait_until_ready()

    # 起動時: 取得できるまでリトライして必ず既存IDを全部既読にする
    while not _jisin_initialized:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.p2pquake.net/v2/history?codes=551&limit=30",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for q in data:
                            qid = q.get("id") or q.get("_id", "")
                            if qid:
                                _last_jisin_id.add(str(qid))
                        _jisin_initialized = True
                    else:
                        await asyncio.sleep(10)
        except Exception:
            await asyncio.sleep(10)

    while not client.is_closed():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.p2pquake.net/v2/history?codes=551&limit=5",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(30)
                        continue
                    quakes = await resp.json()

            settings = load_settings()
            jisin_channels = settings.get("jisin", {})

            for quake in reversed(quakes):  # 古い順に処理
                qid = str(quake.get("id") or quake.get("_id", ""))
                if not qid or qid in _last_jisin_id:
                    continue

                _last_jisin_id.add(qid)
                if len(_last_jisin_id) > 300:
                    _last_jisin_id = set(list(_last_jisin_id)[-300:])

                if not jisin_channels:
                    continue

                embed = _parse_jisin(quake)

                for guild_id, ch_id in jisin_channels.items():
                    guild = client.get_guild(int(guild_id))
                    if not guild:
                        continue
                    ch = guild.get_channel(int(ch_id))
                    if not ch:
                        continue
                    try:
                        await ch.send(embed=embed)
                    except Exception:
                        pass

        except Exception:
            pass
        await asyncio.sleep(30)

@client.event
async def on_ready():
    client.add_view(CasinoPanelView())
    client.add_view(NisePanelView())
    client.add_view(BetButtonView())
    client.add_view(TicketCreateButton())
    client.add_view(LuaPurchaseView())
    client.add_view(TicketPanelView())
    client.add_view(ProductSelectView())
    client.add_view(HubPriceView())
    client.add_view(LuaPriceView())
    client.add_view(GiveawayPanelView())
    client.add_view(CloseConfirmView())
    await tree.sync()
    settings = load_settings()
    for gid in settings.get("autoreply_guilds", []):
        autoreply_guilds.add(int(gid))
    for gid in settings.get("snitch_guilds", []):
        snitch_guilds.add(int(gid))
    for gid in settings.get("ikari_guilds", []):
        ikari_guilds.add(int(gid))
    print(f"✅ 起動しました: {client.user}")
    client.loop.create_task(jisin_task())
    client.loop.create_task(giveaway_timer_task())
    client.loop.create_task(_cache_cleanup_task())

@client.event
async def on_member_join(member: discord.Member):
    settings = load_settings()
    guild_id = str(member.guild.id)
    notify = settings.get("notify", {}).get(guild_id, {})
    ch_id = notify.get("join_channel")
    msg = notify.get("join_msg", "{user}が来たぞ！")
    if ch_id:
        ch = member.guild.get_channel(int(ch_id))
        if ch:
            await ch.send(msg.replace("{user}", member.mention))

@client.event
async def on_member_remove(member: discord.Member):
    settings = load_settings()
    guild_id = str(member.guild.id)
    notify = settings.get("notify", {}).get(guild_id, {})
    ch_id = notify.get("leave_channel")
    msg = notify.get("leave_msg", "{user}が退出しました！")
    if ch_id:
        ch = member.guild.get_channel(int(ch_id))
        if ch:
            await ch.send(msg.replace("{user}", member.mention))

# ==================== 削除メッセージ監視 ====================
# キャッシュ: {message_id: {"content": str, "author_id": int, "guild_id": int, "channel_id": int, "time": float}}
_msg_cache = {}

@client.event
async def on_message(message):
    if message.author.bot:
        # BOT自身のメッセージIDを登録してsnitchしないようにする
        if client.user and message.author.id == client.user.id:
            _bot_sent_ids.add(message.id)
        return
    if message.guild is None:
        await message.channel.send("このBOTへのメッセージは確認できません。")
        return
    # ikariチェック（最速削除・awaitで即実行）
    if message.guild and message.guild.id in ikari_guilds and not message.author.guild_permissions.administrator:
        import re as _ik
        has_link = bool(_ik.search(r'https?://\S+|discord\.gg/\S+', message.content or ""))
        has_attachment = bool(message.attachments)
        if has_link or has_attachment:
            _bot_deleted_ids.add(message.id)
            try:
                await message.delete()
            except Exception:
                pass
            return

    # キャッシュに保存（5時間後に消える）。コマンドは保存しない
    attachments_urls = [a.url for a in message.attachments] if message.attachments else []
    _is_command = message.content.strip().startswith(("?", "？", "!", "！")) if message.content else False
    if _is_command:
        _bot_deleted_ids.add(message.id)
    elif message.content or attachments_urls:
        _msg_cache[message.id] = {
            "content": message.content,
            "author_id": message.author.id,
            "guild_id": message.guild.id,
            "channel_id": message.channel.id,
            "attachments": attachments_urls,
            "time": _time.time()
        }
    asyncio.get_event_loop().create_task(_handle_message(message))

# snitch ON状態のギルドID
snitch_guilds = set()
# ikari（リンク・画像自動削除）ON状態のギルドID
ikari_guilds = set()
# BOTが自分で消したメッセージID（snitchしない）
_bot_deleted_ids = set()
# BOTが送信したメッセージID（snitchしない）
_bot_sent_ids = set()

@client.event
async def on_message_delete(message):
    if not message.guild:
        return
    if message.guild.id not in snitch_guilds:
        return

    # BOTが送信・削除したメッセージは通知しない
    if message.id in _bot_deleted_ids:
        _bot_deleted_ids.discard(message.id)
        return
    if message.id in _bot_sent_ids:
        _bot_sent_ids.discard(message.id)
        return

    # authorがBOT（キャッシュあり・なし両対応）
    if message.author and message.author.bot:
        return
    # client.userと照合（最も確実）
    if client.user and message.author and message.author.id == client.user.id:
        return

    # discord.pyキャッシュのcontentでコマンド判定（最速）
    raw_content = message.content
    if raw_content and raw_content.strip().startswith(("?", "？", "!", "！")):
        return

    content = raw_content
    author = message.author
    attachments = list(message.attachments)  # discord.pyキャッシュから

    # 自前キャッシュから補完
    cached = _msg_cache.get(message.id)
    if cached:
        if not content:
            content = cached.get("content", "")
        member = message.guild.get_member(cached["author_id"])
        if member:
            author = member
        # キャッシュのauthor_idがBOT自身なら通知しない
        if client.user and cached["author_id"] == client.user.id:
            return
        # 添付ファイルURLをキャッシュから取得（discord側のURLが切れていても対応）
        if not attachments:
            cached_urls = cached.get("attachments", [])
        else:
            cached_urls = []
    else:
        cached_urls = []

    if author and author.bot:
        return

    # テキストも画像も何もなければスルー
    has_content = bool(content)
    has_attachments = bool(attachments) or bool(cached_urls)
    if not has_content and not has_attachments:
        return

    # コマンドは送らない
    if content and content.strip().startswith(("?", "？", "!", "！")):
        return

    mention = author.mention if author else "不明"

    # テキスト+画像を一緒に送る
    import io
    files = []
    for att in attachments:
        try:
            async with aiohttp.ClientSession() as _s:
                async with _s.get(att.proxy_url or att.url) as _r:
                    if _r.status == 200:
                        data = await _r.read()
                        files.append(discord.File(io.BytesIO(data), filename=att.filename))
        except Exception:
            pass

    text = f"{mention} : {content}" if has_content else f"{mention} が削除した画像"

    try:
        if files:
            await message.channel.send(text, files=files)
        elif has_content:
            await message.channel.send(text)
    except Exception:
        # ファイル送信失敗時はテキストだけ送る
        try:
            if has_content:
                await message.channel.send(text)
        except Exception:
            pass

    # キャッシュ内のURLのみの場合（ダウンロード失敗時）
    for url in cached_urls:
        try:
            await message.channel.send(f"{mention} が削除した画像: {url}")
        except Exception:
            pass

@client.event
async def on_message_edit(before, after):
    """編集は無視（削除検知の誤爆防止）"""
    pass

# BOT自身が送ったメッセージをすべて_bot_sent_idsに登録
_original_send = None
async def _track_bot_message(coro):
    msg = await coro
    if msg and hasattr(msg, "id"):
        _bot_sent_ids.add(msg.id)
    return msg

@client.event
async def on_raw_message_delete(payload):
    """キャッシュにないメッセージが消えた場合も_bot_sent_idsでガード"""
    pass

async def _cache_cleanup_task():
    while True:
        await asyncio.sleep(600)
        now = _time.time()
        expired = [mid for mid, data in _msg_cache.items() if now - data["time"] > 18000]
        for mid in expired:
            _msg_cache.pop(mid, None)
        if len(_bot_sent_ids) > 1000:
            to_remove = list(_bot_sent_ids)[:500]
            for mid in to_remove:
                _bot_sent_ids.discard(mid)

async def _handle_message(message):
    if not message.guild:
        return

    guild_id = str(message.guild.id)
    settings = load_settings()

    settings = load_settings()
    guild_id = str(message.guild.id) if message.guild else None
    if guild_id and message.attachments:
        open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
        ticket_owner = next((uid for uid, cid in open_tickets.items() if cid == str(message.channel.id)), None)
        if ticket_owner and str(message.author.id) == ticket_owner:
            mention_role_id = settings.get("ticket", {}).get(guild_id, {}).get("mention_role_id")
            if mention_role_id:
                role = message.guild.get_role(int(mention_role_id))
                if role:
                    await message.channel.send(f"{role.mention} 購入確認の写真が届きました！")

    # ?ikari - リンク・画像自動削除ON/OFF
    if message.content.strip() in ("?ikari", "？ikari"):
        if not message.author.guild_permissions.administrator:
            await message.reply("⚠️ このコマンドは管理者のみ使えます。")
            return
        gid = message.guild.id
        settings = load_settings()
        if gid in ikari_guilds:
            ikari_guilds.discard(gid)
            ig_list = settings.get("ikari_guilds", [])
            if str(gid) in ig_list:
                ig_list.remove(str(gid))
            settings["ikari_guilds"] = ig_list
            msg = await message.reply("🔴 リンク・画像自動削除をOFFにしました。")
        else:
            ikari_guilds.add(gid)
            ig_list = settings.get("ikari_guilds", [])
            if str(gid) not in ig_list:
                ig_list.append(str(gid))
            settings["ikari_guilds"] = ig_list
            msg = await message.reply("🟢 リンク・画像自動削除をONにしました。")
        save_settings(settings)
        _bot_deleted_ids.add(message.id)
        await message.delete()
        await asyncio.sleep(10)
        _bot_deleted_ids.add(msg.id)
        try:
            await msg.delete()
        except Exception:
            pass
        return

    # ?k - 削除メッセージ表示ON/OFF
    if message.content.strip() in ("?k", "？k"):
        gid = message.guild.id
        settings = load_settings()
        if gid in snitch_guilds:
            snitch_guilds.discard(gid)
            sg_list = settings.get("snitch_guilds", [])
            if str(gid) in sg_list:
                sg_list.remove(str(gid))
            settings["snitch_guilds"] = sg_list
            msg = await message.reply("🔴 削除メッセージ表示をOFFにしました。")
        else:
            snitch_guilds.add(gid)
            sg_list = settings.get("snitch_guilds", [])
            if str(gid) not in sg_list:
                sg_list.append(str(gid))
            settings["snitch_guilds"] = sg_list
            msg = await message.reply("🟢 削除メッセージ表示をONにしました。")
        save_settings(settings)
        _bot_deleted_ids.add(message.id)

        await message.delete()
        await asyncio.sleep(10)
        try:

            await msg.delete()

        except Exception:

            pass
        return

    # 自動削除キーワードチェック
    if message.content:
        kes_keywords = settings.get("kes", {}).get(guild_id, [])
        for kw in kes_keywords:
            if kw.lower() in message.content.lower():
                _bot_deleted_ids.add(message.id)
                try:
                    await message.delete()
                except Exception:
                    pass
                try:
                    warn = await message.channel.send(
                        f"⚠️ {message.author.mention} のメッセージを自動削除しました。"
                    )
                    _bot_sent_ids.add(warn.id)
                    await asyncio.sleep(5)
                    _bot_deleted_ids.add(warn.id)
                    try:
                        await warn.delete()
                    except Exception:
                        pass
                except Exception:
                    pass
                return

    # ?jo - 自動返信モデル確認
    if message.content.strip() == "?jo":
        _bot_deleted_ids.add(message.id)

        await message.delete()
        settings2 = load_settings()
        yak = settings2.get("yak", {}).get(str(message.guild.id), {})
        if yak:
            lines_txt = []
            for uid, style in yak.items():
                member = message.guild.get_member(int(uid))
                name = member.display_name if member else f"不明({uid})"
                lines_txt.append(f"{name}: {style}")
            text = message.author.mention + " ⚙️ 自動返信モード一覧\n" + "\n".join(lines_txt) + "\n*(このメッセージは10秒後に消えます)*"
        else:
            text = message.author.mention + " ⚙️ モード設定なし\n*(このメッセージは10秒後に消えます)*"
        msg = await message.channel.send(text)
        await asyncio.sleep(10)
        try:

            await msg.delete()

        except Exception:

            pass
        return

    # !kjn - カジノパネル
    if message.content.strip() == "!kjn":
        settings = load_settings()
        if not can_use_casino(message, settings):
            await message.reply("⚠️ このコマンドを使用できるロールがありません。")
            return
        desc = "ボタンを押してスロットを回そう！貭け金は自分で入力できます。多く貭けるほど当たりやすくなります！"
        embed = discord.Embed(title="🎰 カジノ", description=desc, color=discord.Color.gold())
        embed.add_field(name="💰 貭け金×倍率", value="💎💎💎 ×5.0 | 7️⃣7️⃣7️⃣ ×4.0 | ⭐⭐⭐ ×3.0 | 🍇🍇🍇 ×2.0 | 🍋🍋🍋 ×1.0 | 🍒🍒🍒 ×0.5")
        await message.channel.send(embed=embed, view=CasinoPanelView())
        _bot_deleted_ids.add(message.id)

        await message.delete()
        return

    # !zn+ @user 金額
    if message.content.startswith("!zn+ "):
        if not message.author.guild_permissions.administrator:
            await message.reply("⚠️ このコマンドは管理者のみ使えます。")
            return
        parts = message.content.split()
        if len(parts) < 3 or not message.mentions:
            await message.reply("使い方: `!zn+ @ユーザー 金額`")
            return
        target = message.mentions[0]
        try:
            amount = int(parts[-1])
            if amount <= 0:
                raise ValueError
        except ValueError:
            await message.reply("金額は1以上の整数で指定してください。")
            return
        settings = load_settings()
        guild_id = str(message.guild.id)
        cur = get_balance(settings, guild_id, target.id)
        set_balance(settings, guild_id, target.id, cur + amount)
        asyncio.ensure_future(update_kjnpn(message.guild))
        await message.reply(target.mention + " に **" + str(amount) + "円** 付与しました！残高: **" + str(cur + amount) + "円**")
        return

    # !zn- @user 金額
    if message.content.startswith("!zn- "):
        if not message.author.guild_permissions.administrator:
            await message.reply("⚠️ このコマンドは管理者のみ使えます。")
            return
        parts = message.content.split()
        if len(parts) < 3 or not message.mentions:
            await message.reply("使い方: `!zn- @ユーザー 金額`")
            return
        target = message.mentions[0]
        try:
            amount = int(parts[-1])
            if amount <= 0:
                raise ValueError
        except ValueError:
            await message.reply("金額は1以上の整数で指定してください。")
            return
        settings = load_settings()
        guild_id = str(message.guild.id)
        cur = get_balance(settings, guild_id, target.id)
        set_balance(settings, guild_id, target.id, cur - amount)
        await message.reply(target.mention + " の残高を **" + str(amount) + "円** 減らしました。残高: **" + str(cur - amount) + "円**")
        asyncio.ensure_future(update_kjnpn(message.guild))
        return

    # !v @user 金額 - 送金
    if message.content.startswith("!v "):
        settings = load_settings()
        guild_id = str(message.guild.id)
        parts = message.content.split()
        if len(parts) < 3 or not message.mentions:
            await message.reply("使い方: `!v @ユーザー 金額`")
            return
        target = message.mentions[0]
        try:
            amount = int(parts[-1])
        except ValueError:
            await message.reply("金額は整数で指定してください。")
            return
        if amount <= 0:
            await message.reply("1円以上を指定してください。")
            return
        sender_bal = get_balance(settings, guild_id, message.author.id)
        if sender_bal < amount:
            await message.reply(f"💸 残高が足りません！現在の残高: **{sender_bal}円**")
            return
        target_bal = get_balance(settings, guild_id, target.id)
        set_balance(settings, guild_id, message.author.id, sender_bal - amount)
        set_balance(settings, guild_id, target.id, target_bal + amount)
        asyncio.ensure_future(update_kjnpn(message.guild))
        txt = "✅ " + target.mention + " に **" + str(amount) + "円** 送金しました！\nあなたの残高: **" + str(sender_bal - amount) + "円**"
        await message.reply(txt)
        return

    # ?givk @ユーザー - ギブアウェイから指定ユーザーを除外（管理者のみ）
    if message.content.startswith("?givk") and message.mentions:
        if not message.author.guild_permissions.administrator:
            await message.reply("⚠️ このコマンドは管理者のみ使えます。")
            return
        _bot_deleted_ids.add(message.id)
        await message.delete()
        target = message.mentions[0]
        settings = load_settings()
        guild_id = str(message.guild.id)
        giveaways = settings.get("giveaways", {}).get(guild_id, {})
        active = {k: v for k, v in giveaways.items() if not v.get("ended") and _time.time() < v.get("end_time", 0)}
        if not active:
            tmp = await message.channel.send("⚠️ 進行中のギブアウェイはありません。")
            await asyncio.sleep(5)
            try:

                await tmp.delete()

            except Exception:

                pass
            return
        removed = []
        for msg_id, gw in active.items():
            entries = gw.get("entries", [])
            if str(target.id) in entries:
                entries.remove(str(target.id))
                gw["entries"] = entries
                settings["giveaways"][guild_id][msg_id] = gw
                removed.append(gw["prize"])
                # パネルのembedを更新
                ch = message.guild.get_channel(int(gw.get("channel_id", 0)))
                if ch:
                    try:
                        panel_msg = await ch.fetch_message(int(msg_id))
                        await _update_giveaway_message(panel_msg, gw, message.guild)
                    except Exception:
                        pass
        if removed:
            save_settings(settings)
            tmp = await message.channel.send(f"✅ {target.mention} を **{'**, **'.join(removed)}** から除外しました。")
        else:
            tmp = await message.channel.send(f"⚠️ {target.mention} は進行中のギブアウェイに参加していません。")
        await asyncio.sleep(5)
        _bot_deleted_ids.add(tmp.id)
        try:

            await tmp.delete()

        except Exception:

            pass
        return

    # ?giv - 進行中ギブアウェイの参加者を表示
    if message.content.strip() == "?giv":
        _bot_deleted_ids.add(message.id)
        try:
            await message.delete()
        except Exception:
            pass
        settings = load_settings()
        guild_id = str(message.guild.id)
        giveaways = settings.get("giveaways", {}).get(guild_id, {})
        active = {k: v for k, v in giveaways.items() if not v.get("ended") and _time.time() < v.get("end_time", 0)}
        if not active:
            tmp = await message.channel.send("現在進行中のギブアウェイはありません。")
            await asyncio.sleep(10)
            try:

                await tmp.delete()

            except Exception:

                pass
            return
        lines = []
        for msg_id, gw in active.items():
            entries = gw.get("entries", [])
            entry_mentions = " ".join(f"<@{uid}>" for uid in entries) if entries else "まだ誰も参加していません"
            lines.append(f"**{gw['prize']}** ({len(entries)}人)\n{entry_mentions}")
        tmp = await message.channel.send("🎉 **進行中のギブアウェイ参加者**\n\n" + "\n\n".join(lines) + "\n\n*（このメッセージは10秒後に消えます）*")
        await asyncio.sleep(10)
        try:

            await tmp.delete()

        except Exception:

            pass
        return

    # ?gif - ランダムプレゼント
    if message.content.startswith("?gif"):
        parts = message.content.split()
        gif_amount = 500
        if len(parts) >= 2:
            try:
                gif_amount = int(parts[1])
            except ValueError:
                pass
        desc = "ボタンを押して参加しよう！\n当選者に **" + str(gif_amount) + "円** プレゼント！"
        embed = discord.Embed(title="🎁 プレゼント抽選！", description=desc, color=discord.Color.gold())
        await message.channel.send(embed=embed, view=GifView(gif_amount))
        _bot_deleted_ids.add(message.id)

        await message.delete()
        return

    # ?close - チケットを閉じる確認パネル
    if message.content.strip() == "?close":
        settings = load_settings()
        guild_id = str(message.guild.id)
        open_tickets = settings.get("ticket", {}).get(guild_id, {}).get("open_tickets", {})
        # チケットチャンネルかチェック
        owner_id = next((uid for uid, cid in open_tickets.items() if cid == str(message.channel.id)), None)
        if not owner_id:
            await message.reply("⚠️ ここはチケットチャンネルではありません。", delete_after=5)
            _bot_deleted_ids.add(message.id)
            await message.delete()
            return
        _bot_deleted_ids.add(message.id)
        await message.delete()
        embed = discord.Embed(
            title="🎫 チケットを閉じますか？",
            description="削除ボタンを押すとチケットが閉じられます。",
            color=discord.Color.red()
        )
        await message.channel.send(embed=embed, view=CloseConfirmView())
        return

    # ?kick @ユーザー 理由 - キック（理由省略可）
    if message.content.startswith("?kick ") and message.mentions:
        if not message.author.guild_permissions.kick_members:
            await message.reply("⚠️ キック権限がありません。")
            return
        target = message.mentions[0]
        import re as _kickre
        raw = message.content[6:].strip()
        reason_text = _kickre.sub(r"<@!?\d+>", "", raw).strip()
        reason = f"{reason_text}（by {message.author.name}）" if reason_text else f"KICKby{message.author.name}"
        try:
            await target.kick(reason=reason)
            reason_disp = f"　理由: {reason_text}" if reason_text else ""
            await message.reply(f"👢 {target.mention} をキックしました。{reason_disp}")
        except Exception as e:
            await message.reply(f"⚠️ エラー: {e}")
        return

    # ?b @ユーザー 理由 - BAN（理由省略可）
    if message.content.startswith("?b ") and message.mentions:
        if not message.author.guild_permissions.ban_members:
            await message.reply("⚠️ BAN権限がありません。")
            return
        target = message.mentions[0]
        # メンション部分を除いた残りを理由にする
        raw = message.content[3:].strip()
        # メンションテキストを除去
        import re as _bre
        reason_text = _bre.sub(r"<@!?\d+>", "", raw).strip()
        reason = f"{reason_text}（by {message.author.name}）" if reason_text else f"BANby{message.author.name}"
        try:
            await target.ban(reason=reason)
            reason_disp = f"　理由: {reason_text}" if reason_text else ""
            await message.reply(f"🔨 {target.mention} をBANしました。{reason_disp}")
        except Exception as e:
            await message.reply(f"⚠️ エラー: {e}")
        return

    # ?t @ユーザー 時間 理由 - タイムアウト（理由省略可）
    if message.content.startswith("?t ") and message.mentions:
        if not message.author.guild_permissions.moderate_members:
            await message.reply("⚠️ タイムアウト権限がありません。")
            return
        import re as _tre
        # メンションと時間を抽出
        parts = message.content.split()
        # 時間文字列を探す（m/h/d で終わるもの）
        time_str = None
        time_idx = None
        for i, p in enumerate(parts):
            if p.lower().endswith(("m", "h", "d")) and p[:-1].isdigit():
                time_str = p.lower()
                time_idx = i
                break
        if not time_str:
            await message.reply("使い方: `?t @ユーザー 時間 理由` (例: ?t @user 1h 荒らし)")
            return
        seconds = 0
        if time_str.endswith("m"):
            seconds = int(time_str[:-1]) * 60
        elif time_str.endswith("h"):
            seconds = int(time_str[:-1]) * 3600
        elif time_str.endswith("d"):
            seconds = int(time_str[:-1]) * 86400
        seconds = min(seconds, 28 * 86400)
        # 時間より後の部分を理由にする
        reason_text = " ".join(parts[time_idx+1:]).strip() if time_idx is not None else ""
        reason = f"{reason_text}（by {message.author.name}）" if reason_text else f"TOby{message.author.name}"
        target = message.mentions[0]
        from datetime import timedelta, timezone, datetime
        until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        try:
            await target.timeout(until, reason=reason)
            reason_disp = f"　理由: {reason_text}" if reason_text else ""
            await message.reply(f"⏰ {target.mention} を {time_str} タイムアウトしました。{reason_disp}")
        except Exception as e:
            await message.reply(f"⚠️ エラー: {e}")
        return

    # ?j - 自動返信ON/OFF
    if message.content.strip() in ("?j", "？j", "?ｊ", "？ｊ"):
        gid = message.guild.id
        settings = load_settings()
        if gid in autoreply_guilds:
            autoreply_guilds.discard(gid)
            ar_list = settings.get("autoreply_guilds", [])
            if str(gid) in ar_list:
                ar_list.remove(str(gid))
            settings["autoreply_guilds"] = ar_list
            msg = await message.reply("🔴 自動返信をOFFにしました。")
        else:
            autoreply_guilds.add(gid)
            ar_list = settings.get("autoreply_guilds", [])
            if str(gid) not in ar_list:
                ar_list.append(str(gid))
            settings["autoreply_guilds"] = ar_list
            msg = await message.reply("🟢 自動返信をONにしました。")
        save_settings(settings)
        _bot_deleted_ids.add(message.id)

        await message.delete()
        await asyncio.sleep(10)
        try:

            await msg.delete()

        except Exception:

            pass
        return

    # 自動返信処理
    open_tickets_all = settings.get("ticket", {}).get(str(message.guild.id), {}).get("open_tickets", {})
    if any(cid == str(message.channel.id) for cid in open_tickets_all.values()):
        pass
    bl_channels = settings.get("bl_channels", {}).get(str(message.guild.id), [])
    if message.guild.id in autoreply_guilds and not message.content.strip().startswith(("?", "？")) and not any(cid == str(message.channel.id) for cid in open_tickets_all.values()) and str(message.channel.id) not in bl_channels:
        has_human_mention = message.mentions and any(not m.bot for m in message.mentions)
        ref = message.reference
        is_reply_to_human = False
        is_reply_to_bot = False
        is_bot_mention = message.mentions and all(m.bot for m in message.mentions)
        if ref:
            ref_msg = ref.resolved
            if ref_msg is None:
                try:
                    ref_msg = await message.channel.fetch_message(ref.message_id)
                except Exception:
                    ref_msg = None
            if ref_msg and hasattr(ref_msg, "author"):
                if ref_msg.author.bot:
                    is_reply_to_bot = True
                else:
                    is_reply_to_human = True
        if has_human_mention or is_reply_to_human:
            return
        if True:
            import random
            import unicodedata
            from google.genai import types
            ai = get_ai_client()
            mes_dict = settings.get("mes", {}).get(guild_id, {})
            matched = None
            for keyword, replies in mes_dict.items():
                if keyword.lower() in message.content.lower():
                    matched = random.choice(replies)
                    break

            if matched:
                _sent = await message.reply(matched)
                _bot_sent_ids.add(_sent.id)
            else:
                impersonate_target = None
                impersonate_name = None
                has_nani_kw = False
                nani_keywords = ["なにしてる", "何してる", "なにやってる", "何やってる", "いまなに", "今何", "なにしてん", "何してん"]
                for kw in nani_keywords:
                    if kw in message.content:
                        has_nani_kw = True
                        if is_bot_mention or is_reply_to_bot:
                            has_nani_kw = False
                            break
                        for member in message.guild.members:
                            if member.bot:
                                continue
                            for name in [member.display_name, member.name]:
                                idx_name = message.content.find(name)
                                idx_kw = message.content.find(kw)
                                if idx_name != -1 and idx_kw != -1 and idx_name < idx_kw:
                                    impersonate_target = member
                                    impersonate_name = name
                                    break
                            if impersonate_target:
                                break
                        break

                if has_nani_kw and not impersonate_target and not is_bot_mention and not is_reply_to_bot:
                    return

                ai_model = "gemini-3.1-flash-lite"
                cfg = types.GenerateContentConfig(
                    max_output_tokens=120
                )

                if impersonate_target:
                    system = f"Discordで友達に{impersonate_name}が何してるか聞かれた。「たぶん〇〇してないの絵文字」みたいな超短い一言で答える。推測口調。絵文字1個だけ。句読点なし。"
                    cfg.system_instruction = system
                    cfg.max_output_tokens = 60
                    response = await ai.aio.models.generate_content(
                        model="gemini-3.1-flash-lite",
                        contents=[{"role": "user", "parts": [{"text": message.content}]}],
                        config=cfg
                    )
                    if response.text and response.text.strip():
                        try:
                            _sent = await message.reply(response.text.strip()[:60])
                            _bot_sent_ids.add(_sent.id)
                        except Exception:
                            try:
                                _sent = await message.channel.send(response.text.strip()[:60])
                                _bot_sent_ids.add(_sent.id)
                            except Exception:
                                pass
                else:
                    yak_style = settings.get("yak", {}).get(guild_id, {}).get(str(message.author.id))

                    jp_chars = sum(1 for c in message.content if unicodedata.east_asian_width(c) in ('W', 'F', 'H') and ord(c) > 127)
                    en_words = [w for w in re.findall(r"[a-zA-Z]+", message.content) if len(w) >= 2]
                    is_english = jp_chars == 0 and len(en_words) >= 1

                    if is_english:
                        if yak_style:
                            sys_prompt = f"You ARE a real human on Discord. Your personality: {yak_style}. STRICT RULES: text casually like a real person, no punctuation at end of sentences, typos are fine, keep replies short for short messages, max 1 question per reply, NEVER sound like AI or assistant, stay in character as {yak_style} at ALL times."
                        else:
                            sys_prompt = "you're a real person on discord. text like a normal human - casual, no punctuation at end, typos ok, short replies for short messages. never ask multiple questions. don't sound like AI."
                    else:
                        if yak_style:
                            sys_prompt = f"あなたは今からDiscordでチャットしてるリアルな人間です。絶対に守るルール：キャラクター「{yak_style}」として振る舞う、改行しない、句読点なし、ため口、質問は1回まで、AIっぽい言い方禁止、アシスタントっぽい返答禁止。「{yak_style}」のキャラを絶対に崩さないで。"
                        else:
                            sys_prompt = "discordでチャットしてる高校生。改行絶対しない。句読点なし。ため口。wやwwや笑で感情表現。質問は1つまで。AIっぽさ禁止。"
                    cfg.system_instruction = sys_prompt
                    cfg.max_output_tokens = 200

                    if message.attachments and message.attachments[0].content_type and message.attachments[0].content_type.startswith("image/"):
                        img_url = message.attachments[0].url
                        async with aiohttp.ClientSession() as _sess:
                            async with _sess.get(img_url) as _resp:
                                img_bytes = await _resp.read()
                                import base64
                                img_b64 = base64.b64encode(img_bytes).decode()
                                ct = message.attachments[0].content_type
                        contents = [{"role": "user", "parts": [{"inline_data": {"mime_type": ct, "data": img_b64}}, {"text": message.content or "この画像について一言"}]}]
                    else:
                        contents = [{"role": "user", "parts": [{"text": message.content or "…"}]}]

                    hist = autoreply_histories.get(message.author.id, [])
                    full_contents = hist + contents if hist else contents
                    response = None
                    for _retry in range(3):
                        try:
                            response = await ai.aio.models.generate_content(
                                model="gemini-3.1-flash-lite",
                                contents=full_contents,
                                config=cfg
                            )
                            break
                        except Exception as _e:
                            if ("503" in str(_e) or "unavailable" in str(_e).lower()) and _retry < 2:
                                await asyncio.sleep(2)
                                continue
                            raise
                    if response and response.text:
                        import re as _re
                        reply_text = response.text.strip()
                        reply_text = _re.sub(r"\n{3,}", "\n", reply_text)
                        if reply_text:
                            hist = autoreply_histories.setdefault(message.author.id, [])
                            hist.append({"role": "user", "parts": [{"text": message.content}]})
                            hist.append({"role": "model", "parts": [{"text": reply_text}]})
                            if len(hist) > 20:
                                autoreply_histories[message.author.id] = hist[-20:]
                            try:
                                _sent = await message.reply(reply_text[:1000])
                                _bot_sent_ids.add(_sent.id)
                            except Exception:
                                try:
                                    _sent = await message.channel.send(reply_text[:1000])
                                    _bot_sent_ids.add(_sent.id)
                                except Exception:
                                    pass
        return

    # ?rol
    if message.content == "?rol":
        allowed = settings.get(guild_id, {}).get("allowed_roles", [])
        if not allowed:
            text = "📋 現在、使用可能なロールが設定されていません（誰も使えません）。"
        else:
            role_names = []
            for rid in allowed:
                role = message.guild.get_role(int(rid))
                role_names.append(f"・{role.name}" if role else f"・削除済みロール({rid})")
            text = "📋 **BOT使用可能ロール**\n" + "\n".join(role_names)
        _bot_deleted_ids.add(message.id)

        await message.delete()
        msg = await message.channel.send(f"{message.author.mention}\n{text}\n*（このメッセージは10秒後に消えます）*")
        await asyncio.sleep(10)
        try:

            await msg.delete()

        except Exception:

            pass
        return

    # ?mod
    if message.content == "?mod":
        cfg = get_user_settings(message.author.id)
        m = cfg.get("model", DEFAULT_MODEL)
        t = cfg.get("thinking", DEFAULT_THINKING)
        text = f"⚙️ **あなたの現在の設定**\nモデル: `{m}`\n思考レベル: `{t}`\n\n変更するには `/mod` を使ってください。"
        _bot_deleted_ids.add(message.id)

        await message.delete()
        msg = await message.channel.send(f"{message.author.mention}\n{text}\n*（このメッセージは10秒後に消えます）*")
        await asyncio.sleep(10)
        try:

            await msg.delete()

        except Exception:

            pass
        return

    # ?nan - コードを難読化
    if message.content.startswith("?nan"):
        lua_code = ""

        if message.attachments:
            att = message.attachments[0]
            if att.filename.endswith((".lua", ".txt")):
                lua_bytes = await att.read()
                lua_code = lua_bytes.decode("utf-8", errors="ignore")
            else:
                await message.reply("⚠️ `.lua` または `.txt` ファイルを添付してください。")
                return
        else:
            raw = message.content[4:].strip()
            pattern = r"```(?:\w+)?\n(.*?)```"
            code_match = re.search(pattern, raw, re.DOTALL)
            lua_code = code_match.group(1).strip() if code_match else raw

        if not lua_code:
            await message.reply("使い方: `?nan コード` または `?nan` にファイルを添付してください。")
            return

        status_msg = await message.reply("🔒 難読化中...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://wearedevs.net/api/obfuscate",
                    json={"script": lua_code},
                    headers={"Content-Type": "application/json", "Referer": "https://wearedevs.net/obfuscator"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        obfuscated = data.get("code") or data.get("result") or data.get("obfuscated")
                        if obfuscated:
                            file = discord.File(io.BytesIO(obfuscated.encode("utf-8")), filename="obfuscated.lua")
                            await status_msg.edit(content="✅ 難読化完了！")
                            await message.channel.send(file=file)
                        else:
                            await status_msg.edit(content=f"⚠️ 難読化失敗: `{str(data)[:200]}`")
                    else:
                        body = await resp.text()
                        await status_msg.edit(content=f"⚠️ サーバーエラー ({resp.status}): `{body[:200]}`")
        except Exception as e:
            await status_msg.edit(content=f"⚠️ エラー: `{e}`")
        return

    is_new = message.content.startswith("?ai ")
    is_cont = message.content.startswith("?a ")
    is_code_only = message.content.startswith("?aii ")
    if not is_new and not is_cont and not is_code_only:
        return

    if not can_use_bot(message):
        await message.reply("⚠️ あなたのロールはBOTを使用できません。")
        return

    if is_code_only:
        query = message.content[5:].strip()
        query = f"以下の内容のコードだけを作成してください。説明や解説は不要です。コードのみ返してください。\n\n{query}"
        reply_msg = await message.reply("⚙️ コード生成中...")
        try:
            answer = await ask_gemini_stream(message.channel.id, query, reply_msg, new_conversation=True, user_id=message.author.id)
            code, ext = extract_code(answer)
            if not code:
                code = answer.strip()
                ext = "txt"
            run_commands = {"py": "python code.py", "js": "node code.js", "ts": "ts-node code.ts", "sh": "bash code.sh"}
            run_cmd = run_commands.get(ext)
            run_msg = f"\n```\n{run_cmd}\n```" if run_cmd else ""
            file = discord.File(io.BytesIO(code.encode("utf-8")), filename=f"code.{ext}")
            await reply_msg.edit(content=f"📎 コード生成完了{run_msg}")
            await message.channel.send(file=file)
        except Exception as e:
            await reply_msg.edit(content=f"⚠️ エラー: {e}")
        return

    query = message.content[4:].strip() if is_new else message.content[3:].strip()

    reply_msg = await message.reply("🤖 考え中...")

    try:
        answer = await ask_gemini_stream(message.channel.id, query, reply_msg, new_conversation=is_new, user_id=message.author.id)

        code, ext = extract_code(answer)
        text = remove_code_blocks(answer) if code else answer
        if len(text) > 1900:
            text = text[:1900] + "..."

        prefix = "🤖" if is_new else "🔁"

        if code:
            file = discord.File(io.BytesIO(code.encode("utf-8")), filename=f"code.{ext}")
            await reply_msg.edit(content=f"{prefix} {text}")
            await message.channel.send(file=file)
        else:
            await reply_msg.edit(content=f"{prefix} {text}")

    except Exception as e:
        await reply_msg.edit(content=f"⚠️ エラー: {e}")

# ============================================================
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# ============================================================

client.run(DISCORD_TOKEN)
