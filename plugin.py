"""自我认知记忆插件

让Bot拥有自己的观点、偏好和对参与过的讨论的记忆，以及对群友的认知画像。

工作流程：
- 提示注入（零API调用）：从两类槽静态读取已召回记忆注入上下文
    - 话题记忆槽（max 2）：Bot自身对话题的观点、参与过的讨论
    - 角色画像槽（max 4）：Bot对特定群友的了解
- LLM主动刷新：发现注入内容与当前话题/人物不符时，调用对应刷新工具（AGENT）
- LLM搜索后形成观点：对时效性话题调用「搜索网络」获取信息，再入库
- LLM写入：对话中形成明确认知时，调用「记录认知」（BEHAVIOR）
- 主动监控：15分钟内群聊有未回复的讨论时，插件主动触发LLM判断是否参与
"""

import asyncio
import datetime
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx
from pydantic import Field
from qdrant_client import models as qdrant_models

from nekro_agent.api import core as nekro_core
from nekro_agent.api import schemas
from nekro_agent.core import logger
from nekro_agent.schemas.chat_message import ChatMessage
from nekro_agent.schemas.signal import MsgSignal
from nekro_agent.services.memory.embedding_service import embed_text, get_memory_embedding_dimension
from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin
from nekro_agent.services.plugin.schema import SandboxMethodType

plugin = NekroPlugin(
    name="自我认知记忆",
    module_name="self_cognition",
    description="让Bot记住自己的观点和对群友的了解，主动搜索时效性话题，并在群聊冷场时主动判断是否参与讨论。",
    version="4.0.0",
    author="Teeea",
    url="https://github.com/nekotea/nekro-agent-self-cognition",
)

COLLECTION_NAME = "nekro_self_cognition"
SEARCH_ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@plugin.mount_config()
class SelfCognitionConfig(ConfigBase):
    # 记忆槽
    MAX_TOPIC_SLOTS: int = Field(default=2, title="最大话题槽数",
        description="同时注入的最大话题记忆数量，超出时淘汰最旧的。")
    MAX_PERSONA_SLOTS: int = Field(default=4, title="最大角色画像槽数",
        description="同时注入的最大群友画像数量，超出时淘汰最旧的。")
    MAX_SLOT_CONTENT_CHARS: int = Field(default=300, title="单槽内容字数上限",
        description="注入到提示词中单个槽的最大字符数（超出部分截断）。")
    SLOT_TTL: int = Field(default=7200, title="槽过期时间(秒)",
        description="话题槽和画像槽闲置超过此时间后自动失效。")
    # 召回
    RECALL_TOP_K: int = Field(default=4, title="最大召回条数",
        description="每次刷新最多从向量数据库召回的记忆条数。")
    RECALL_SCORE_THRESHOLD: float = Field(default=0.72, title="召回相似度阈值",
        description="低于此分数的记忆不会被召回（0-1）。")
    DEDUP_THRESHOLD: float = Field(default=0.90, title="去重相似度阈值",
        description="写入记忆时已存在相似度高于此值的记忆则更新而非新建（0-1）。")
    # 搜索
    SEARCH_API_KEY: str = Field(default="", title="百度搜索 API Key",
        description="百度千帆 ai_search API Key（格式：bce-v3/ALTAK-...），留空则禁用搜索功能。")
    SEARCH_DAILY_LIMIT: int = Field(default=50, title="搜索每日限额",
        description="每日最多调用搜索API次数，超出后当日自动禁用搜索。")
    SEARCH_MAX_RESULTS: int = Field(default=5, title="搜索最大返回条数",
        description="每次搜索最多返回给LLM的参考资料条数（1-10）。")
    # 主动触发
    PROACTIVE_ENABLED: bool = Field(default=True, title="启用主动监控",
        description="是否在群聊有未回复讨论时主动触发LLM判断是否参与。")
    PROACTIVE_TRIGGER_MINUTES: int = Field(default=15, title="主动触发阈值(分钟)",
        description="未回复讨论超过此时间后触发LLM主动判断。")
    PROACTIVE_MIN_MESSAGES: int = Field(default=2, title="触发所需最少消息数",
        description="至少积累多少条未回复消息才触发主动判断（过滤单条消息）。")
    PROACTIVE_CHECK_INTERVAL: int = Field(default=300, title="监控检查间隔(秒)",
        description="后台任务检查未回复消息的间隔，默认5分钟。")


config = plugin.get_config(SelfCognitionConfig)


# ---------------------------------------------------------------------------
# 槽数据结构
# ---------------------------------------------------------------------------

@dataclass
class TopicSlot:
    topic_query: str
    injected_text: str
    created_at: float


@dataclass
class PersonaSlot:
    person_name: str
    injected_text: str
    created_at: float


_topic_slots: dict[str, list[TopicSlot]] = {}
_persona_slots: dict[str, list[PersonaSlot]] = {}


def _get_active_topic_slots(chat_key: str) -> list[TopicSlot]:
    now = time.time()
    slots = _topic_slots.get(chat_key, [])
    active = [s for s in slots if (now - s.created_at) < config.SLOT_TTL]
    if len(active) != len(slots):
        _topic_slots[chat_key] = active
    return active


def _get_active_persona_slots(chat_key: str) -> list[PersonaSlot]:
    now = time.time()
    slots = _persona_slots.get(chat_key, [])
    active = [s for s in slots if (now - s.created_at) < config.SLOT_TTL]
    if len(active) != len(slots):
        _persona_slots[chat_key] = active
    return active


def _push_topic_slot(chat_key: str, slot: TopicSlot) -> None:
    slots = _get_active_topic_slots(chat_key)
    for i, s in enumerate(slots):
        if s.topic_query == slot.topic_query:
            slots[i] = slot
            _topic_slots[chat_key] = slots
            return
    if len(slots) >= config.MAX_TOPIC_SLOTS:
        slots.sort(key=lambda s: s.created_at)
        slots.pop(0)
    slots.append(slot)
    _topic_slots[chat_key] = slots


def _push_persona_slot(chat_key: str, slot: PersonaSlot) -> None:
    slots = _get_active_persona_slots(chat_key)
    for i, s in enumerate(slots):
        if s.person_name == slot.person_name:
            slots[i] = slot
            _persona_slots[chat_key] = slots
            return
    if len(slots) >= config.MAX_PERSONA_SLOTS:
        slots.sort(key=lambda s: s.created_at)
        slots.pop(0)
    slots.append(slot)
    _persona_slots[chat_key] = slots


# ---------------------------------------------------------------------------
# 主动监控状态
# ---------------------------------------------------------------------------

@dataclass
class _PendingMessage:
    ts: float
    sender_name: str
    text: str


# {chat_key: [PendingMessage, ...]}  — 未被Bot回复的消息
_unresponded: dict[str, list[_PendingMessage]] = {}
# 已触发主动请求的 chat_key，避免重复触发（Bot响应后清除）
_triggered: set[str] = set()
_bg_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# 搜索配额
# ---------------------------------------------------------------------------

_search_counter: dict = {"date": "", "count": 0}


def _search_quota_ok() -> bool:
    today = datetime.date.today().isoformat()
    if _search_counter["date"] != today:
        _search_counter["date"] = today
        _search_counter["count"] = 0
    return bool(config.SEARCH_API_KEY) and _search_counter["count"] < config.SEARCH_DAILY_LIMIT


def _increment_search_usage() -> int:
    _search_counter["count"] += 1
    return _search_counter["count"]


# ---------------------------------------------------------------------------
# Qdrant 工具函数
# ---------------------------------------------------------------------------

_collection_ready = False


async def _ensure_collection() -> bool:
    global _collection_ready
    if _collection_ready:
        return True
    client = await nekro_core.get_qdrant_client()
    if client is None:
        return False
    try:
        collections = await client.get_collections()
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            dim = get_memory_embedding_dimension()
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=dim, distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(f"[自我认知] 创建 Collection {COLLECTION_NAME}")
        _collection_ready = True
        return True
    except Exception as e:
        logger.error(f"[自我认知] Collection 初始化失败: {e}")
        return False


async def _search_qdrant(vector: list, top_k: int, score_threshold: float) -> list[dict]:
    logger.debug(f"[自我认知][Qdrant] 搜索: top_k={top_k} threshold={score_threshold} vec_dim={len(vector)}")
    client = await nekro_core.get_qdrant_client()
    if client is None:
        return []
    try:
        t0 = time.time()
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        elapsed = time.time() - t0
        hits = [
            {
                "id": str(r.id),
                "score": r.score,
                "topic": r.payload.get("topic", ""),
                "content": r.payload.get("content", ""),
                "created_at": r.payload.get("created_at", 0.0),
                "updated_at": r.payload.get("updated_at", 0.0),
            }
            for r in results
        ]
        logger.debug(
            f"[自我认知][Qdrant] 搜索完成 {elapsed*1000:.0f}ms 命中{len(hits)}条: "
            + " | ".join(f"{h['topic'][:20]} ({h['score']:.3f})" for h in hits)
        )
        return hits
    except Exception as e:
        logger.error(f"[自我认知] Qdrant 搜索失败: {e}")
        return []


async def _upsert_qdrant(point_id: str, vector: list, payload: dict) -> bool:
    logger.debug(
        f"[自我认知][Qdrant] upsert: id={point_id[:8]}… "
        f"topic={payload.get('topic','')!r} content={str(payload.get('content',''))[:60]!r}"
    )
    client = await nekro_core.get_qdrant_client()
    if client is None:
        return False
    try:
        t0 = time.time()
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qdrant_models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.debug(f"[自我认知][Qdrant] upsert 完成 {(time.time()-t0)*1000:.0f}ms")
        return True
    except Exception as e:
        logger.error(f"[自我认知] Qdrant upsert 失败: {e}")
        return False


async def _embed(text: str, caller: str) -> list:
    """embed_text 包装器，记录调用方、文本和耗时"""
    logger.debug(f"[自我认知][Embed] {caller}: {text[:80]!r}")
    t0 = time.time()
    vec = await embed_text(text)
    logger.debug(f"[自我认知][Embed] {caller} 完成 {(time.time()-t0)*1000:.0f}ms dim={len(vec)}")
    return vec


def _build_slot_text(header: str, memories: list[dict]) -> str:
    lines = [header]
    chars_used = 0
    for m in memories:
        ts = m["updated_at"] or m["created_at"]
        date_str = f" [{time.strftime('%Y-%m-%d', time.localtime(ts))}]" if ts else ""
        line = f"- **{m['topic']}**{date_str}: {m['content']}"
        remaining = config.MAX_SLOT_CONTENT_CHARS - chars_used
        if remaining <= 0:
            break
        if len(line) > remaining:
            lines.append(line[:remaining] + "…")
            break
        lines.append(line)
        chars_used += len(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 后台主动监控任务
# ---------------------------------------------------------------------------

async def _proactive_monitor_loop():
    """后台循环：定期检查是否有未回复的讨论需要主动触发LLM"""
    from nekro_agent.services.message_service import message_service

    logger.info("[自我认知] 主动监控后台任务已启动")
    while True:
        try:
            await asyncio.sleep(config.PROACTIVE_CHECK_INTERVAL)
            if not config.PROACTIVE_ENABLED:
                logger.debug("[自我认知][Monitor] 主动监控已禁用，跳过本轮")
                continue

            now = time.time()
            trigger_threshold = config.PROACTIVE_TRIGGER_MINUTES * 60
            logger.debug(
                f"[自我认知][Monitor] 开始检查周期: {len(_unresponded)} 个频道有待处理消息 "
                f"触发阈值={trigger_threshold}s 最少消息数={config.PROACTIVE_MIN_MESSAGES}"
            )

            for chat_key, msgs in list(_unresponded.items()):
                if chat_key in _triggered:
                    logger.debug(f"[自我认知][Monitor] {chat_key}: 已触发，跳过")
                    continue
                if len(msgs) < config.PROACTIVE_MIN_MESSAGES:
                    logger.debug(f"[自我认知][Monitor] {chat_key}: {len(msgs)}条消息，未达阈值{config.PROACTIVE_MIN_MESSAGES}，跳过")
                    continue
                oldest_ts = msgs[0].ts
                age = int(now - oldest_ts)
                logger.debug(f"[自我认知][Monitor] {chat_key}: {len(msgs)}条消息，最老={age}s前，阈值={int(trigger_threshold)}s")
                if now - oldest_ts < trigger_threshold:
                    logger.debug(f"[自我认知][Monitor] {chat_key}: 未到触发时间（还差{int(trigger_threshold-age)}s），跳过")
                    continue

                # 构造主动触发的系统消息
                msg_lines = [f"- [{time.strftime('%H:%M', time.localtime(m.ts))}] {m.sender_name}: {m.text[:100]}"
                             for m in msgs[-10:]]  # 最多显示最近10条
                prompt = (
                    f"[自动整理] 以下是过去 {config.PROACTIVE_TRIGGER_MINUTES} 分钟内群聊中未收到你回复的讨论：\n"
                    + "\n".join(msg_lines)
                    + "\n\n请判断是否有你感兴趣或有能力参与的话题。"
                    "如果有观点想表达，请基于你的记忆和认知自行决定是否回复，并选择合适的时机插话。"
                    "如果没有特别想说的，直接结束本轮不回复即可。"
                )

                try:
                    await message_service.push_system_message(
                        chat_key=chat_key,
                        agent_messages=prompt,
                        trigger_agent=True,
                    )
                    _triggered.add(chat_key)
                    logger.info(f"[自我认知] 已主动触发 {chat_key}（{len(msgs)} 条未回复消息）")
                except Exception as e:
                    logger.error(f"[自我认知] 主动触发 {chat_key} 失败: {e}")

        except asyncio.CancelledError:
            logger.info("[自我认知] 主动监控后台任务已取消")
            break
        except Exception as e:
            logger.error(f"[自我认知] 主动监控循环异常: {e}")


def _ensure_bg_task():
    global _bg_task
    if not config.PROACTIVE_ENABLED:
        return
    if _bg_task is None or _bg_task.done():
        _bg_task = asyncio.create_task(_proactive_monitor_loop())


# ---------------------------------------------------------------------------
# 钩子：捕获用户消息用于主动监控
# ---------------------------------------------------------------------------

@plugin.mount_on_user_message()
async def _capture_message(ctx: schemas.AgentCtx, message: ChatMessage) -> Optional[MsgSignal]:
    _ensure_bg_task()
    text = (message.content_text or "").strip()
    if text:
        chat_key = ctx.chat_key
        pending = _unresponded.setdefault(chat_key, [])
        pending.append(_PendingMessage(
            ts=time.time(),
            sender_name=message.sender_name or message.sender_id,
            text=text[:200],
        ))
        if len(pending) > 30:
            _unresponded[chat_key] = pending[-30:]
        logger.debug(
            f"[自我认知][Monitor] 记录消息 chat={chat_key} "
            f"sender={message.sender_name or message.sender_id} "
            f"pending={len(_unresponded[chat_key])} text={text[:50]!r}"
        )
    return None


# ---------------------------------------------------------------------------
# 提示注入：静态读取两类槽，零 API 调用；同时清除未回复记录
# ---------------------------------------------------------------------------

@plugin.mount_prompt_inject_method("self_cognition_inject", "自我认知记忆注入")
async def self_cognition_inject(ctx: schemas.AgentCtx) -> str:
    chat_key = ctx.chat_key
    logger.debug(f"[自我认知][Inject] 开始注入 chat_key={chat_key}")
    # Bot 即将响应，清除该频道的未回复消息和主动触发标记
    cleared = len(_unresponded.pop(chat_key, []))
    was_triggered = chat_key in _triggered
    _triggered.discard(chat_key)
    if cleared or was_triggered:
        logger.debug(f"[自我认知][Inject] 清除未回复记录 {cleared} 条 triggered={was_triggered}")

    topic_slots = _get_active_topic_slots(chat_key)
    persona_slots = _get_active_persona_slots(chat_key)

    logger.debug(
        f"[自我认知][Inject] 槽状态: "
        f"话题={[s.topic_query for s in topic_slots]} "
        f"画像={[s.person_name for s in persona_slots]}"
    )

    PERSONA_HINT = (
        "【角色画像使用规则】群友在对话中发言时，若其画像不在下方槽中，"
        "请主动调用「刷新角色画像」加载，再结合已知信息参与对话。"
        "得知群友新信息时，调用「记录认知」保存。"
    )
    TOPIC_HINT = (
        "【话题记忆使用规则】当前话题与下方槽内容不符时，"
        "调用「刷新话题记忆」更新。对不熟悉的时效性话题可先「搜索网络」再「记录认知」。"
    )

    if not topic_slots and not persona_slots:
        result = (
            "## 我的观点与认知 (Self Cognition)\n"
            "（暂无已加载的记忆。）\n"
            f"{TOPIC_HINT}\n"
            f"{PERSONA_HINT}"
        )
        logger.debug(f"[自我认知][Inject] 无槽，注入提示文字")
        return result

    lines = ["## 我的观点与认知 (Self Cognition)", TOPIC_HINT]

    lines.append("### 话题记忆（最多2个）")
    if topic_slots:
        for slot in topic_slots:
            lines.append(slot.injected_text)
    else:
        lines.append("（暂无，可调用「刷新话题记忆」加载。）")

    lines.append(f"### 角色画像（最多4个）\n{PERSONA_HINT}")
    if persona_slots:
        for slot in persona_slots:
            lines.append(slot.injected_text)
    else:
        lines.append("（暂无。）")

    result = "\n".join(lines)
    logger.debug(f"[自我认知][Inject] 注入内容({len(result)}字):\n{result}")
    return result


# ---------------------------------------------------------------------------
# 沙盒方法：搜索网络
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="搜索网络",
    description=(
        "对时效性较强或不够了解的话题进行网络搜索，获取最新信息后自行整理成观点。"
        "搜索完成后请调用「记录认知」保存形成的观点，再作出回应。"
        f"注意：每日限额 {'{SEARCH_DAILY_LIMIT}'} 次，用完后当日自动禁用。"
    ).replace("{SEARCH_DAILY_LIMIT}", str(50)),
)
async def web_search(
    _ctx: schemas.AgentCtx,
    query: str,
) -> str:
    """Web Search (搜索网络)

    Args:
        query: 搜索词，应清晰具体，例如 "怪物猎人荒野弓箭毕业配装2025" 而非 "怪猎配装"
    """
    logger.debug(f"[自我认知][Search] 调用: query={query!r} 配额={_search_counter['count']}/{config.SEARCH_DAILY_LIMIT}")
    if not query:
        return "错误：请提供搜索词"

    if not _search_quota_ok():
        if not config.SEARCH_API_KEY:
            logger.debug("[自我认知][Search] 未配置 API Key，跳过")
            return "搜索功能未配置（请在插件设置中填写 SEARCH_API_KEY）"
        logger.debug(f"[自我认知][Search] 配额已用尽 {_search_counter['count']}/{config.SEARCH_DAILY_LIMIT}")
        return f"搜索配额已用尽（今日已使用 {_search_counter['count']}/{config.SEARCH_DAILY_LIMIT} 次），明日恢复"

    try:
        t0 = time.time()
        logger.debug(f"[自我认知][Search] 发起请求 → {SEARCH_ENDPOINT}")
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                SEARCH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {config.SEARCH_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"messages": [{"role": "user", "content": query}]},
            )
            r.raise_for_status()
            data = r.json()
        elapsed = time.time() - t0
        logger.debug(f"[自我认知][Search] 响应 {r.status_code} {elapsed*1000:.0f}ms")
    except httpx.HTTPStatusError as e:
        logger.error(f"[自我认知] 搜索请求失败: {e.response.status_code} {e.response.text[:200]}")
        return f"搜索失败（HTTP {e.response.status_code}）"
    except Exception as e:
        logger.error(f"[自我认知] 搜索异常: {e}")
        return f"搜索失败: {e}"

    used = _increment_search_usage()
    refs = data.get("references", [])
    max_r = min(config.SEARCH_MAX_RESULTS, len(refs))
    logger.debug(
        f"[自我认知][Search] 得到 {len(refs)} 条结果，取前 {max_r} 条，今日已用 {used}/{config.SEARCH_DAILY_LIMIT}: "
        + " | ".join(f"{r.get('title','')[:30]}({r.get('date','')[:10]})" for r in refs[:max_r])
    )

    if not refs:
        return f"搜索「{query}」未找到相关结果（今日已用 {used}/{config.SEARCH_DAILY_LIMIT} 次）"

    lines = [f"搜索「{query}」的结果（今日已用 {used}/{config.SEARCH_DAILY_LIMIT} 次）：\n"]
    for ref in refs[:max_r]:
        title = ref.get("title", "")
        date = ref.get("date", "")[:10]
        content = (ref.get("markdown_text") or ref.get("content") or ref.get("snippet") or "")[:300]
        url = ref.get("url", "")
        lines.append(f"**{title}**（{date}）\n{content}\n来源：{url}\n")

    lines.append("请根据以上信息整理你自己的观点，并调用「记录认知」保存后再回复。")
    result = "\n".join(lines)
    logger.debug(f"[自我认知][Search] 返回内容 {len(result)} 字")
    return result


# ---------------------------------------------------------------------------
# 沙盒方法：刷新话题记忆
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="刷新话题记忆",
    description=(
        "当话题记忆槽中的内容与当前讨论话题不匹配时调用，或对话涉及新话题时主动加载。"
        "用自己的语言描述当前话题，附上简短上下文，系统将语义搜索并更新话题记忆槽（最多2个）。"
        "仅用于Bot自身观点/立场/参与过的讨论，群友相关请用「刷新角色画像」。"
    ),
)
async def refresh_topic_memory(
    _ctx: schemas.AgentCtx,
    topic_description: str,
    context_hint: str = "",
) -> str:
    """Refresh Topic Memory (刷新话题记忆)

    Args:
        topic_description: 用自己的语言描述当前话题，例如 "AI是否会取代创意工作者"
        context_hint: 当前对话的简短背景，帮助更准确地匹配记忆
    """
    logger.debug(f"[自我认知][刷新话题] 入参: topic={topic_description!r} hint={context_hint!r} chat={_ctx.chat_key}")
    if not topic_description:
        return "错误：请提供话题描述"
    if not await _ensure_collection():
        return "错误：向量数据库暂不可用"

    query = f"{topic_description} {context_hint}".strip()
    try:
        vec = await _embed(query, "刷新话题记忆")
    except Exception as e:
        return f"错误：嵌入失败 - {e}"

    memories = await _search_qdrant(vec, config.RECALL_TOP_K, config.RECALL_SCORE_THRESHOLD)

    if not memories:
        slot_text = f"#### 话题：{topic_description}\n（暂无相关记忆）"
        result_msg = (
            f"话题「{topic_description}」暂无相关记忆。"
            "若这是时效性或不熟悉的话题，可先调用「搜索网络」获取信息，整理观点后再调用「记录认知」保存。"
        )
    else:
        slot_text = _build_slot_text(f"#### 话题：{topic_description}", memories)
        items = "\n".join(f"  - {m['topic']}: {m['content'][:80]}" for m in memories)
        result_msg = f"已加载话题「{topic_description}」的记忆（{len(memories)} 条）：\n{items}"

    _push_topic_slot(_ctx.chat_key, TopicSlot(
        topic_query=topic_description, injected_text=slot_text, created_at=time.time(),
    ))
    logger.debug(f"[自我认知][刷新话题] 完成: 找到 {len(memories)} 条，槽已更新 topic={topic_description!r}")
    return result_msg


# ---------------------------------------------------------------------------
# 沙盒方法：刷新角色画像
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="刷新角色画像",
    description=(
        "加载你对某位群友的已知信息（职业、爱好、设备、游戏、项目、宠物等）到角色画像槽。"
        "【主动调用时机】：① 群友在当前对话中发言，而其画像不在槽中时；"
        "② 刚得知关于某群友的新信息，想先查看已有记录再决定是否更新时。"
        "最多同时缓存4人，超出时自动淘汰最早的。"
        "若想保存关于群友的新信息，请在查看后调用「记录认知」。"
    ),
)
async def refresh_persona(
    _ctx: schemas.AgentCtx,
    person_name: str,
    context_hint: str = "",
) -> str:
    """Refresh Persona (刷新角色画像)

    Args:
        person_name: 群友的名称或昵称，例如 "小明"、"阿强"
        context_hint: 当前对话中关于此人的简短背景
    """
    logger.debug(f"[自我认知][刷新画像] 入参: person={person_name!r} hint={context_hint!r} chat={_ctx.chat_key}")
    if not person_name:
        return "错误：请提供群友名称"
    if not await _ensure_collection():
        return "错误：向量数据库暂不可用"

    query = f"群友{person_name} {context_hint}".strip()
    try:
        vec = await _embed(query, "刷新角色画像")
    except Exception as e:
        return f"错误：嵌入失败 - {e}"

    memories = await _search_qdrant(vec, config.RECALL_TOP_K, config.RECALL_SCORE_THRESHOLD)

    if not memories:
        slot_text = f"#### 群友：{person_name}\n（暂无相关记忆）"
        result_msg = f"群友「{person_name}」暂无画像记忆，可在了解更多后调用「记录认知」保存。"
    else:
        slot_text = _build_slot_text(f"#### 群友：{person_name}", memories)
        items = "\n".join(f"  - {m['topic']}: {m['content'][:80]}" for m in memories)
        result_msg = f"已加载群友「{person_name}」的画像（{len(memories)} 条）：\n{items}"

    _push_persona_slot(_ctx.chat_key, PersonaSlot(
        person_name=person_name, injected_text=slot_text, created_at=time.time(),
    ))
    logger.debug(f"[自我认知][刷新画像] 完成: 找到 {len(memories)} 条，槽已更新 person={person_name!r}")
    return result_msg


# ---------------------------------------------------------------------------
# 沙盒方法：记录认知
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="记录认知",
    description=(
        "记录或更新以下两类认知，若已存在高度相似的记忆则自动更新，否则新建词条：\n"
        "1. Bot自身的观点、偏好、立场——Bot在对话中形成明确观点、或内化了他人观点时调用；\n"
        "2. Bot对某位群友的了解——得知群友的职业、爱好、宠物、设备、在玩的游戏、"
        "在做的项目等具体信息时，或主动向群友提问后得到回答时调用。\n"
        "topic 格式示例：\n"
        "  - 自身观点：「音乐偏好」「对AI伦理的看法」\n"
        "  - 群友认知：「群友[小明]的游戏爱好」「群友[小红]的职业背景」\n"
        "content 示例：\n"
        "  - 「我偏好电子音乐，觉得lo-fi很适合写代码时听」\n"
        "  - 「小明最近在玩怪物猎人和CSGO，我问了怪猎用什么武器，他说是弓」"
    ),
)
async def record_cognition(
    _ctx: schemas.AgentCtx,
    topic: str,
    content: str,
) -> str:
    """Record Cognition (记录认知)

    Args:
        topic: 主题标签。自身观点用「偏好/看法/立场」等，群友认知用「群友[名字]的xxx」格式
        content: 具体内容，以第一人称叙述，记录事实或观点及其来源
    """
    logger.debug(f"[自我认知][记录] 入参: topic={topic!r} content={content[:60]!r} chat={_ctx.chat_key}")
    if not topic or not content:
        return "错误：topic 和 content 均不能为空"
    if not await _ensure_collection():
        return "错误：向量数据库暂不可用"

    summary = f"{topic}：{content}"
    try:
        vec = await _embed(summary, "记录认知")
    except Exception as e:
        return f"错误：嵌入失败 - {e}"

    now = time.time()
    existing = await _search_qdrant(vec, top_k=1, score_threshold=config.DEDUP_THRESHOLD)

    if existing:
        ex = existing[0]
        logger.debug(f"[自我认知][记录] 去重命中: score={ex['score']:.3f} existing_topic={ex['topic']!r} → 更新")
        ok = await _upsert_qdrant(ex["id"], vec, {
            "topic": topic, "content": content,
            "created_at": ex.get("created_at", now), "updated_at": now,
        })
        if ok:
            _topic_slots.pop(_ctx.chat_key, None)
            _persona_slots.pop(_ctx.chat_key, None)
            logger.debug(f"[自我认知][记录] 更新完成，已清除 {_ctx.chat_key} 的槽缓存")
            return f"已更新现有记忆（相似度 {ex['score']:.2f}）：「{ex['topic']}」→「{topic}」"
        return "写入失败，请稍后重试"
    else:
        logger.debug(f"[自我认知][记录] 无重复，新建词条 topic={topic!r}")
        ok = await _upsert_qdrant(str(uuid.uuid4()), vec, {
            "topic": topic, "content": content, "created_at": now, "updated_at": now,
        })
        if ok:
            logger.debug(f"[自我认知][记录] 新词条写入成功")
            return f"已记录新记忆：「{topic}」"
        return "写入失败，请稍后重试"


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

@plugin.mount_cleanup_method()
async def clean_up():
    global _bg_task
    if _bg_task and not _bg_task.done():
        _bg_task.cancel()
        _bg_task = None
    _topic_slots.clear()
    _persona_slots.clear()
    _unresponded.clear()
    _triggered.clear()
    logger.info("[自我认知] 插件资源已清理。")
