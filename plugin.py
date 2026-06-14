"""自我认知记忆插件

让Bot拥有自己的观点、偏好和对参与过的讨论的记忆，以及对群友的认知画像。

工作流程：
- 提示注入（零API调用）：从两类槽静态读取已召回记忆注入上下文
    - 话题记忆槽（max 2）：Bot自身对话题的观点、参与过的讨论
    - 角色画像槽（max 4）：Bot对特定群友的了解
- LLM主动刷新：发现注入内容与当前话题/人物不符时，调用对应刷新工具（AGENT，触发重新调用）
- LLM写入：对话中形成明确认知时，调用「记录认知」（BEHAVIOR）
"""

import time
import uuid
from dataclasses import dataclass
from typing import Optional

from pydantic import Field
from qdrant_client import models as qdrant_models

from nekro_agent.api import core as nekro_core
from nekro_agent.api import schemas
from nekro_agent.core import logger
from nekro_agent.services.memory.embedding_service import embed_text, get_memory_embedding_dimension
from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin
from nekro_agent.services.plugin.schema import SandboxMethodType

plugin = NekroPlugin(
    name="自我认知记忆",
    module_name="self_cognition",
    description="让Bot记住自己的观点和对群友的了解，对话时分类语义召回，使Bot更有自我意识和人情味。",
    version="3.0.0",
    author="Teeea",
    url="https://github.com/nekotea/nekro-agent-self-cognition",
)

COLLECTION_NAME = "nekro_self_cognition"


@plugin.mount_config()
class SelfCognitionConfig(ConfigBase):
    MAX_TOPIC_SLOTS: int = Field(
        default=2,
        title="最大话题槽数",
        description="同时注入的最大话题记忆数量（Bot自身观点/讨论），超出时淘汰最旧的。",
    )
    MAX_PERSONA_SLOTS: int = Field(
        default=4,
        title="最大角色画像槽数",
        description="同时注入的最大群友画像数量，超出时淘汰最旧的。",
    )
    MAX_SLOT_CONTENT_CHARS: int = Field(
        default=300,
        title="单槽内容字数上限",
        description="注入到提示词中单个槽的最大字符数（超出部分截断）。",
    )
    RECALL_TOP_K: int = Field(
        default=4,
        title="最大召回条数",
        description="每次刷新最多从向量数据库召回的记忆条数。",
    )
    RECALL_SCORE_THRESHOLD: float = Field(
        default=0.72,
        title="召回相似度阈值",
        description="低于此分数的记忆不会被召回（0-1，越高越严格）。",
    )
    DEDUP_THRESHOLD: float = Field(
        default=0.90,
        title="去重相似度阈值",
        description="写入记忆时已存在相似度高于此值的记忆则更新而非新建（0-1）。",
    )
    SLOT_TTL: int = Field(
        default=7200,
        title="槽过期时间(秒)",
        description="话题槽和画像槽闲置超过此时间后自动失效，防止跨会话残留。",
    )


config = plugin.get_config(SelfCognitionConfig)


# ---------------------------------------------------------------------------
# 槽数据结构
# ---------------------------------------------------------------------------

@dataclass
class TopicSlot:
    topic_query: str    # LLM 描述的话题，作为唯一标识
    injected_text: str
    created_at: float


@dataclass
class PersonaSlot:
    person_name: str    # 群友名称，作为唯一标识
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
                    size=dim,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(f"[自我认知] 创建 Collection {COLLECTION_NAME}")
        _collection_ready = True
        return True
    except Exception as e:
        logger.error(f"[自我认知] Collection 初始化失败: {e}")
        return False


async def _search(vector: list, top_k: int, score_threshold: float) -> list[dict]:
    client = await nekro_core.get_qdrant_client()
    if client is None:
        return []
    try:
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
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
    except Exception as e:
        logger.error(f"[自我认知] 搜索失败: {e}")
        return []


async def _upsert(point_id: str, vector: list, payload: dict) -> bool:
    client = await nekro_core.get_qdrant_client()
    if client is None:
        return False
    try:
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qdrant_models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return True
    except Exception as e:
        logger.error(f"[自我认知] upsert 失败: {e}")
        return False


def _build_slot_text(header: str, memories: list[dict]) -> str:
    """将召回记忆格式化为注入块，限制总字符数"""
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
# 提示注入：静态读取两类槽，零 API 调用
# ---------------------------------------------------------------------------

@plugin.mount_prompt_inject_method("self_cognition_inject", "自我认知记忆注入")
async def self_cognition_inject(ctx: schemas.AgentCtx) -> str:
    topic_slots = _get_active_topic_slots(ctx.chat_key)
    persona_slots = _get_active_persona_slots(ctx.chat_key)

    if not topic_slots and not persona_slots:
        return (
            "## 我的观点与认知 (Self Cognition)\n"
            "（暂无已加载的记忆。若当前对话涉及你可能有观点的话题，请调用「刷新话题记忆」；"
            "若涉及某位群友，请调用「刷新角色画像」。）"
        )

    lines = ["## 我的观点与认知 (Self Cognition)"]

    lines.append(
        "### 话题记忆（最多2个，不符时调用「刷新话题记忆」更新）"
    )
    if topic_slots:
        for slot in topic_slots:
            lines.append(slot.injected_text)
    else:
        lines.append("（暂无，可调用「刷新话题记忆」加载。）")

    lines.append(
        "### 角色画像（最多4个，缺少某人时调用「刷新角色画像」加载）"
    )
    if persona_slots:
        for slot in persona_slots:
            lines.append(slot.injected_text)
    else:
        lines.append("（暂无，可调用「刷新角色画像」加载。）")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 沙盒方法：刷新话题记忆
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="刷新话题记忆",
    description=(
        "当话题记忆槽中的内容与当前讨论话题不匹配时调用，或对话涉及新话题时主动加载。"
        "用自己的语言描述当前话题，附上简短上下文，"
        "系统将语义搜索并更新话题记忆槽（最多同时保留2个话题）。"
        "仅用于Bot自身观点/立场/参与过的讨论，群友相关请用「刷新角色画像」。"
        "示例：topic_description='关于AI是否会取代创意工作的讨论'"
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
    if not topic_description:
        return "错误：请提供话题描述"
    if not await _ensure_collection():
        return "错误：向量数据库暂不可用"

    query = f"{topic_description} {context_hint}".strip()
    try:
        vec = await embed_text(query)
    except Exception as e:
        return f"错误：嵌入失败 - {e}"

    memories = await _search(vec, config.RECALL_TOP_K, config.RECALL_SCORE_THRESHOLD)

    if not memories:
        slot_text = f"#### 话题：{topic_description}\n（暂无相关记忆）"
        result_msg = f"话题「{topic_description}」暂无相关记忆，可在形成观点后调用「记录认知」保存。"
    else:
        slot_text = _build_slot_text(f"#### 话题：{topic_description}", memories)
        items = "\n".join(f"  - {m['topic']}: {m['content'][:80]}" for m in memories)
        result_msg = f"已加载话题「{topic_description}」的记忆（{len(memories)} 条）：\n{items}"

    _push_topic_slot(_ctx.chat_key, TopicSlot(
        topic_query=topic_description,
        injected_text=slot_text,
        created_at=time.time(),
    ))
    logger.debug(f"[自我认知] 话题槽更新: {topic_description} ({len(memories)} 条)")
    return result_msg


# ---------------------------------------------------------------------------
# 沙盒方法：刷新角色画像
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="刷新角色画像",
    description=(
        "当对话涉及某位群友，而角色画像槽中没有此人或内容已过时时调用。"
        "提供群友的名称或昵称，系统将语义搜索该人的相关记忆并更新角色画像槽"
        "（最多同时保留4个人的画像）。"
        "示例：person_name='小明'，context_hint='他在说自己的游戏配置'"
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
        context_hint: 当前对话中关于此人的简短背景，帮助更准确地匹配记忆
    """
    if not person_name:
        return "错误：请提供群友名称"
    if not await _ensure_collection():
        return "错误：向量数据库暂不可用"

    query = f"群友{person_name} {context_hint}".strip()
    try:
        vec = await embed_text(query)
    except Exception as e:
        return f"错误：嵌入失败 - {e}"

    memories = await _search(vec, config.RECALL_TOP_K, config.RECALL_SCORE_THRESHOLD)

    if not memories:
        slot_text = f"#### 群友：{person_name}\n（暂无相关记忆）"
        result_msg = f"群友「{person_name}」暂无画像记忆，可在了解更多后调用「记录认知」保存。"
    else:
        slot_text = _build_slot_text(f"#### 群友：{person_name}", memories)
        items = "\n".join(f"  - {m['topic']}: {m['content'][:80]}" for m in memories)
        result_msg = f"已加载群友「{person_name}」的画像（{len(memories)} 条）：\n{items}"

    _push_persona_slot(_ctx.chat_key, PersonaSlot(
        person_name=person_name,
        injected_text=slot_text,
        created_at=time.time(),
    ))
    logger.debug(f"[自我认知] 角色画像槽更新: {person_name} ({len(memories)} 条)")
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
        "  - 群友认知：「群友[小明]的游戏爱好」「群友[小红]的职业背景」「群友[阿强]的机器配置」\n"
        "content 示例：\n"
        "  - 「我偏好电子音乐，觉得lo-fi很适合写代码时听」\n"
        "  - 「小明最近在玩怪物猎人和CSGO，我问了怪猎用什么武器，他说是弓」\n"
        "  - 「小红是前端工程师，最近在做一个React项目」"
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
    if not topic or not content:
        return "错误：topic 和 content 均不能为空"
    if not await _ensure_collection():
        return "错误：向量数据库暂不可用"

    summary = f"{topic}：{content}"
    try:
        vec = await embed_text(summary)
    except Exception as e:
        return f"错误：嵌入失败 - {e}"

    now = time.time()
    existing = await _search(vec, top_k=1, score_threshold=config.DEDUP_THRESHOLD)

    if existing:
        ex = existing[0]
        payload = {
            "topic": topic,
            "content": content,
            "created_at": ex.get("created_at", now),
            "updated_at": now,
        }
        ok = await _upsert(ex["id"], vec, payload)
        if ok:
            _topic_slots.pop(_ctx.chat_key, None)
            _persona_slots.pop(_ctx.chat_key, None)
            return f"已更新现有记忆（相似度 {ex['score']:.2f}）：「{ex['topic']}」→「{topic}」"
        return "写入失败，请稍后重试"
    else:
        point_id = str(uuid.uuid4())
        payload = {"topic": topic, "content": content, "created_at": now, "updated_at": now}
        ok = await _upsert(point_id, vec, payload)
        if ok:
            return f"已记录新记忆：「{topic}」"
        return "写入失败，请稍后重试"


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

@plugin.mount_cleanup_method()
async def clean_up():
    _topic_slots.clear()
    _persona_slots.clear()
    logger.info("[自我认知] 插件资源已清理。")
