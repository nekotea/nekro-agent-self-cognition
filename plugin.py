"""自我认知记忆插件

让Bot拥有自己的观点、偏好和对参与过的讨论的记忆。

工作流程：
- 提示注入（零API调用）：从话题槽静态读取已召回记忆注入上下文
- LLM主动刷新：发现注入记忆与当前话题不符时，调用「刷新话题记忆」（AGENT）
  LLM自己描述话题和上下文，系统据此向量搜索并更新话题槽
- LLM写入：对话中形成明确观点时，调用「记录自我认知」（BEHAVIOR）
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
    description="让Bot记住自己的观点、偏好和参与过的讨论，对话时语义召回相关记忆，使Bot更有自我意识。",
    version="2.0.0",
    author="Teeea",
    url="https://github.com/nekotea/nekro-agent-self-cognition",
)

COLLECTION_NAME = "nekro_self_cognition"


@plugin.mount_config()
class SelfCognitionConfig(ConfigBase):
    MAX_TOPIC_SLOTS: int = Field(
        default=2,
        title="最大话题槽数",
        description="同时注入到上下文的最大话题数量，超出时淘汰最旧的话题槽。",
    )
    MAX_TOPIC_CONTENT_CHARS: int = Field(
        default=300,
        title="单话题内容字数上限",
        description="注入到提示词中单个话题记忆的最大字符数（超出部分截断）。",
    )
    RECALL_TOP_K: int = Field(
        default=4,
        title="最大召回条数",
        description="每次话题刷新最多召回的记忆条数。",
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
        title="话题槽过期时间(秒)",
        description="话题槽闲置超过此时间后自动失效，防止跨会话残留。",
    )


config = plugin.get_config(SelfCognitionConfig)


# ---------------------------------------------------------------------------
# 话题槽（每个 chat_key 最多 MAX_TOPIC_SLOTS 个）
# ---------------------------------------------------------------------------

@dataclass
class TopicSlot:
    topic_query: str    # LLM 描述的话题，作为槽的唯一标识
    injected_text: str  # 格式化好的注入块（限字数后）
    created_at: float


_topic_slots: dict[str, list[TopicSlot]] = {}


def _get_active_slots(chat_key: str) -> list[TopicSlot]:
    now = time.time()
    slots = _topic_slots.get(chat_key, [])
    active = [s for s in slots if (now - s.created_at) < config.SLOT_TTL]
    if len(active) != len(slots):
        _topic_slots[chat_key] = active
    return active


def _push_slot(chat_key: str, slot: TopicSlot) -> None:
    slots = _get_active_slots(chat_key)
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


def _build_slot_text(topic_query: str, memories: list[dict]) -> str:
    """将召回记忆格式化为话题块，限制总字符数"""
    lines = [f"### 话题：{topic_query}"]
    chars_used = 0
    for m in memories:
        ts = m["updated_at"] or m["created_at"]
        date_str = f" [{time.strftime('%Y-%m-%d', time.localtime(ts))}]" if ts else ""
        line = f"- **{m['topic']}**{date_str}: {m['content']}"
        remaining = config.MAX_TOPIC_CONTENT_CHARS - chars_used
        if remaining <= 0:
            break
        if len(line) > remaining:
            lines.append(line[:remaining] + "…")
            break
        lines.append(line)
        chars_used += len(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 提示注入：静态读取话题槽，零 API 调用
# ---------------------------------------------------------------------------

@plugin.mount_prompt_inject_method("self_cognition_inject", "自我认知记忆注入")
async def self_cognition_inject(ctx: schemas.AgentCtx) -> str:
    slots = _get_active_slots(ctx.chat_key)
    lines = ["## 我的观点与记忆 (Self Cognition)"]
    if not slots:
        lines.append(
            "（暂无已加载的记忆。若当前对话涉及你可能有观点的话题、"
            "或涉及你可能认识的群友，请调用「刷新话题记忆」工具主动查询。）"
        )
    else:
        lines.append(
            "（若以下记忆与当前话题或人物不匹配，请调用「刷新话题记忆」工具更新；"
            "最多同时保留两个话题槽。）"
        )
        for slot in slots:
            lines.append(slot.injected_text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 沙盒方法：LLM 主动刷新话题记忆
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="刷新话题记忆",
    description=(
        "当上下文中注入的话题记忆与当前讨论话题不匹配时调用，"
        "或对话涉及新话题、特定群友时主动加载相关记忆。"
        "用自己的语言描述当前话题或人物，并附上简短的上下文，"
        "系统将据此语义搜索并更新注入到对话中的记忆槽（最多保留两个话题）。"
        "示例：topic_description='小明的游戏习惯'，topic_description='关于AI取代创意工作的讨论'。"
    ),
)
async def refresh_topic_memory(
    _ctx: schemas.AgentCtx,
    topic_description: str,
    context_hint: str = "",
) -> str:
    """Refresh Topic Memory (刷新话题记忆)

    Args:
        topic_description: 用自己的语言描述当前话题或人物，例如 "AI是否会取代创意工作者"、"小明的近况和爱好"
        context_hint: 当前对话的简短上下文补充，帮助更准确地匹配记忆
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
        slot_text = f"### 话题：{topic_description}\n（暂无相关记忆）"
        result_msg = f"话题「{topic_description}」暂无相关记忆，可继续对话后调用「记录自我认知」保存观点。"
    else:
        slot_text = _build_slot_text(topic_description, memories)
        items = "\n".join(f"  - {m['topic']}: {m['content'][:80]}" for m in memories)
        result_msg = f"已加载话题「{topic_description}」的记忆（{len(memories)} 条）：\n{items}"

    _push_slot(_ctx.chat_key, TopicSlot(
        topic_query=topic_description,
        injected_text=slot_text,
        created_at=time.time(),
    ))
    logger.debug(f"[自我认知] 话题槽更新: chat={_ctx.chat_key} topic={topic_description} memories={len(memories)}")
    return result_msg


# ---------------------------------------------------------------------------
# 沙盒方法：记录自我认知
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="记录认知",
    description=(
        "记录或更新以下两类认知，若已存在高度相似的记忆则自动更新，否则新建词条：\n"
        "1. Bot自身的观点、偏好、立场——调用时机：Bot在对话中形成明确观点、或内化了他人观点时；\n"
        "2. Bot对某位群友的了解——调用时机：得知群友的职业、爱好、宠物、设备、在玩的游戏、"
        "在做的项目等具体信息时，或主动向群友提问后得到回答时。\n"
        "topic 格式示例：\n"
        "  - 自身观点：「音乐偏好」「对AI伦理的看法」\n"
        "  - 群友认知：「群友[小明]的游戏爱好」「群友[小红]的职业背景」「群友[阿强]的机器配置」\n"
        "content 示例：\n"
        "  - 「我偏好电子音乐，觉得lo-fi很适合写代码时听」\n"
        "  - 「小明最近在玩怪物猎人和CSGO，我问了怪猎用什么武器，他说是弓」\n"
        "  - 「小红是前端工程师，最近在做一个React项目」"
    ),
)
async def record_self_cognition(
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
            # 该话题记忆已更新，清除当前会话所有槽让LLM下次重新拉取最新内容
            _topic_slots.pop(_ctx.chat_key, None)
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
    logger.info("[自我认知] 插件资源已清理。")
