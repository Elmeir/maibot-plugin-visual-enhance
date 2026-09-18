"""图片重看能力（mixin 模块）

当首遍图片描述不够用时，让 planner 按**当前问题**重新调用 VLM 看图。

移植自 kumburovicbranko682-boop/maibot-image-relook（MIT；致谢与许可见 CHANGELOG
与 LICENSE）。重看范围覆盖表情包组件（``type == "emoji"``）：组件带内嵌字节时直接
使用，否则按 hash 从宿主 Images 表补读原图（表情包同样登记在该表）。

取不到图 / 识图失败都返回可读提示，不影响消息链。
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from maibot_sdk import Field, PluginConfigBase, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType

RELOOK_MIN_BASE64_LENGTH = 64
"""判定"内嵌字节有效"的最小 base64 长度（短于该值视为占位/空值）。"""

RELOOK_BASE64_URL_PREFIX = "base64://"
"""组件里可能出现的 base64 URL 前缀。"""

RELOOK_IMAGE_KEYS: Tuple[str, ...] = ("binary_data_base64", "base64", "image_base64", "data")
"""组件里可能携带内嵌字节的键（按优先级）。"""

RELOOK_TEXT_KEYS: Tuple[str, ...] = ("response", "content", "text", "message")
"""LLM 返回里可能承载文本的键。"""


class RelookSectionConfig(PluginConfigBase):
    """图片重看配置。"""

    __ui_label__ = "图片重看"
    __ui_icon__ = "eye"
    __ui_order__ = 4

    enabled: bool = Field(
        default=True,
        description="启用图片重看（inspect_image 工具）",
        json_schema_extra={
            "label": "图片重看",
            "hint": "首遍描述不够时，让麦麦按当前问题重新看图；关闭后不提供该工具",
        },
    )
    include_emoji: bool = Field(
        default=True,
        description="重看范围包含表情包组件",
        json_schema_extra={
            "label": "含表情包",
            "hint": "开（默认）= 表情包也能重看；关 = 只看普通图片（与上游行为一致）",
        },
    )
    lookback_hours: float = Field(
        default=24.0,
        description="向前查找图片的小时数（0 = 不限）",
        json_schema_extra={
            "label": "回溯小时数",
            "hint": "只在该时间窗内的消息里找图；0 = 不限，按最近条数上限取",
        },
    )
    recent_message_limit: int = Field(
        default=40,
        description="拉取最近消息条数上限",
        json_schema_extra={"label": "最近消息条数", "hint": "越多越全，但拉取开销略增"},
    )
    max_images: int = Field(
        default=10,
        description="候选图片数量上限",
        json_schema_extra={"label": "候选图片上限", "hint": "去重后保留的图数量；image_index 在此范围内取值"},
    )
    llm_task: str = Field(
        default="vlm",
        description="识图用的宿主模型任务名（不是模型名）",
        json_schema_extra={
            "label": "识图任务名",
            "hint": "宿主**模型任务名**（不是模型名），需 visual=true 的多模态任务；默认 vlm",
            "placeholder": "默认 vlm",
        },
    )
    vlm_model: str = Field(
        default="",
        description="可选：识图用的具体模型名（留空按任务策略选择）",
        json_schema_extra={
            "label": "识图模型名（可选）",
            "hint": "留空=按「识图任务名」的模型选择策略；仅当日志报「未找到名为 xxx 的模型」时，填 model_config 的 [models] 里定义的具体模型名",
            "placeholder": "留空",
        },
    )
    temperature: float = Field(
        default=0.2,
        description="重看生成温度",
        json_schema_extra={"label": "温度", "hint": "越低越稳定；识图建议 0.1~0.3"},
    )
    max_tokens: int = Field(
        default=1024,
        description="重看回答最大 token",
        json_schema_extra={"label": "最大 token", "hint": "观察结果的长度上限"},
    )


def _normalize_base64(value: Any) -> str:
    """把组件里的字节字段归一化为可解码的 base64 串；无效返回空串。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(RELOOK_BASE64_URL_PREFIX):
        text = text[len(RELOOK_BASE64_URL_PREFIX):]
    if text.lower().startswith("data:image"):
        marker = "base64,"
        index = text.find(marker)
        if index >= 0:
            text = text[index + len(marker):]
    compact = "".join(text.split())
    if not compact:
        return ""
    try:
        base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return ""
    return compact


class RelookMixin:
    """图片重看能力（mixin）：由主插件类拼装，配置来自主配置的 relook 段。"""

    def __init__(self) -> None:
        super().__init__()

    # ── 配置读取 ────────────────────────────────────────────

    def _relook_opt(self, key: str, default: Any = None) -> Any:
        """安全读取 relook 段配置。"""
        try:
            return getattr(getattr(self.config, "relook", None), key, default)
        except Exception:
            return default

    def _relook_enabled(self) -> bool:
        return bool(self._relook_opt("enabled", True))

    # ── 组件与字节 ──────────────────────────────────────────

    @staticmethod
    def _relook_guess_format(comp: Dict[str, Any], image_base64: str) -> str:
        """推断图片格式：优先组件自带声明，其次按字节魔数，最后回退 png。"""
        for key in ("format", "mime_type", "image_format"):
            value = str(comp.get(key) or "").strip().lower()
            if not value:
                continue
            normalized = value.split("/")[-1].replace("jpg", "jpeg")
            if normalized in {"png", "jpeg", "webp", "gif", "bmp"}:
                return normalized
        try:
            head = base64.b64decode(image_base64[:16] + "==")
        except Exception:
            head = b""
        if head.startswith(b"\x89PNG"):
            return "png"
        if head.startswith(b"\xff\xd8"):
            return "jpeg"
        if head.startswith(b"GIF8"):
            return "gif"
        if head.startswith(b"RIFF"):
            return "webp"
        return "png"

    @staticmethod
    def _relook_component_payload(comp: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """从组件里取出内嵌字节 (format, base64)；没有则 None。"""
        for key in RELOOK_IMAGE_KEYS:
            if key not in comp:
                continue
            image_base64 = _normalize_base64(comp.get(key))
            if image_base64 and len(image_base64) >= RELOOK_MIN_BASE64_LENGTH:
                return (
                    RelookMixin._relook_guess_format(comp, image_base64),
                    image_base64,
                )
        return None

    def _relook_iter_components(self, node: Any, include_emoji: bool) -> List[Dict[str, Any]]:
        """递归收集图片组件；**含表情包**（type=emoji）是本模块相对上游的修复点。"""
        found: List[Dict[str, Any]] = []
        if isinstance(node, list):
            for item in node:
                found.extend(self._relook_iter_components(item, include_emoji))
            return found
        if not isinstance(node, dict):
            return found

        wanted = {"image"} | ({"emoji"} if include_emoji else set())
        item_type = str(node.get("type") or "").strip().lower()
        if item_type in wanted:
            found.append(node)
        elif item_type == "forward":
            found.extend(self._relook_iter_components(node.get("data"), include_emoji))
        elif item_type == "dict":
            nested = node.get("data")
            if isinstance(nested, dict) and str(nested.get("type") or "").lower() in wanted:
                found.append(nested if "data" in nested else node)
            else:
                found.extend(self._relook_iter_components(nested, include_emoji))

        for key in ("raw_message", "content", "components", "message"):
            if key in node:
                found.extend(self._relook_iter_components(node[key], include_emoji))
        return found

    def _relook_format_from_path(self, path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "jpg":
            return "jpeg"
        if suffix in {"png", "jpeg", "webp", "gif", "bmp"}:
            return suffix
        guessed, _ = mimetypes.guess_type(str(path))
        if guessed and guessed.startswith("image/"):
            return guessed.split("/", 1)[1].replace("jpg", "jpeg")
        return "png"

    def _relook_read_file(self, path: Path) -> Optional[Tuple[str, str]]:
        """读取本地图片文件，返回 (format, base64)。"""
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            # 相对路径按 MaiBot 根解析：插件数据目录 = <root>/data/plugins/<plugin_id>
            try:
                root = self.ctx.paths.data_dir.parents[2]
                candidates.append(root / path)
            except Exception:
                pass
            candidates.append(Path.cwd() / path)
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                data = candidate.read_bytes()
            except OSError:
                continue
            if data:
                return (self._relook_format_from_path(candidate), base64.b64encode(data).decode("ascii"))
        return None

    @staticmethod
    def _relook_rows(result: Any) -> List[Dict[str, Any]]:
        """把 database.query 的返回整理成行列表。"""
        if isinstance(result, list):
            return [row for row in result if isinstance(row, dict)]
        if not isinstance(result, dict):
            return []
        if result.get("success") is False:
            return []
        for key in ("data", "result", "items", "records", "rows"):
            value = result.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            if isinstance(value, dict):
                return [value]
        return [result]

    async def _relook_load_by_hash(self, image_hash: str) -> Optional[Tuple[str, str]]:
        """按 hash 从 Images 表取 full_path 并读字节。

        图片与表情包都登记在 Images 表（image_type 不同），这里**不限类型**，
        因此表情包组件只要带 hash 就能补读到原图。
        """
        image_hash = str(image_hash or "").strip()
        if not image_hash:
            return None
        try:
            result = await self.ctx.db.query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": image_hash},
                limit=3,
                single_result=False,
            )
        except Exception as exc:
            self._dbg("[视觉增强·重看] 按 hash 查 Images 失败: %s", exc)
            return None
        for row in self._relook_rows(result):
            full_path = str(row.get("full_path") or "").strip()
            if not full_path:
                continue
            loaded = self._relook_read_file(Path(full_path))
            if loaded:
                return loaded
        return None

    # ── 候选收集 ────────────────────────────────────────────

    def _relook_within_lookback(self, message: Dict[str, Any], hours: float) -> bool:
        try:
            if float(hours) <= 0:
                return True
        except (TypeError, ValueError):
            return True
        raw = message.get("timestamp") or message.get("time")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return True  # 拿不到时间不过滤
        if value > 1e11:  # 毫秒时间戳
            value /= 1000.0
        return (time.time() - value) <= float(hours) * 3600.0

    async def _relook_fetch_recent(self, chat_id: str) -> List[Dict[str, Any]]:
        """拉取最近消息（message.get_recent）。"""
        try:
            limit = int(self._relook_opt("recent_message_limit", 40) or 40)
        except (TypeError, ValueError):
            limit = 40
        try:
            result = await self.ctx.message.get_recent(chat_id, limit=max(1, limit))
        except Exception as exc:
            self._dbg("[视觉增强·重看] 拉取最近消息失败: %s", exc)
            return []
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            for key in ("messages", "data", "result", "items"):
                value = result.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    async def _relook_collect(
        self,
        messages: List[Dict[str, Any]],
        prefer_msg_id: str = "",
    ) -> List[Dict[str, Any]]:
        """收集候选图片（最新在前，去重，按上限截断）。"""
        include_emoji = bool(self._relook_opt("include_emoji", True))
        try:
            max_images = int(self._relook_opt("max_images", 10) or 10)
        except (TypeError, ValueError):
            max_images = 10
        lookback = self._relook_opt("lookback_hours", 24.0)

        preferred: List[Dict[str, Any]] = []
        ordered: List[Dict[str, Any]] = []

        for message in messages:
            if not self._relook_within_lookback(message, lookback):
                continue
            message_id = str(message.get("message_id") or "").strip()
            components = self._relook_iter_components(
                message.get("raw_message") or message, include_emoji
            )
            for comp in components:
                parsed = self._relook_component_payload(comp)
                image_hash = str(comp.get("hash") or "").strip()
                if parsed is None:
                    if not image_hash:
                        continue
                    parsed = await self._relook_load_by_hash(image_hash)
                if parsed is None:
                    continue
                image_format, image_base64 = parsed
                item = {
                    "format": image_format,
                    "base64": image_base64,
                    "hash": image_hash,
                    "message_id": message_id,
                    "kind": str(comp.get("type") or "").strip().lower(),
                }
                if prefer_msg_id and message_id == prefer_msg_id:
                    preferred.append(item)
                ordered.append(item)

        # 消息通常旧 -> 新；反转后 index=1 为最新图
        ordered.reverse()
        preferred.reverse()
        source = preferred or ordered

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in source:
            key = item.get("hash") or str(item["base64"])[:64]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_images:
                break
        return deduped

    # ── 识图 ────────────────────────────────────────────────

    @staticmethod
    def _relook_extract_text(result: Any) -> str:
        """从 llm.generate 的返回里取出文本。"""
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in RELOOK_TEXT_KEYS:
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("data", "result"):
                inner = result.get(key)
                if isinstance(inner, (str, dict)):
                    text = RelookMixin._relook_extract_text(inner)
                    if text:
                        return text
        return ""

    async def _relook_ask(self, question: str, image_format: str, image_base64: str) -> str:
        """带着问题把图再交给 VLM 看一次。"""
        prompt: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请根据用户问题直接观察图片并作答。"
                            "只回答问题需要的事实，不要空泛描述整张图。"
                            "如果看不清或图片里没有对应信息，就明确说看不清/没有。"
                            f"\n用户问题：{question}"
                        ),
                    },
                    {
                        "type": "image",
                        "image_format": image_format,
                        "image_base64": image_base64,
                    },
                ],
            }
        ]
        try:
            # 新版 SDK 语义：model = 具体模型名（留空表示按任务策略选择），
            # task_name = 宿主模型任务名。上游实现把任务名塞进 model，宿主会
            # 按"具体模型名"查找并报「未找到名为 vlm 的模型」——这里按新语义传参。
            result = await self.ctx.llm.generate(
                prompt=prompt,
                task_name=str(self._relook_opt("llm_task", "vlm") or "vlm"),
                model=str(self._relook_opt("vlm_model", "") or ""),
                temperature=float(self._relook_opt("temperature", 0.2) or 0.2),
                max_tokens=int(self._relook_opt("max_tokens", 1024) or 1024),
            )
        except Exception as exc:
            self._dbg("[视觉增强·重看] 识图调用失败: %s", exc)
            return f"[识图失败] {exc}"
        return self._relook_extract_text(result) or "[识图失败] 模型没有返回有效内容"

    # ── 工具 ────────────────────────────────────────────────

    @Tool(
        "inspect_image",
        description=(
            "按当前问题重新查看聊天里的图片（含表情包）。"
            "当首遍图片描述不够、缺失关键细节（数量、文字、颜色、位置等），"
            "或用户追问图片细节时必须使用；不要用猜测代替看图。"
            "参数：question=具体问题；image_index=从最近往前第几张图（1 起，默认 1）；"
            "也可传 msg_id 指定消息。"
        ),
        parameters=[
            ToolParameterInfo(
                name="question",
                param_type=ToolParamType.STRING,
                description="要向图片提出的具体问题，例如：图里有几根手指？招牌上写着什么？",
                required=True,
            ),
            ToolParameterInfo(
                name="image_index",
                param_type=ToolParamType.INTEGER,
                description="从最近往前数第几张图，1=最近一张。默认 1。",
                required=False,
            ),
            ToolParameterInfo(
                name="msg_id",
                param_type=ToolParamType.STRING,
                description="可选。优先查看该消息里的图片。",
                required=False,
            ),
        ],
        visibility="visible",
    )
    async def handle_inspect_image(
        self,
        question: str = "",
        image_index: Any = 1,
        msg_id: str = "",
        stream_id: str = "",
        chat_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """inspect_image：按问题重看最近的一张图（含表情包）。"""
        if not self._relook_enabled():
            return {"success": False, "error": "图片重看已在配置中关闭"}

        clean_question = str(question or "").strip()
        if not clean_question:
            return {"success": False, "error": "请提供要看图回答的具体问题"}

        session_id = str(chat_id or stream_id or "").strip()
        if not session_id:
            return {"success": False, "error": "缺少会话 ID，无法定位图片"}

        prefer_msg_id = str(
            msg_id or kwargs.get("msg_id") or kwargs.get("message_id") or ""
        ).strip()

        messages = await self._relook_fetch_recent(session_id)
        images = await self._relook_collect(messages, prefer_msg_id)
        if not images:
            return {
                "success": False,
                "error": "最近消息里没有可用的图片（含表情包）；可确认图片是否已被清理或超出回溯时间",
            }

        # planner 有时会传 index=0
        raw_index = kwargs.get("index", image_index)
        try:
            index = int(raw_index if raw_index is not None else 1)
        except (TypeError, ValueError):
            index = 1
        index = max(1, min(len(images), index))
        chosen = images[index - 1]

        answer = await self._relook_ask(
            clean_question, str(chosen["format"]), str(chosen["base64"])
        )
        kind = "表情包" if chosen.get("kind") == "emoji" else "图片"
        self._dbg(
            "[视觉增强·重看] 第 %s 张（%s）已重看：%s", index, kind, clean_question[:40]
        )
        return {
            "success": True,
            "content": f"[图片重看·第{index}张/{kind}] 问题：{clean_question}\n观察结果：{answer}",
        }
