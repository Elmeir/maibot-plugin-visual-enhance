"""视觉增强（visual-enhance）

给麦麦补上三块「看」的能力，全部通过插件 SDK 实现、不改宿主一行代码：

1. **GIF 分镜**（storyboard.py）：动图交给视觉模型之前，把各帧合成到同一张
   网格静态图上，让模型看到动画的内容与时序；另含动漫角色识别增强。
2. **图片重看**（relook.py）：首遍描述不够时，让 planner 带着具体问题重新
   调用 VLM 看图——含表情包。
3. **视频理解**（video.py）：识别聊天中的 B 站/抖音链接，把视频信息、章节
   要点与官方 AI 总结附加到上下文（不主动回复）+ 两个解析工具。

各能力以 mixin 拼装（配置段与方法前缀各自独立），生命周期与配置页签由
本入口统一编排。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from maibot_sdk import Field, MaiBotPlugin, PluginConfigBase

from .relook import RelookMixin, RelookSectionConfig
from .storyboard import (
    AnimeSectionConfig,
    MergeSectionConfig,
    OutputSectionConfig,
    PluginSectionConfig,
    StoryboardMixin,
)
from .video import CredentialSectionConfig, ParseSectionConfig, VideoMixin

SUPPORTED_CONFIG_VERSION = "1.1.0"
"""插件支持的配置版本（新增段由宿主自动补齐默认值，无需再升）。"""


class DebugSectionConfig(PluginConfigBase):
    """调试日志（统一开关，覆盖三块能力的诊断输出）。"""

    __ui_label__ = "调试"
    __ui_icon__ = "bug"
    __ui_order__ = 5

    enabled: bool = Field(
        default=False,
        description="开启调试日志",
        json_schema_extra={
            "label": "调试日志",
            "hint": "开=解析/识图/凭据的诊断信息以 INFO 输出（日志搜「视频」「图片重看」「凭据」）；日常保持关闭",
        },
    )


class ToolInfoBaseConfig(PluginConfigBase):
    """工具信息基类（只读展示：LLM 视角的工具定义，加载时自动写入）。

    WebUI 对字段的显示值取自配置值本身（schema.default 会被空配置值覆盖），
    展示文本由 on_load 写入 config.toml 对应段；读取处忽略这些字段（纯展示）。
    """

    __ui_icon__ = "wrench"
    __ui_order__ = 10

    visibility: str = Field(
        default="",
        description="工具对 LLM 的可见性（运行时生成，只读）",
        json_schema_extra={
            "label": "可见性",
            "hint": "deferred = 不在常驻工具列表（按需发现，可被 tool_search 搜到）；visible = 始终提供给 LLM",
            "disabled": True,
            "rows": 2,
        },
    )
    description: str = Field(
        default="",
        description="LLM 看到的工具描述（运行时生成，只读）",
        json_schema_extra={
            "label": "描述",
            "hint": "LLM 实际看到的工具描述；每次插件加载时自动刷新",
            "disabled": True,
            "rows": 5,
        },
    )
    parameters: str = Field(
        default="",
        description="工具参数清单（运行时生成，只读）",
        json_schema_extra={
            "label": "参数",
            "hint": "每个参数一行：名称（类型，必填/可选）：说明",
            "disabled": True,
            "rows": 5,
        },
    )


class ToolInspectImageConfig(ToolInfoBaseConfig):
    """图片重看工具（inspect_image）。"""

    __ui_label__ = "inspect_image"


class ToolParseBilibiliConfig(ToolInfoBaseConfig):
    """B站解析工具（parse_bilibili_video）。"""

    __ui_label__ = "parse_bilibili_video"


class ToolParseDouyinConfig(ToolInfoBaseConfig):
    """抖音解析工具（parse_douyin_video）。"""

    __ui_label__ = "parse_douyin_video"


def _collect_tool_info(handler: Any) -> Dict[str, str]:
    """从组件声明生成单个工具的展示字段（可见性 / 描述 / 参数）。"""
    info = getattr(handler, "__maibot_component_info__", None)
    if info is None:
        return {}
    metadata = getattr(info, "metadata", None)
    visibility = ""
    if isinstance(metadata, dict):
        visibility = str(metadata.get("visibility") or "").strip()
    description = str(
        getattr(info, "brief_description", "") or getattr(info, "description", "") or ""
    ).strip() or "（无描述）"
    parameters = getattr(info, "parameters", None) or []
    param_lines: List[str] = []
    for param in parameters:
        param_name = str(getattr(param, "name", "") or "")
        param_type = getattr(param, "param_type", None)
        type_text = (
            getattr(param_type, "value", None)
            or getattr(param_type, "name", None)
            or "string"
        )
        required = "必填" if bool(getattr(param, "required", False)) else "可选"
        param_desc = str(getattr(param, "description", "") or "")
        param_lines.append(f"{param_name}（{type_text}，{required}）: {param_desc}")
    return {
        "visibility": visibility or "deferred（未显式声明时的宿主默认）",
        "description": description,
        "parameters": "\n".join(param_lines) if param_lines else "（无参数）",
    }


def _collect_all_tool_info() -> Dict[str, Dict[str, str]]:
    """收集全部工具的展示字段（段名 → 字段字典；加载同步与 Scheme 注入共用）。"""
    return {
        "tool_inspect_image": _collect_tool_info(RelookMixin.handle_inspect_image),
        "tool_parse_bilibili_video": _collect_tool_info(VideoMixin.tool_parse_bilibili_video),
        "tool_parse_douyin_video": _collect_tool_info(VideoMixin.tool_parse_douyin_video),
    }


def _sync_component_info_sections(
    values: Dict[str, Dict[str, str]], config_path: Optional[Path] = None
) -> None:
    """把组件信息展示字段写入 config.toml 对应段（每段内容有变化才写）。

    实现与 reply-control 一致：段内容完全由本函数管理（整段重写），文本用
    JSON 转义（TOML 基础字符串兼容）；失败静默（不影响插件运行）。
    """
    if not values:
        return
    try:
        target = config_path or (Path(__file__).parent / "config.toml")
        if not target.exists():
            return
        content = target.read_text(encoding="utf-8")
        original = content
        for section, fields in values.items():
            if not fields:
                continue
            body = [
                f"{name} = " + json.dumps(value, ensure_ascii=False)
                for name, value in fields.items()
            ]
            content = _replace_section_body(content, section, body)
        if content != original:
            target.write_text(content, encoding="utf-8")
    except Exception:
        pass  # 展示同步失败不影响插件运行


def _replace_section_body(content: str, section: str, body: List[str]) -> str:
    """重写 TOML 指定段的段体（段不存在时追加）；内容未变化时原样返回。"""
    lines = content.splitlines()
    start: Optional[int] = None
    end = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"[{section}]":
            start = index
            continue
        if start is not None and stripped.startswith("[") and stripped.endswith("]"):
            end = index
            break
    trailing = "\n" if content.endswith("\n") else ""
    if start is None:
        suffix = "" if content.endswith("\n") else "\n"
        return f"{content}{suffix}\n[{section}]\n" + "\n".join(body) + "\n"
    current = [line for line in lines[start + 1 : end] if line.strip()]
    if current == body:
        return content  # 未变化：不写盘、不触发配置事件
    if end >= len(lines):
        return "\n".join([*lines[: start + 1], *body]) + trailing
    return "\n".join([*lines[: start + 1], *body, "", *lines[end:]]) + trailing


class VisualEnhanceConfig(PluginConfigBase):
    """插件根配置。"""

    plugin: PluginSectionConfig = Field(
        default_factory=PluginSectionConfig, json_schema_extra={"label": "主页"}
    )
    merge: MergeSectionConfig = Field(
        default_factory=MergeSectionConfig, json_schema_extra={"label": "帧合成"}
    )
    output: OutputSectionConfig = Field(
        default_factory=OutputSectionConfig, json_schema_extra={"label": "输出图像"}
    )
    anime: AnimeSectionConfig = Field(
        default_factory=AnimeSectionConfig, json_schema_extra={"label": "动漫识别"}
    )
    relook: RelookSectionConfig = Field(
        default_factory=RelookSectionConfig, json_schema_extra={"label": "图片重看"}
    )
    parse: ParseSectionConfig = Field(
        default_factory=ParseSectionConfig, json_schema_extra={"label": "视频解析"}
    )
    credential: CredentialSectionConfig = Field(
        default_factory=CredentialSectionConfig, json_schema_extra={"label": "视频凭据"}
    )
    debug: DebugSectionConfig = Field(
        default_factory=DebugSectionConfig, json_schema_extra={"label": "调试"}
    )
    tool_inspect_image: ToolInspectImageConfig = Field(
        default_factory=ToolInspectImageConfig
    )
    tool_parse_bilibili_video: ToolParseBilibiliConfig = Field(
        default_factory=ToolParseBilibiliConfig
    )
    tool_parse_douyin_video: ToolParseDouyinConfig = Field(
        default_factory=ToolParseDouyinConfig
    )


class VisualEnhancePlugin(StoryboardMixin, RelookMixin, VideoMixin, MaiBotPlugin):
    """视觉增强：GIF 分镜 + 图片重看 + 视频理解。"""

    config_model = VisualEnhanceConfig

    # ── 生命周期：按 mixin 编排（同 reply-control） ─────────────────────

    async def on_load(self) -> None:
        await super().on_load()
        await self._video_on_load()
        # 工具信息只读展示：WebUI 取值依赖配置值本身，加载时同步一次
        # （内容有变化才写盘；纯展示字段，读取处忽略）
        _sync_component_info_sections(_collect_all_tool_info())

    async def on_unload(self) -> None:
        await self._video_on_unload()
        await super().on_unload()

    async def on_config_update(
        self, scope: str, config_data: Dict[str, Any], version: str
    ) -> None:
        await super().on_config_update(scope, config_data, version)
        await self._video_on_config_update(scope, config_data, version)

    # ── WebUI 页签布局 ─────────────────────────────────────────────────
    # SDK 默认 layout=auto（所有 section 堆在一页）；这里改为 tabs——
    # 相关段合并展示（视频解析 + 视频凭据同属「视频理解」），调试独立一页。
    @classmethod
    def build_config_schema(
        cls,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> Dict[str, Any]:
        schema = super().build_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )
        if isinstance(schema, dict) and schema.get("sections"):
            schema["layout"] = {
                "type": "tabs",
                "tabs": [
                    {"id": "main", "title": "主页", "sections": ["plugin"], "order": 0},
                    {"id": "merge", "title": "帧合成", "sections": ["merge"], "order": 1},
                    {"id": "output", "title": "输出图像", "sections": ["output"], "order": 2},
                    {"id": "anime", "title": "动漫识别", "sections": ["anime"], "order": 3},
                    {"id": "relook", "title": "图片重看", "sections": ["relook"], "order": 4},
                    {
                        "id": "video",
                        "title": "视频理解",
                        "sections": ["parse", "credential"],
                        "order": 5,
                    },
                    {
                        "id": "debug",
                        "title": "调试",
                        "sections": [
                            "debug",
                            "tool_inspect_image",
                            "tool_parse_bilibili_video",
                            "tool_parse_douyin_video",
                        ],
                        "order": 6,
                    },
                ],
            }
            # 工具信息卡：各段字段 default 注入（双保险；框内值由 on_load 写入配置值）
            sections_map = schema.get("sections") or {}
            for section_name, fields in _collect_all_tool_info().items():
                section = sections_map.get(section_name)
                if not isinstance(section, dict):
                    continue
                section_fields = section.get("fields")
                if not isinstance(section_fields, dict):
                    continue
                for field_name, value in fields.items():
                    field = section_fields.get(field_name)
                    if isinstance(field, dict):
                        field["default"] = value
        return schema


def create_plugin() -> MaiBotPlugin:
    """插件工厂函数，由 SDK Runner 调用。"""
    return VisualEnhancePlugin()
