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

from typing import Any, Dict

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


class VisualEnhancePlugin(StoryboardMixin, RelookMixin, VideoMixin, MaiBotPlugin):
    """视觉增强：GIF 分镜 + 图片重看 + 视频理解。"""

    config_model = VisualEnhanceConfig

    # ── 生命周期：按 mixin 编排（同 reply-control） ─────────────────────

    async def on_load(self) -> None:
        await super().on_load()
        await self._video_on_load()

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
                    {"id": "debug", "title": "调试", "sections": ["debug"], "order": 6},
                ],
            }
        return schema


def create_plugin() -> MaiBotPlugin:
    """插件工厂函数，由 SDK Runner 调用。"""
    return VisualEnhancePlugin()
