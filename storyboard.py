"""GIF 分镜能力（mixin 模块）

在麦麦把 GIF 动图交给视觉模型之前，把动画各帧按时序抽帧并合成到同一张
网格静态图上（带帧序号与说明条），让视觉模型看到动画的完整内容与时序，
而不是只看到第一帧。只处理普通图片组件（type=image）中的 GIF；
表情包组件不处理——宿主表情链路原生支持 GIF 抽帧拼接分析，且其
description 兼作情绪标签来源（逗号分隔标签格式），插件旁路会造成
双倍解析并覆写标签格式，已废弃。

实现（挂在宿主钩子上，不改宿主一行代码）：
- ``chat.receive.before_process``：检测 GIF 魔数，多帧动画的图片组件
  临时替换为合成图（原图字节暂存插件内部登记表，按"组件索引+hash"登记）；
- ``chat.receive.after_process``：落库前用暂存字节恢复被替换的组件，
  并调度后台描述搬运任务；
- 描述搬运：等宿主 VLM 写出合成图描述后，用 ctx.db 写回原图 hash 的
  Images 记录，宿主视觉占位刷新器自动回填进麦麦上下文；
  同图重发走"已就绪描述快速路径"，宿主直接跳过识别；
- 动漫角色识别（可选，默认开）：描述就绪后若命中动漫关键词，把图片送
  AnimeTrace（ai.animedb.cn）识图，把「角色《作品》」结果追加进描述
  （GIF 用最清晰帧，静态图原样上传/降采样）；去重标记防重复外发。

任何环节失败一律原样放行，绝不阻塞消息链。
"""

import asyncio
import base64
import hashlib
import io
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import aiohttp
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from maibot_sdk import Field, HookHandler, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

SUPPORTED_CONFIG_VERSION = "1.1.0"

GIF_MAGICS = (b"GIF87a", b"GIF89a")
"""GIF 文件头魔数。"""

MERGE_CACHE_MAX_ENTRIES = 64
"""合成结果内存缓存上限（键=原图哈希+参数指纹，超出按最旧淘汰）。"""

RELOCATE_TIMEOUT_SECONDS = 600.0
"""描述搬运任务的最长等待时间（宿主后台 VLM 识别合成图的宽限期）。"""

RELOCATE_POLL_INTERVAL_SECONDS = 5.0
"""描述搬运任务的轮询间隔。"""

ORIGINAL_BYTES_CACHE_MAX_ENTRIES = 16
"""替换窗口内暂存原图字节的缓存上限（key=orig_hash）。

before_process 把组件二进制替换为合成图（让宿主识别多帧），after_process
在落库前用这里暂存的原图字节恢复组件。条目生命周期只有一次消息处理
（几秒），并发 GIF 数量通常为个位数；超出上限时放弃替换（原样放行），
避免原图字节因缓存淘汰而丢失导致合成图落库。
"""

DHASH_DUPLICATE_DISTANCE = 6
"""感知哈希（64bit）汉明距离低于该值时，视为与上一选中帧"几乎相同"。"""

ANIMETRACE_API_URL = "https://api.animetrace.com/v1/search"
"""AnimeTrace 识图接口（ai.animedb.cn 页面对应的官方 API）。"""

ANIME_KEYWORDS_DEFAULT = "动漫,动画,番剧,二次元,漫画,卡通,插画,角色,anime,アニメ,動漫,動畫"
"""动漫触发关键词默认表：描述命中任一才调用识图接口。"""

ANIME_INFO_MARKER = "[动漫角色识别]"
"""识图结果写入描述时的标记，兼作去重依据（已带标记的描述不再重复识别）。"""

ANIME_FRAME_CACHE_MAX_ENTRIES = 16
"""待识别素材暂存上限（key=原图 hash，零拷贝引用，超出按最旧淘汰）。"""

ANIME_FRAME_MAX_SIDE = 1600
"""送识图的帧最长边像素上限（接口限 4MB，先降采样控制体积）。"""

ANIME_UPLOAD_MAX_BYTES = 3_500_000
"""识图上传体积软上限（接口硬限 4MB，留余量）；超过则先抽帧降采样。"""

ANIME_WATCHED_MAX_ENTRIES = 1024
"""静态图观察去重上限（key=原图 hash，超出按最旧淘汰，防长期驻留膨胀）。"""

ANIME_ATTEMPTED_MAX_ENTRIES = 256
"""识图已尝试去重上限（key=原图 hash）：接口无结果/失败的图不再走"补课"重试，
避免同图每次重发都重复替换+识图循环。"""

ANIME_STATIC_POLL_INTERVAL_SECONDS = 2.0
"""静态图描述观察轮询间隔（比描述搬运更密，缩小与宿主占位刷新器的竞态窗口）。"""

ANIME_CACHE_MAX_ENTRIES = 2000
"""识别结果磁盘缓存条目上限（key=原图 hash，超出按最旧淘汰）。"""

ANIME_WORK_OMIT_PEOPLE = 3
"""总人数达到该值视为"较多"：追加文本只写角色名、省略《作品》，控制描述长度。"""


# ─── 配置模型 ────────────────────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否启用插件（总开关；关闭后 GIF 一律原样交给视觉模型）",
        json_schema_extra={
            "label": "插件总开关",
            "hint": "关闭后 GIF 一律原样交给视觉模型，钩子直接放行",
        },
    )
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（勿改）",
        json_schema_extra={"hidden": True, "disabled": True},
    )


class MergeSectionConfig(PluginConfigBase):
    """帧合成行为配置。"""

    __ui_label__ = "帧合成"
    __ui_icon__ = "film"
    __ui_order__ = 1

    process_image: bool = Field(
        default=True,
        description="处理普通图片组件中的 GIF 动图",
        json_schema_extra={
            "label": "处理图片",
            "hint": "普通图片（type=image）里的 GIF 动图会被合成多帧网格图",
        },
    )
    max_frames: int = Field(
        default=9,
        ge=2,
        le=25,
        description="最多抽取的帧数（多帧动画均匀抽样；9=3x3 网格，兼顾信息量与视觉模型 token 开销）",
        json_schema_extra={
            "label": "最多帧数",
            "hint": "9=3x3 网格；帧越多 token 越贵",
        },
    )
    min_frames: int = Field(
        default=2,
        ge=2,
        le=10,
        description="动画至少包含多少帧才触发合成（单帧 GIF 等价静态图，无需处理）",
        json_schema_extra={
            "label": "最少帧数",
            "hint": "帧数不足该值的 GIF 视为静态图，原样放行",
        },
    )
    adaptive_frames: bool = Field(
        default=True,
        description=(
            "根据源动画的运动量与帧数，自动在『最少帧数~最多帧数』之间决定实际抽帧数："
            "画面几乎静止/循环重复的简单动画少抽帧（省 token），运动丰富、帧数多的动画多抽帧（保证证据）"
        ),
        json_schema_extra={
            "label": "自适应帧数",
            "hint": "按动画运动量自动调节实际帧数；关闭则始终抽『最多帧数』",
        },
    )
    draw_frame_index: bool = Field(
        default=True,
        description="在每帧左上角绘制序号，帮助视觉模型理解播放顺序",
        json_schema_extra={
            "label": "绘制帧序号",
            "hint": "每帧左上角标 1、2、3…，视觉模型更容易按时序描述动画",
        },
    )
    clean_ghost_components: bool = Field(
        default=True,
        description="识别完成后把被替换的组件恢复为原图，防止合成图进入聊天记录/WebUI",
        json_schema_extra={
            "label": "落库前恢复原图",
            "hint": "关闭后合成图会随消息落库显示九宫格",
        },
    )
    storyboard_caption: bool = Field(
        default=True,
        description="在合成图顶部绘制一行说明文字，引导视觉模型把网格图理解为一段动画而不是一张拼图",
        json_schema_extra={
            "label": "动画分镜说明条",
            "hint": "无中文字体时自动改用英文说明",
        },
    )


class OutputSectionConfig(PluginConfigBase):
    """合成图输出配置。"""

    __ui_label__ = "输出图像"
    __ui_icon__ = "image"
    __ui_order__ = 2

    output_format: Literal["jpeg", "png"] = Field(
        default="jpeg",
        description="合成图编码格式：jpeg 体积小（推荐）；png 无损、体积较大",
        json_schema_extra={
            "label": "输出格式",
            "hint": "jpeg 体积小省 token（推荐）；png 无损但明显更大",
        },
    )
    jpeg_quality: int = Field(
        default=85,
        ge=50,
        le=95,
        description="JPEG 质量（仅输出格式为 jpeg 时生效）",
        json_schema_extra={
            "label": "JPEG 质量",
            "hint": "50~95，85 兼顾清晰度与体积；仅 jpeg 格式生效",
        },
    )
    max_cell_size: int = Field(
        default=480,
        ge=128,
        le=1024,
        description="网格中每帧格子的最长边像素上限（帧原图更大时等比缩小）",
        json_schema_extra={
            "label": "单帧格子边长上限",
            "hint": "帧原图超过该尺寸会等比缩小；480 在多数视觉模型上清晰度与开销均衡",
        },
    )
    grid_gap: int = Field(
        default=4,
        ge=0,
        le=32,
        description="帧与帧之间的间隔像素（白底）",
        json_schema_extra={
            "label": "帧间隔（像素）",
            "hint": "网格帧之间的白色间隔，0~32",
        },
    )


class AnimeSectionConfig(PluginConfigBase):
    """动漫角色识别（AnimeTrace / ai.animedb.cn）配置。"""

    __ui_label__ = "动漫识别"
    __ui_icon__ = "face"
    __ui_order__ = 3

    enabled: bool = Field(
        default=True,
        description=(
            "启用动漫角色识别：图片描述就绪后若命中动漫关键词，把图片送 "
            "AnimeTrace（ai.animedb.cn）识图，角色与作品名追加进图片描述"
        ),
        json_schema_extra={
            "label": "动漫角色识别",
            "hint": "失败只记日志；已识别过不重复外发",
        },
    )
    process_gif: bool = Field(
        default=True,
        description=(
            "GIF 动图也参与动漫角色识别：随分镜描述搬运衔接识图。"
            "与静态图片识别（process_static）相互独立，可分别开关"
        ),
        json_schema_extra={
            "label": "GIF 动图识别",
            "hint": "关闭后只对静态图片识图；GIF 只做分镜合成",
        },
    )
    process_static: bool = Field(
        default=True,
        description=(
            "静态图片也参与动漫角色识别：插件观察所有普通图片（type=image）的宿主识别描述，"
            "命中动漫关键词时把图片送识图接口。多帧 GIF 由分镜流程自动覆盖，无需开启此项"
        ),
        json_schema_extra={
            "label": "静态图片识别",
            "hint": "关闭后只识别 GIF 分镜",
        },
    )
    keywords: str = Field(
        default=ANIME_KEYWORDS_DEFAULT,
        description=(
            "动漫触发关键词（逗号分隔）：视觉模型描述命中任一关键词才调用识图接口，"
            "避免真人照片等无关图片外发到第三方服务"
        ),
        json_schema_extra={
            "label": "触发关键词",
            "placeholder": "动漫,动画,番剧,二次元…",
            "hint": "逗号/换行分隔，不区分大小写；清空时回退默认表",
        },
    )
    api_url: str = Field(
        default=ANIMETRACE_API_URL,
        description="AnimeTrace 识图接口地址（ai.animedb.cn 页面对应的官方 API）",
        json_schema_extra={
            "label": "识图接口",
            "hint": "官方接口；服务迁移域名时才改",
        },
    )
    model: str = Field(
        default="",
        description="识别模型 ID，留空=服务端默认（官方模型列表动态增减，不建议写死）",
        json_schema_extra={
            "label": "识别模型",
            "placeholder": "留空=默认（当前如 animetrace-yuri-4.2）",
            "hint": "留空走服务端默认最稳",
        },
    )
    max_characters: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最多写入描述的人物数（多人画面按检测顺序取前 N 个）",
        json_schema_extra={
            "label": "人物数量上限",
            "hint": "识别到多个人物时最多把前 N 个写进描述，控制描述长度",
        },
    )
    max_candidates: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "每个人物写入的候选数：接口只返回按可能性排序的候选列表（无概率值），"
            "实测存在把公会名等非角色条目排在首位的情况（且 not_confident=false），"
            "默认取前 2 个候选（用『或』连接），正确角色大概率在列，交由麦麦结合上下文判断"
        ),
        json_schema_extra={
            "label": "每人候选数",
            "hint": "top-1 可能是公会名等错条，2 更稳",
        },
    )
    smart_filter: bool = Field(
        default=True,
        description=(
            "智能筛选：一人写了多个候选（候选较多）或总人数较多（≥3 人）时，"
            "追加文本只写角色名、省略《作品》控制描述长度；关闭则始终带作品名"
        ),
        json_schema_extra={
            "label": "智能筛选",
            "hint": "多候选时只写角色名、省略《作品》",
        },
    )
    filter_not_confident: bool = Field(
        default=True,
        description=(
            "过滤低置信结果：存在高置信结果时跳过官方标记为 not_confident"
            "（候选过多、需人工确认）的检测框——宁可缺判也不错判；全部为"
            "低置信时兜底写入（带「（低置信）」后缀）；关闭后低置信检测框"
            "始终写入"
        ),
        json_schema_extra={
            "label": "过滤低置信结果",
            "hint": "有高置信时丢弃低置信框；全低置信时带标注兜底",
        },
    )
    cache_enabled: bool = Field(
        default=True,
        description=(
            "识别结果本地缓存：同一张图（按内容 hash）的识别结果落盘复用，"
            "重启后也不再重复调用识图接口；仅缓存成功结果，失败/无结果不缓存以便重试"
        ),
        json_schema_extra={
            "label": "识别结果缓存",
            "hint": "按 hash 落盘复用，上限 2000 条",
        },
    )
    debug_mode: bool = Field(
        default=False,
        description=(
            "调试日志：把动漫识别链路的诊断信息（关键词未命中、素材缺失、观察超时等）"
            "以 INFO 级别输出，便于排查『为什么没识别』；日常使用建议关闭"
        ),
        json_schema_extra={
            "label": "调试日志",
            "hint": "日志搜『动漫识别』看链路每一步",
        },
    )
    inject_enabled: bool = Field(
        default=False,
        description=(
            "阻塞注入（首次发送即见角色）：收到图片后在等待秒数内完成识图，"
            "把角色标签直接写入本轮消息。开启后消息会阻塞等待（通常 1~3 秒），"
            "且门控关闭时所有图片都会外发 AnimeTrace"
        ),
        json_schema_extra={
            "label": "阻塞注入",
            "hint": "开=首次发送即见角色标签；关=同图下次出现时生效",
        },
    )
    inject_timeout_seconds: float = Field(
        default=6.0,
        ge=1.0,
        le=30.0,
        description="阻塞注入的等待秒数：识图（含可选 VLM 门控）超过该时长即放行消息，转后台观察兜底",
        json_schema_extra={
            "label": "注入等待（秒）",
            "hint": "超时放行消息，转后台观察兜底",
        },
    )
    inject_wait_vlm: bool = Field(
        default=True,
        description=(
            "门控开关（是否等待 VLM 关键词结果）：开启时先调麦麦的 VLM 任务生成描述，"
            "命中动漫关键词才外发识图（描述同时写入组件，宿主跳过重复识别）；"
            "关闭时跳过门控直接识图——更快，但所有图片都会外发 AnimeTrace"
        ),
        json_schema_extra={
            "label": "等待 VLM 门控",
            "hint": "开=先 VLM 门控再识图；关=直接识图（全部外发）",
        },
    )
    vlm_model: str = Field(
        default="",
        description=(
            "VLM 具体模型名（可选）：门控调用默认使用任务自身的模型选择策略，"
            "仅在日志报『未找到名为 xxx 的模型』时，填 model_config 的 [models] 里定义的模型名"
        ),
        json_schema_extra={
            "label": "VLM 模型名（可选）",
            "hint": "留空=用任务默认模型；门控报『未找到模型』时才填",
        },
    )
    timeout_seconds: float = Field(
        default=15.0,
        ge=5.0,
        le=60.0,
        description="识图接口请求超时（秒）",
        json_schema_extra={
            "label": "接口超时（秒）",
            "hint": "单次识图请求的最长等待时间",
        },
    )


class StoryboardMixin:
    """GIF 分镜能力（mixin 模块）。

    由主插件类拼装（mixin 模式，同 reply-control）：本文件只提供能力，
    配置来自主配置的 merge / output / anime 段，入口见 plugin.py。
    """

    def __init__(self) -> None:
        super().__init__()
        # (原图 sha256 + 参数指纹) -> 合成图 bytes；OrderedDict 做 LRU 淘汰
        self._merge_cache: "OrderedDict[str, bytes]" = OrderedDict()
        # 替换窗口内暂存的原图字节（key=orig_hash），after_process 恢复组件后删除
        self._original_bytes: "OrderedDict[str, bytes]" = OrderedDict()
        # 组件索引 -> (合成图 hash, 原图 hash)：after_process 按"索引 + hash"
        # 精确恢复被替换的组件。不要把恢复信息写进组件本身——宿主钩子之间
        # 会做序列化往返，组件对象上的未知字段会被丢弃。
        self._replaced_index: "OrderedDict[int, tuple[str, str]]" = OrderedDict()
        # 已在跑的描述搬运任务（按合成图 hash 去重）
        self._relocate_tasks: "set[str]" = set()
        # 动漫角色识别的待识别素材（key=orig_hash，原图原始字节的零拷贝
        # 引用）；描述就绪后由后台任务现抽识别帧并消费，观察超时/失败时
        # 随任务 finally 释放，上限内 LRU 淘汰
        self._anime_frames: "OrderedDict[str, bytes]" = OrderedDict()
        # 静态图动漫识别的观察去重（key=原图 hash）：同图重发不重复观察/外发
        self._anime_watched: "OrderedDict[str, None]" = OrderedDict()
        # 识图已尝试（key=原图 hash）：接口无结果/失败的图不再走快速路径
        # "补课"重试，避免同图重发的替换+识图循环
        self._anime_attempted: "OrderedDict[str, None]" = OrderedDict()
        # 识别结果缓存（key=原图 hash → 追加信息行），懒加载自插件数据目录
        self._anime_cache: Optional[Dict[str, str]] = None
        # 数据库能力不可用时的告警去重标志（避免每轮轮询刷屏）
        self._db_warned: bool = False

    @staticmethod
    def _b64_decode(data: Any) -> bytes:
        """组件二进制 base64 解码，失败返回空字节串。"""
        if not isinstance(data, str) or not data:
            return b""
        try:
            return base64.b64decode(data)
        except Exception:
            return b""

    # ── 配置读取 ────────────────────────────────────────────────────────

    def _opt(self, section: str, key: str, default: Any = None) -> Any:
        """安全读取插件配置项。"""
        try:
            return getattr(getattr(self.config, section, None), key, default)
        except Exception:
            return default

    def _enabled(self) -> bool:
        return bool(self._opt("plugin", "enabled", True))

    def _component_managed(self, comp_type: str) -> bool:
        """组件类型在当前配置下是否会被处理（仅普通图片组件；表情包不处理）。"""
        return comp_type == "image" and bool(self._opt("merge", "process_image", True))

    def _settings_fingerprint(self) -> str:
        """合成参数指纹，参与缓存键：参数变了缓存自动失效。"""
        return "|".join(
            str(
                self._opt(section, key, default)
            )
            for section, key, default in (
                ("merge", "max_frames", 9),
                ("merge", "min_frames", 2),
                ("merge", "adaptive_frames", True),
                ("merge", "draw_frame_index", True),
                ("merge", "storyboard_caption", True),
                ("output", "output_format", "jpeg"),
                ("output", "jpeg_quality", 85),
                ("output", "max_cell_size", 480),
                ("output", "grid_gap", 4),
            )
        )

    def _anime_keywords(self) -> "tuple[str, ...]":
        """动漫触发关键词：配置值按中英文逗号/顿号/分号/换行切分，空值回退默认表。"""
        raw = str(self._opt("anime", "keywords", ANIME_KEYWORDS_DEFAULT) or "").strip()
        keywords = tuple(k.strip() for k in re.split(r"[,，、;；\n\r]+", raw) if k.strip())
        if keywords:
            return keywords
        return tuple(k.strip() for k in ANIME_KEYWORDS_DEFAULT.split(",") if k.strip())

    def _looks_anime(self, description: str) -> bool:
        """描述是否命中任一动漫关键词（不区分大小写）。"""
        lowered = description.lower()
        return any(kw.lower() in lowered for kw in self._anime_keywords())

    # ── 钩子：入站消息改写 ──────────────────────────────────────────────

    @HookHandler(
        "chat.receive.before_process",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        name="gif_frame_merge",
        description="把 GIF 动图各帧合成网格图，临时替换图片组件交给视觉链路（落库前恢复原图）",
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_process(self, message: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """入站钩子：消息里的 GIF 动图原位替换为多帧合成图。

        使用 LATE 槽位：排在消息防抖类插件（在 NORMAL 槽阻塞窗口并合并消息）
        之后，看到的是防抖放行后的最终消息——防抖窗口内并入的第 2..N 条消息
        的 GIF 也能被分镜处理，且被防抖 abort 的消息不会再触发本插件
        （避免登记永远不会被回收的替换记录）。
        """
        if not self._enabled() or not isinstance(message, dict):
            return {"action": "continue"}

        components = message.get("raw_message")
        if not isinstance(components, list) or not components:
            return {"action": "continue"}

        relevant = False
        for comp in components:
            if not isinstance(comp, dict):
                continue
            comp_type = str(comp.get("type") or "").strip().lower()
            if (
                self._component_managed(comp_type)
                or self._static_anime_active(comp_type)
                or self._blocking_inject_active(comp_type)
            ):
                relevant = True
                break
        if not relevant:
            return {"action": "continue"}

        # 阻塞注入：收集可注入的静态图片，并行识图（总预算 = 注入等待秒数）
        inject_enabled = bool(self._opt("anime", "inject_enabled", False))
        inject_results: Dict[int, "tuple[Any, Optional[str]]"] = {}  # 原索引 -> (组件, 追加文本)
        if inject_enabled:
            try:
                inject_timeout = max(1.0, float(self._opt("anime", "inject_timeout_seconds", 6.0) or 6.0))
            except (TypeError, ValueError):
                inject_timeout = 6.0
            inject_jobs: Dict[int, "asyncio.Task[tuple[Any, Optional[str]]]"] = {}
            for idx, comp in enumerate(components):
                if isinstance(comp, dict) and self._blocking_inject_active(
                    str(comp.get("type") or "").strip().lower()
                ):
                    inject_jobs[idx] = asyncio.get_running_loop().create_task(
                        self._blocking_inject_image(comp)
                    )
            if inject_jobs:
                done, pending = await asyncio.wait(
                    inject_jobs.values(), timeout=inject_timeout
                )
                for task in pending:
                    task.cancel()
                for idx, task in inject_jobs.items():
                    if task in done and not task.cancelled() and task.exception() is None:
                        inject_results[idx] = task.result()
                    else:
                        inject_results[idx] = (None, None)  # 超时/异常 → 原样放行 + 后台观察兜底

        new_components: List[Any] = []
        changed = False
        out_idx = 0  # 输出列表索引：注入会插入文本组件，登记必须用输出位置
        for idx, comp in enumerate(components):
            results = await self._process_component(comp)
            if len(results) == 1 and results[0] is comp:
                # 未被分镜替换：优先消费阻塞注入结果，其次静态观察
                if idx in inject_results:
                    inj_comp, inj_text = inject_results[idx]
                    new_components.append(inj_comp if inj_comp is not None else comp)
                    out_idx += 1
                    if inj_comp is not None:
                        changed = True
                    if inj_text:
                        new_components.append({"type": "text", "data": inj_text})
                        out_idx += 1
                        changed = True
                    if inj_comp is None and isinstance(comp, dict):
                        # 注入超时/异常：转后台观察兜底
                        await self._watch_static_anime(comp)
                    continue
                new_components.append(comp)
                out_idx += 1
                # 未被替换的普通图片（静态图/单帧 GIF/合成失败件）走静态
                # 动漫识别观察；被替换的 GIF 由描述搬运流程衔接识图
                if isinstance(comp, dict):
                    await self._watch_static_anime(comp)
                continue
            changed = True

            comp_hash = str(comp.get("hash") or "").strip()
            result = results[0]
            result_hash = str(result.get("hash") or "").strip()

            if len(results) == 1 and comp_hash and result_hash and result_hash != comp_hash:
                # 替换模式：按输出索引登记（注入插入会使输出位置偏移），
                # after_process 据此恢复原图。登记信息放在插件内部而非组件
                # 字段——宿主钩子间的序列化往返会丢弃组件对象上的未知字段。
                orig_bytes = self._b64_decode(comp.get("binary_data_base64"))
                if orig_bytes:
                    self._original_bytes[comp_hash] = orig_bytes
                    self._original_bytes.move_to_end(comp_hash)
                    while len(self._original_bytes) > ORIGINAL_BYTES_CACHE_MAX_ENTRIES:
                        self._original_bytes.popitem(last=False)
                self._replaced_index[out_idx] = (result_hash, comp_hash)
            new_components.extend(results)
            out_idx += len(results)

        # 防御：索引登记只服务于紧随其后的一条消息，避免异常路径残留累积
        while len(self._replaced_index) > ORIGINAL_BYTES_CACHE_MAX_ENTRIES * 4:
            self._replaced_index.popitem(last=False)

        if not changed:
            return {"action": "continue"}

        new_message = dict(message)
        new_message["raw_message"] = new_components
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": new_message}}

    @HookHandler(
        "chat.receive.after_process",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        name="gif_frame_ghost_cleanup",
        description="落库前恢复被替换的图片组件，防止合成图进入聊天记录，并调度多帧描述搬运",
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_after_process(self, message: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """after 钩子：宿主 process() 已完成识别调度，此刻用 before_process
        暂存的原图字节按"组件索引 + hash"恢复被替换的组件——随后的消息落库
        按原图 hash 保存文件与记录，图片库/WebUI 全部保真；此时原图记录尚未被
        宿主识别（不存在首帧描述），搬运任务把合成图的多帧描述写到原图 hash
        名下后，刷新器即可安全回填。
        """
        if not isinstance(message, dict):
            return {"action": "continue"}
        if not bool(self._opt("merge", "clean_ghost_components", True)):
            self._replaced_index.clear()
            return {"action": "continue"}

        components = message.get("raw_message")
        if not isinstance(components, list) or not components:
            return {"action": "continue"}

        kept: List[Any] = []
        handled: List[tuple[str, str]] = []  # (合成图 hash, 原图 hash)
        for idx, comp in enumerate(components):
            comp_hash = str(comp.get("hash") or "").strip() if isinstance(comp, dict) else ""

            if idx in self._replaced_index:
                merged_hash, orig_hash = self._replaced_index[idx]
                if comp_hash != merged_hash:
                    # 组件列表被其他环节改动导致索引错位：放弃恢复（保持合成图），
                    # 避免把原图字节写错到别的组件上。
                    self.ctx.logger.warning(
                        "[视觉增强·分镜] 替换组件索引错位（hash 不匹配），放弃恢复：idx=%s hash=%s",
                        idx,
                        comp_hash[:12],
                    )
                    self._replaced_index.pop(idx, None)
                    kept.append(comp)
                    continue
                self._replaced_index.pop(idx, None)
                restored = self._restore_replaced_component(comp, orig_hash)
                # 恢复失败（暂存字节缺失）时组件保持合成图状态，但保留组件本体
                kept.append(restored if restored is not None else comp)
                handled.append((merged_hash, orig_hash))
                continue

            kept.append(comp)

        if not handled:
            return {"action": "continue"}

        # 恢复全部完成后统一清理暂存的原图字节（同 hash 多组件共用一份）
        for merged_hash, orig_hash in handled:
            # 动漫角色识别：零拷贝登记原图字节（识别帧由后台任务在描述
            # 就绪后现抽，不给消息链增加任何同步耗时）
            orig_bytes = self._original_bytes.get(orig_hash)
            if orig_bytes is not None:
                self._stash_anime_frame(orig_hash, orig_bytes)
            self._original_bytes.pop(orig_hash, None)
            self._spawn_relocate_task(merged_hash, orig_hash)
        self.ctx.logger.info(
            "[视觉增强·分镜] 已处理 %d 个组件（恢复原图/回收合成图），多帧描述将在后台搬运",
            len(handled),
        )

        new_message = dict(message)
        new_message["raw_message"] = kept
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": new_message}}

    def _restore_replaced_component(self, comp: Dict[str, Any], orig_hash: str) -> Optional[Dict[str, Any]]:
        """把替换成合成图的组件恢复为原图（二进制与 hash），失败返回 None。"""
        original_bytes = self._original_bytes.get(orig_hash) if orig_hash else None
        if original_bytes is None:
            self.ctx.logger.warning(
                "[视觉增强·分镜] 暂存的原图字节缺失，组件将保持合成图状态 hash=%s",
                str(comp.get("hash") or "")[:12],
            )
            return None
        restored = dict(comp)
        restored["binary_data_base64"] = base64.b64encode(original_bytes).decode("ascii")
        restored["hash"] = orig_hash
        restored["data"] = ""  # 保持空占位，等待多帧描述回填
        return restored

    def _spawn_relocate_task(self, merged_hash: str, orig_hash: str) -> None:
        """按合成图 hash 去重后启动描述搬运后台任务。"""
        if merged_hash in self._relocate_tasks:
            return
        self._relocate_tasks.add(merged_hash)
        try:
            asyncio.get_running_loop().create_task(
                self._relocate_description(merged_hash, orig_hash)
            )
        except RuntimeError:
            self._relocate_tasks.discard(merged_hash)

    async def _relocate_description(self, merged_hash: str, orig_hash: str) -> None:
        """等待宿主 VLM 完成合成图识别，把多帧描述搬到原图 hash 名下。

        - 等落库流程建好原图记录（替换模式原图不参与宿主识别，
          记录由 after_process 之后的落库创建）；
        - 超时放弃时行为退化为宿主原生识别（原图/首帧描述），不影响消息链。
        """
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + RELOCATE_TIMEOUT_SECONDS
            while loop.time() < deadline:
                if await self._try_relocate(merged_hash, orig_hash):
                    if self._gif_anime_active():
                        await self._maybe_recognize_anime(merged_hash, orig_hash)
                    return
                await asyncio.sleep(RELOCATE_POLL_INTERVAL_SECONDS)

            self.ctx.logger.warning(
                "[视觉增强·分镜] 描述搬运超时放弃（宿主识别未完成或记录缺失）：merged=%s orig=%s",
                merged_hash[:12],
                orig_hash[:12],
            )
        finally:
            self._relocate_tasks.discard(merged_hash)
            self._anime_frames.pop(orig_hash, None)  # 释放未消费的识别帧

    async def _try_relocate(self, merged_hash: str, orig_hash: str) -> bool:
        """尝试搬运一次多帧描述，成功（或已搬运过）返回 True。

        Args:
            merged_hash: 合成网格图 hash（描述来源）。
            orig_hash: 原图 hash（描述落点）。

        Note:
            只写 IMAGE 类型记录；EMOJI 记录的 description 兼作情绪标签来源
            （逗号分隔标签格式），多帧长描述不可覆写。
        """
        source = await self._db_get_image_record(merged_hash)
        if not (source and source.get("vlm_processed")):
            return False
        source_desc = str(source.get("description") or "").strip()
        if not source_desc:
            return False

        target = await self._db_get_image_record(orig_hash)
        if target is None:
            return False  # 记录尚未由落库流程创建，下轮轮询再试
        if str(target.get("description") or "").strip() == source_desc:
            return True  # 已搬运过（重复轮询/重复触发时直接视为完成）

        await self._db_update_description(orig_hash, source_desc)
        self.ctx.logger.info(
            "[视觉增强·分镜] 多帧动画描述已搬运至原图记录 %s",
            orig_hash[:12],
        )
        return True

    def _warn_db_unavailable(self, reason: Any) -> None:
        """数据库能力不可用时只告警一次，避免轮询刷屏。"""
        if self._db_warned:
            return
        self._db_warned = True
        self.ctx.logger.warning(
            "[视觉增强·分镜] database.query 能力不可用（多帧描述无法搬运回原图，"
            "视觉识别会退化为只看首帧）：%s。请确认 _manifest.json 的 capabilities "
            '已声明 "database.query"，并重载插件',
            reason,
        )

    async def _db_get_image_record(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """按 hash 查询宿主 Images 表 IMAGE 记录，兼容不同的返回包装结构。"""
        try:
            result = await self.ctx.db.query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": image_hash, "image_type": "image"},
                limit=1,
                single_result=True,
            )
        except Exception as exc:  # noqa: BLE001 数据库不可达时静默放弃本轮
            self._warn_db_unavailable(exc)
            return None

        if not isinstance(result, dict):
            return None
        if result.get("success") is False:
            self._warn_db_unavailable(result.get("error") or "database.query 返回失败")
            return None
        if "description" in result:
            return result
        for key in ("data", "result", "items", "records"):
            inner = result.get(key)
            if isinstance(inner, dict) and "description" in inner:
                return inner
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                return inner[0]
        return None

    async def _db_update_description(self, image_hash: str, description: str) -> None:
        """把多帧描述写入原图 hash 的 Images 记录（刷新器即可安全回填，无首帧固化竞态）。

        只写 IMAGE 类型记录；EMOJI 记录的 description 兼作情绪标签来源
        （逗号分隔标签格式），不可被多帧长描述覆写。
        """
        try:
            await self.ctx.db.query(
                model_name="Images",
                query_type="update",
                data={"description": description, "vlm_processed": True},
                filters={"image_hash": image_hash, "image_type": "image"},
            )
        except Exception as exc:  # noqa: BLE001 更新失败不影响消息链
            self.ctx.logger.warning("[视觉增强·分镜] 更新原图描述失败 hash=%s: %s", image_hash[:12], exc)

    # ── 动漫角色识别（AnimeTrace / ai.animedb.cn） ──────────────────────

    def _gif_anime_active(self) -> bool:
        """GIF 分镜链路的动漫识别是否生效（总开关 + GIF 开关）。"""
        return bool(self._opt("anime", "enabled", True)) and bool(
            self._opt("anime", "process_gif", True)
        )

    def _remember_anime_attempted(self, orig_hash: str) -> None:
        """记录一次识图尝试：该图后续不再走快速路径"补课"，防重发循环。"""
        self._anime_attempted[orig_hash] = None
        self._anime_attempted.move_to_end(orig_hash)
        while len(self._anime_attempted) > ANIME_ATTEMPTED_MAX_ENTRIES:
            self._anime_attempted.popitem(last=False)

    def _dbg(self, msg: str, *args) -> None:
        """统一调试日志：调试开关开启时以 INFO 输出（便于排查），否则降为 debug。

        开关来源：「调试」页签的总开关，或「动漫识别」页签的调试日志（历史字段）。
        """
        enabled = bool(self._opt("debug", "enabled", False)) or bool(
            self._opt("anime", "debug_mode", False)
        )
        if enabled:
            self.ctx.logger.info(msg, *args)
        else:
            self.ctx.logger.debug(msg, *args)

    def _anime_cache_path(self) -> Optional[Path]:
        """识别缓存文件路径（宿主数据目录下），路径不可用时返回 None。"""
        try:
            data_dir = str(self.ctx.paths.data_dir or "").strip()
        except Exception:  # noqa: BLE001 paths 能力缺失时仅内存缓存
            return None
        return Path(data_dir) / "anime_recognition_cache.json" if data_dir else None

    def _anime_cache_load(self) -> Dict[str, str]:
        """懒加载磁盘缓存到内存；文件缺失/损坏视为空缓存（不阻断识别链路）。"""
        if self._anime_cache is not None:
            return self._anime_cache
        self._anime_cache = {}
        path = self._anime_cache_path()
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._anime_cache = {str(k): str(v) for k, v in data.items() if v}
            except Exception as exc:  # noqa: BLE001 缓存损坏只告警，识别照常
                self.ctx.logger.warning("[动漫识别] 识别缓存读取失败（忽略）: %s", exc)
        return self._anime_cache

    def _anime_cache_get(self, orig_hash: str) -> str:
        """查询识别缓存（cache_enabled 关闭时恒为空）。"""
        if not bool(self._opt("anime", "cache_enabled", True)):
            return ""
        return self._anime_cache_load().get(orig_hash, "")

    def _anime_cache_put(self, orig_hash: str, info: str) -> None:
        """写入识别缓存并落盘（仅缓存成功结果；写盘失败不影响识别链路）。"""
        if not info or not bool(self._opt("anime", "cache_enabled", True)):
            return
        cache = self._anime_cache_load()
        cache[orig_hash] = info
        while len(cache) > ANIME_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        path = self._anime_cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 写盘失败只告警，内存缓存仍生效
            self.ctx.logger.warning("[动漫识别] 识别缓存写入失败（忽略）: %s", exc)

    def _static_anime_active(self, comp_type: str) -> bool:
        """静态图片动漫识别是否对该组件生效（普通图片组件；表情包不参与）。"""
        return (
            comp_type == "image"
            and bool(self._opt("anime", "enabled", True))
            and bool(self._opt("anime", "process_static", True))
        )

    # ── 阻塞注入（首次发送即见角色标签） ────────────────────────────────

    VLM_DESCRIBE_PROMPT = (
        "请用中文详细描述这张图片的内容。如果有文字，请把文字描述概括出来，请留意其主题、直观感受，"
        "输出为一段平文本，最多100字，请注意不要分点，就输出一段文本"
    )

    def _blocking_inject_active(self, comp_type: str) -> bool:
        """阻塞注入是否对该组件生效（普通图片组件；表情包不参与）。"""
        return (
            comp_type == "image"
            and bool(self._opt("anime", "enabled", True))
            and bool(self._opt("anime", "inject_enabled", False))
        )

    async def _vlm_describe(self, image: bytes) -> str:
        """调麦麦 VLM 任务生成图片描述（复用宿主提示词风格），失败返回空串。

        任务名自动探测：优先宿主 `model_task_config` 中的 `vlm` 任务，其次
        名称含 vlm/image/vision 的任务；探测不到时放弃门控（返回空串，调用方
        转后台观察兜底），不硬编码任务名以兼容不同用户的任务命名。

        Args:
            image: 识图素材（JPEG）字节。
        """
        try:
            llm = self.ctx.llm
        except Exception:  # noqa: BLE001 llm 能力缺失时门控不可用
            self._dbg("[动漫识别] VLM 门控不可用（ctx.llm 缺失）")
            return ""

        task_name = ""
        try:
            available = await llm.get_available_models()
            names = [str(n) for n in (available or [])]
            if "vlm" in names:
                task_name = "vlm"
            else:
                task_name = next(
                    (n for n in names if any(k in n.lower() for k in ("vlm", "image", "vision"))),
                    "",
                )
        except Exception as exc:  # noqa: BLE001 探测失败按默认任务名尝试
            self._dbg("[动漫识别] VLM 任务名探测失败，按默认 'vlm' 尝试: %s", exc)
            task_name = "vlm"
        if not task_name:
            self._dbg(
                "[动漫识别] VLM 门控不可用：宿主 model_task_config 未找到 VLM 类任务，"
                "阻塞注入转后台观察兜底"
            )
            return ""
        self._dbg("[动漫识别] VLM 门控使用任务名: %s", task_name)

        # SDK 兼容：新版 generate 支持 task_name 具名参数（任务名语义）；
        # 旧版落 **kwargs 透传，宿主按自身语义解析。vlm_model 配置了具体
        # 模型名时优先用它（适配把 model 当模型名的宿主版本）。
        vlm_model = str(self._opt("anime", "vlm_model", "") or "").strip()
        try:
            response = await llm.generate(
                prompt=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.VLM_DESCRIBE_PROMPT},
                            {
                                "type": "image",
                                "image_format": "jpeg",
                                "image_base64": base64.b64encode(image).decode("ascii"),
                            },
                        ],
                    }
                ],
                model=vlm_model,
                task_name=task_name,
                temperature=0,
                max_tokens=220,
            )
        except Exception as exc:  # noqa: BLE001 门控失败不阻塞注入流程
            self._dbg("[动漫识别] VLM 门控调用失败: %s", exc)
            return ""
        if not isinstance(response, dict) or not response.get("success", False):
            self._dbg("[动漫识别] VLM 门控返回失败: %s", response)
            return ""
        return str(response.get("response") or "").strip()[:200]

    async def _blocking_inject_image(self, comp: Dict[str, Any]) -> "tuple[Any, Optional[str]]":
        """阻塞注入单张图片：在等待预算内完成识图并写进本轮消息。

        Returns:
            (新组件 or None=未处理, 追加文本 or None)
            - 门控开（inject_wait_vlm）：VLM 描述 → 关键词门控 → 识图 →
              「描述 + 角色」写进组件 data（宿主跳过重复识别，首次发送即完整生效）；
            - 门控关：直接识图 → 有结果时返回追加文本（图片组件不动，宿主照常识别）；
            - 多帧 GIF 返回 (comp, None)（分镜+描述搬运流程负责，注入跳过）；
            - VLM 失败/无二进制返回 (None, None)（调用方转后台观察兜底）。
        """
        orig_hash = str(comp.get("hash") or "").strip()
        raw = self._b64_decode(comp.get("binary_data_base64"))
        if not orig_hash or not raw:
            self._dbg("[动漫识别] 阻塞注入：组件无二进制数据，转后台观察 hash=%s", orig_hash[:12])
            return None, None
        if self._is_gif(raw):
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    if int(getattr(im, "n_frames", 1) or 1) >= 2:
                        return comp, None  # 多帧 GIF：分镜+搬运流程衔接识图
            except Exception:  # noqa: BLE001 打不开按静态图处理
                return None, None

        info = self._anime_cache_get(orig_hash)
        wait_vlm = bool(self._opt("anime", "inject_wait_vlm", True))
        desc = ""
        if wait_vlm:
            upload = await asyncio.to_thread(self._extract_anime_frame, raw) or raw
            desc = await self._vlm_describe(upload)
            if not desc:
                self._dbg("[动漫识别] 阻塞注入：VLM 描述失败，转后台观察 hash=%s", orig_hash[:12])
                return None, None
            if not info and not self._looks_anime(desc):
                # 门控未命中：只写描述（宿主跳过重复识别），不外发识图
                self._dbg(
                    "[动漫识别] 阻塞注入：VLM 描述未命中关键词，跳过识图 hash=%s", orig_hash[:12]
                )
                new_comp = dict(comp)
                new_comp["data"] = f"[图片：{desc}]"
                return new_comp, None

        if not info:
            try:
                frame = await asyncio.to_thread(self._extract_anime_frame, raw)
            except Exception as exc:  # noqa: BLE001 抽帧失败转观察兜底
                self._dbg("[动漫识别] 阻塞注入：识别帧抽取失败 hash=%s: %s", orig_hash[:12], exc)
                return None, None
            self._remember_anime_attempted(orig_hash)
            self.ctx.logger.info("[动漫识别] 阻塞注入：开始识图 hash=%s", orig_hash[:12])
            boxes = await self._animetrace_search(frame)
            info = self._format_anime_info(boxes)
            if info:
                self._anime_cache_put(orig_hash, info)
            else:
                self.ctx.logger.info(
                    "[动漫识别] 阻塞注入：接口无可用角色结果（可能非动漫图）hash=%s",
                    orig_hash[:12],
                )

        if wait_vlm:
            # 描述（+角色）写进组件 data：宿主见 content 非空跳过识别，
            # 本轮上下文立即拿到完整文本；观察任务把结果并入库描述供重发复用
            new_comp = dict(comp)
            new_comp["data"] = f"[图片：{desc}]\n{info}" if info else f"[图片：{desc}]"
            if info:
                self._spawn_static_anime_watch(orig_hash)
            return new_comp, None
        if info:
            return comp, info  # 图片组件不动，角色标签作为文本组件紧随其后
        return comp, None  # 无结果：原样放行（attempted 已记，观察不重复外发）

    async def _watch_static_anime(self, comp: Dict[str, Any]) -> None:
        """静态图片动漫识别登记：按 hash 去重后暂存原图字节并启动观察任务。

        静态图不参与本插件的替换/搬运流程（宿主原生识别），插件在
        before_process 趁二进制还在内存时登记原图字节，后台等宿主写出
        描述后再做关键词门控与识图。
        """
        if not self._static_anime_active(str(comp.get("type") or "").strip().lower()):
            return
        orig_hash = str(comp.get("hash") or "").strip()
        if not orig_hash or orig_hash in self._anime_watched:
            if orig_hash:
                self._dbg("[动漫识别] 静态图已登记过观察，跳过重复登记 hash=%s", orig_hash[:12])
            return
        raw = self._b64_decode(comp.get("binary_data_base64"))
        if not raw:
            self._dbg("[动漫识别] 组件无二进制数据，无法登记静态观察 hash=%s", orig_hash[:12])
            return
        self._anime_watched[orig_hash] = None
        self._anime_watched.move_to_end(orig_hash)
        while len(self._anime_watched) > ANIME_WATCHED_MAX_ENTRIES:
            self._anime_watched.popitem(last=False)
        if orig_hash not in self._anime_frames:
            self._anime_frames[orig_hash] = raw
            self._anime_frames.move_to_end(orig_hash)
            while len(self._anime_frames) > ANIME_FRAME_CACHE_MAX_ENTRIES:
                self._anime_frames.popitem(last=False)
        self._dbg("[动漫识别] 静态图观察已登记，等待宿主描述就绪 hash=%s", orig_hash[:12])
        self._spawn_static_anime_watch(orig_hash)

    def _spawn_static_anime_watch(self, orig_hash: str) -> None:
        """启动静态图片描述观察任务（识图在描述就绪后进行）。"""
        try:
            asyncio.get_running_loop().create_task(self._static_anime_watch(orig_hash))
        except RuntimeError:  # pragma: no cover 无事件循环时放弃观察
            self._anime_watched.pop(orig_hash, None)
            self._anime_frames.pop(orig_hash, None)

    async def _static_anime_watch(self, orig_hash: str) -> None:
        """静态图片动漫识别观察：轮询宿主识别结果，描述就绪后识图追加。

        超时放弃（如未配置视觉模型），识别帧随 finally 释放；轮询只读
        数据库，绝不影响宿主识别链路。
        """
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + RELOCATE_TIMEOUT_SECONDS
            self._dbg("[动漫识别] 静态图观察任务启动，等待宿主识别 hash=%s", orig_hash[:12])
            while loop.time() < deadline:
                record = await self._db_get_image_record(orig_hash)
                if (
                    record
                    and record.get("vlm_processed")
                    and str(record.get("description") or "").strip()
                ):
                    await self._maybe_recognize_anime(orig_hash)
                    return
                await asyncio.sleep(ANIME_STATIC_POLL_INTERVAL_SECONDS)
            self._dbg("[动漫识别] 静态图描述观察超时 hash=%s", orig_hash[:12])
        finally:
            self._anime_frames.pop(orig_hash, None)

    def _stash_anime_frame(self, orig_hash: str, data: bytes) -> None:
        """为动漫角色识别登记原图字节（key=原图 hash，任务完成/超时后释放）。

        只做零拷贝引用登记，不做任何图像处理——after_process 是阻塞钩子，
        必须保持零同步开销，识别帧由后台识图任务在描述就绪后现抽。
        """
        if not self._gif_anime_active():
            self._dbg("[动漫识别] GIF 识别未开启，跳过素材登记 hash=%s", orig_hash[:12])
            return
        if orig_hash in self._anime_frames:
            return
        self._anime_frames[orig_hash] = data
        self._anime_frames.move_to_end(orig_hash)
        while len(self._anime_frames) > ANIME_FRAME_CACHE_MAX_ENTRIES:
            self._anime_frames.popitem(last=False)

    def _extract_anime_frame(self, data: bytes) -> Optional[bytes]:
        """把原图编码为识图用 JPEG：GIF 挑最清晰的一帧，静态图整体降采样。

        送识别的是原动画单帧（而非合成网格图）——网格图上的序号、说明条
        与分格线会干扰人物检测；透明通道合成到白底，超长边降采样控制
        上传体积（接口限 4MB）。仅在后台识图任务中调用，不占用消息链。
        """
        with Image.open(io.BytesIO(data)) as im:
            n_frames = int(getattr(im, "n_frames", 1) or 1)
            best_idx, best_score = 0, -1.0
            for idx in self._sample_frame_indices(n_frames, min(n_frames, 12)):
                try:
                    im.seek(idx)
                    score = self._frame_sharpness(im)
                except Exception:  # noqa: BLE001 单帧失败继续抽下一帧
                    continue
                if score > best_score:
                    best_idx, best_score = idx, score
            im.seek(best_idx)
            rgba = im.convert("RGBA")
        frame = Image.new("RGB", rgba.size, (255, 255, 255))
        frame.paste(rgba, mask=rgba.getchannel("A"))
        if max(frame.size) > ANIME_FRAME_MAX_SIDE:
            scale = ANIME_FRAME_MAX_SIDE / max(frame.size)
            frame = frame.resize(
                (max(1, round(frame.width * scale)), max(1, round(frame.height * scale))),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        frame.save(buf, "JPEG", quality=90)
        return buf.getvalue()

    async def _maybe_recognize_anime(
        self, source_hash: str, target_hash: Optional[str] = None
    ) -> None:
        """描述就绪后的动漫角色识别：描述命中动漫关键词时补充识图结果。

        Args:
            source_hash: 描述来源记录的 hash（GIF=合成图记录；静态图=原图记录）。
            target_hash: 描述落点记录的 hash（GIF=原图记录，识别帧按它暂存；
                静态图与来源相同，省略）。

        - 关键词命中才调用第三方接口，真人照片等无关图片不外发；
        - 识别结果以「[动漫角色识别] 角色《作品》、…」追加进描述，来源与
          落点记录同步更新（前者供快速路径/观察任务复用，后者供宿主刷新器
          回填上下文）；
        - 描述已带识别标记（去重）或识图失败时静默返回，不影响主流程。
        """
        if not bool(self._opt("anime", "enabled", True)):
            return
        source = await self._db_get_image_record(source_hash)
        description = str((source or {}).get("description") or "").strip()
        if not description or ANIME_INFO_MARKER in description:
            self._dbg(
                "[动漫识别] %s，跳过识别 hash=%s",
                "描述为空" if not description else "描述已带识别标记（同图不重复识别）",
                (target_hash or source_hash)[:12],
            )
            return
        if not self._looks_anime(description):
            self._dbg(
                "[动漫识别] 描述未命中动漫关键词，跳过（可在配置调整关键词）hash=%s",
                (target_hash or source_hash)[:12],
            )
            return
        frame_hash = target_hash or source_hash

        # 识别缓存命中：跳过抽帧与接口调用，直接复用上次结果
        cached_info = self._anime_cache_get(frame_hash)
        if cached_info:
            self._anime_frames.pop(frame_hash, None)
            new_description = f"{description}\n{cached_info}"
            await self._db_update_description(source_hash, new_description)
            if target_hash and target_hash != source_hash:
                await self._db_update_description(target_hash, new_description)
            self.ctx.logger.info("[动漫识别] 缓存命中，识别结果已并入描述 hash=%s", frame_hash[:12])
            return

        frame_bytes_stashed = self._anime_frames.pop(frame_hash, None)
        if not frame_bytes_stashed:
            self.ctx.logger.info(
                "[动漫识别] 识别素材缺失（功能刚开启/并发超上限），跳过 hash=%s", frame_hash[:12]
            )
            return
        # 暂存的均为原图原始字节（零拷贝登记，不占消息链耗时），
        # 识图素材（JPEG 帧/降采样图）由本后台任务现抽
        if frame_bytes_stashed[:2] == b"\xff\xd8" and len(frame_bytes_stashed) <= ANIME_UPLOAD_MAX_BYTES:
            frame = frame_bytes_stashed
        else:
            try:
                frame = await asyncio.to_thread(self._extract_anime_frame, frame_bytes_stashed)
            except Exception as exc:  # noqa: BLE001 抽帧失败只影响角色识别
                self._dbg("[动漫识别] 识别帧抽取失败 hash=%s: %s", frame_hash[:12], exc)
                return

        self._remember_anime_attempted(frame_hash)
        self.ctx.logger.info(
            "[动漫识别] 描述命中动漫关键词，开始识图 hash=%s", frame_hash[:12]
        )
        boxes = await self._animetrace_search(frame)
        info = self._format_anime_info(boxes)
        if not info:
            self.ctx.logger.info(
                "[动漫识别] 接口无可用角色结果（可能非动漫图）hash=%s",
                frame_hash[:12],
            )
            return
        self._anime_cache_put(frame_hash, info)
        new_description = f"{description}\n{info}"
        await self._db_update_description(source_hash, new_description)
        if target_hash and target_hash != source_hash:
            await self._db_update_description(target_hash, new_description)
        self.ctx.logger.info("[动漫识别] 角色识别结果已并入描述 hash=%s", frame_hash[:12])

    async def _animetrace_search(self, image: bytes) -> Optional[List[Any]]:
        """调用 AnimeTrace 识图接口，返回人物检测框列表；失败返回 None。

        接口为 multipart/form-data 上传（is_multi=1 返回每个检测框的完整
        候选列表，低置信框候选可能多达数十个，由 _format_anime_info 取前
        N 个；is_multi=0 时每人只返回 1 个候选）；业务码 0/200/17720/17721
        均视为成功（官方文档存在两代状态码体系）。
        """
        url = str(self._opt("anime", "api_url", ANIMETRACE_API_URL) or "").strip() or ANIMETRACE_API_URL
        try:
            timeout = max(5.0, float(self._opt("anime", "timeout_seconds", 15.0) or 15.0))
        except (TypeError, ValueError):
            timeout = 15.0
        model = str(self._opt("anime", "model", "") or "").strip()

        form = aiohttp.FormData()
        form.add_field("is_multi", "1")
        if model:
            form.add_field("model", model)
        form.add_field("file", image, filename="frame.jpg", content_type="image/jpeg")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, data=form, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text(errors="replace")).strip()[:300]
                        self.ctx.logger.warning("[动漫识别] 接口 HTTP %s: %s", resp.status, body)
                        return None
                    payload = await resp.json(content_type=None)
        except asyncio.TimeoutError:
            self.ctx.logger.warning("[动漫识别] 接口请求超时（%.0fs）", timeout)
            return None
        except (aiohttp.ClientError, ValueError) as exc:  # noqa: BLE001 网络层失败只记日志
            self.ctx.logger.warning("[动漫识别] 接口请求失败: %s", exc)
            return None

        if not isinstance(payload, dict):
            self.ctx.logger.warning("[动漫识别] 接口返回非 JSON 结构")
            return None
        code = payload.get("code")
        if code not in (0, 200, 17720, 17721):
            self.ctx.logger.warning(
                "[动漫识别] 接口返回错误 code=%s: %s", code, payload.get("message") or payload.get("msg")
            )
            return None
        data = payload.get("data")
        return data if isinstance(data, list) else []

    def _format_anime_info(self, boxes: Optional[List[Any]]) -> str:
        """把识图检测结果格式化为一行追加信息；无有效角色返回空串。

        筛选：开启 filter_not_confident 时，存在高置信检测框则跳过官方标记
        为低置信（候选过多、需人工确认）的检测框——宁可缺判也不错判；全部
        为低置信时兜底写入（带「（低置信）」后缀标注）。关闭过滤时低置信
        检测框始终写入并标注，提示麦麦谨慎引用。
        每个人物取按可能性排序的前 max_candidates 个候选，重复去重；
        人物数上限 max_characters。

        智能筛选（smart_filter，默认开）：一人写了多个候选（候选较多）或
        总人数 ≥ ANIME_WORK_OMIT_PEOPLE（人数较多）时只写角色名、省略《作品》
        控制描述长度；关闭则始终带作品名。
        """
        if not boxes:
            return ""
        try:
            max_characters = max(1, int(self._opt("anime", "max_characters", 5) or 5))
        except (TypeError, ValueError):
            max_characters = 5
        try:
            max_candidates = max(1, int(self._opt("anime", "max_candidates", 2) or 2))
        except (TypeError, ValueError):
            max_candidates = 2
        filter_low = bool(self._opt("anime", "filter_not_confident", True))
        smart_filter = bool(self._opt("anime", "smart_filter", True))
        # 条件过滤：存在高置信检测框时丢弃低置信框；全部低置信时兜底写入
        usable = [
            (box, bool(box.get("not_confident")))
            for box in boxes
            if isinstance(box, dict)
            and isinstance(box.get("character"), list)
            and box.get("character")
        ]
        drop_low = filter_low and any(not low for _, low in usable)
        people_candidates: List[List[tuple[str, str]]] = []
        people_low: List[bool] = []
        seen: "set[str]" = set()
        for box, low in usable:
            if drop_low and low:
                continue
            candidates = box.get("character")
            if not isinstance(candidates, list) or not candidates:
                continue
            # 单个人物的候选列表：按可能性顺序取前 max_candidates 个有效项
            alts: List[tuple[str, str]] = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                name = str(cand.get("character") or "").strip()
                work = str(cand.get("work") or "").strip()
                if not name:
                    continue
                key = f"{name}|{work}"
                if key in seen:
                    continue
                seen.add(key)
                alts.append((name, work))
                if len(alts) >= max_candidates:
                    break
            if alts:
                people_candidates.append(alts)
                people_low.append(low)
            if len(people_candidates) >= max_characters:
                break
        if not people_candidates:
            return ""
        # 智能筛选：候选较多（一人多备选）或总人数较多 → 只写角色名
        omit_work = smart_filter and (
            len(people_candidates) >= ANIME_WORK_OMIT_PEOPLE
            or any(len(alts) > 1 for alts in people_candidates)
        )
        people: List[str] = []
        for idx, alts in enumerate(people_candidates):
            suffix = "（低置信）" if people_low[idx] else ""
            if omit_work:
                names = list(dict.fromkeys(name for name, _ in alts))
                people.append(" 或 ".join(names) + suffix)
            else:
                people.append(" 或 ".join(f"{n}《{w}》" if w else n for n, w in alts) + suffix)
        if not people:
            return ""
        return f"{ANIME_INFO_MARKER} {'、'.join(people)}"

    async def _process_component(self, comp: Any) -> List[Any]:
        """处理单个消息组件，返回替换后的组件列表。

        组件 data（content）默认保持为空、交给宿主视觉链路；唯一例外是
        多帧描述已就绪的重复 GIF——把已就绪描述直接写入 data，宿主见
        content 非空跳过识别，planner 上下文立即拿到多帧描述。
        """
        if not isinstance(comp, dict):
            return [comp]

        comp_type = str(comp.get("type") or "").strip().lower()
        if not self._component_managed(comp_type):
            return [comp]

        b64_data = comp.get("binary_data_base64")
        if not isinstance(b64_data, str) or not b64_data:
            return [comp]

        try:
            raw = base64.b64decode(b64_data)
        except Exception:
            return [comp]
        if not self._is_gif(raw):
            return [comp]

        merged = await self._merge_with_cache(raw)
        if merged is None:
            return [comp]  # 单帧 GIF / 解码失败 / 合成异常：原样放行

        merged_b64 = base64.b64encode(merged).decode("ascii")
        merged_hash = hashlib.sha256(merged).hexdigest()
        orig_hash = str(comp.get("hash") or "").strip() or hashlib.sha256(raw).hexdigest()

        relocated = await self._lookup_ready_storyboard_description(orig_hash, merged_hash)
        if relocated:
            # 该 GIF 的多帧描述此前已搬运就绪：直接写入图片组件文本，
            # 宿主对 content 非空的组件跳过识别——零额外识别，上下文
            # 立即拿到多帧描述（重发场景无延迟）。
            new_comp = dict(comp)
            new_comp["data"] = f"[图片：{relocated}]"
            self.ctx.logger.info(
                "[视觉增强·分镜] 命中已就绪的分镜描述，直接写入图片组件文本 hash=%s", orig_hash[:12]
            )
            return [new_comp]

        # 替换模式——组件二进制与 hash 暂时替换为合成图，宿主识别的
        # 就是多帧网格图（识别次数 2→1，且原图记录不存在首帧描述，无固化
        # 竞态）；after_process 在落库前用暂存的原图字节恢复组件，原图文件
        # 与图片记录全部保真。替换登记（索引/暂存字节）由 handle_before_process
        # 的外层循环完成。
        new_comp = dict(comp)
        new_comp["binary_data_base64"] = merged_b64
        new_comp["hash"] = merged_hash
        self.ctx.logger.info(
            "[视觉增强·分镜] 图片组件已替换为合成图 %.1fKB（落库前恢复原图）", len(merged) / 1024
        )
        return [new_comp]

    async def _lookup_ready_storyboard_description(self, orig_hash: str, merged_hash: str) -> str:
        """若该 GIF 的多帧描述此前已搬运就绪，返回可直接写入组件文本的描述。

        命中条件（全部满足）：
        - 合成图记录（IMAGE 类型）已完成识别且有非空描述——搬运完成后
          merged_hash 名下的记录即持久化的多帧描述；同图重发时帧抽样与
          JPEG 编码确定，merged_hash 一致，可稳定命中；
        - 原图记录存在且 no_file_flag=False（文件在库中，跳过识别
          不会影响落库与展示）。

        动漫识别"补课"：描述已就绪但还没有识图标记、且命中动漫关键词、
        且本次会话没尝试过识图时，放弃快速路径改走替换+搬运流程，让
        `_maybe_recognize_anime` 有机会补一次识图（宿主对同图识别基本
        缓存命中，几乎零成本；识图无结果/失败的图由 `_anime_attempted`
        挡住，不会形成重发循环）。

        未命中返回空字符串，走替换 + 搬运流程。
        """
        merged = await self._db_get_image_record(merged_hash)
        if not (merged and merged.get("vlm_processed")):
            return ""
        description = str(merged.get("description") or "").strip()
        if not description:
            return ""
        if (
            ANIME_INFO_MARKER not in description
            and self._gif_anime_active()
            and self._looks_anime(description)
            and orig_hash not in self._anime_attempted
        ):
            self._dbg(
                "[动漫识别] 描述缺识别标记且命中关键词，放弃快速路径补一次识图 hash=%s",
                orig_hash[:12],
            )
            return ""  # 放弃快速路径，走替换+搬运以衔接动漫识图
        target = await self._db_get_image_record(orig_hash)
        if target is None or target.get("no_file_flag"):
            return ""
        return description

    @staticmethod
    def _is_gif(data: bytes) -> bool:
        """按文件头魔数判断是否 GIF。"""
        return len(data) >= 6 and data[:6] in GIF_MAGICS

    # ── 合成（带缓存） ──────────────────────────────────────────────────

    async def _merge_with_cache(self, raw: bytes) -> Optional[bytes]:
        """合成 GIF 帧网格图，命中缓存时直接复用。失败返回 None（原样放行）。"""
        cache_key = f"{hashlib.sha256(raw).hexdigest()}|{self._settings_fingerprint()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            merged = await asyncio.to_thread(self._merge_gif_frames, raw)
        except Exception as exc:  # noqa: BLE001 合成失败绝不能阻塞消息链
            self.ctx.logger.warning("[视觉增强·分镜] 帧合成失败，原图原样放行: %s", exc)
            return None

        if merged is None:
            return None

        self._cache_put(cache_key, merged)
        return merged

    def _cache_get(self, key: str) -> Optional[bytes]:
        cached = self._merge_cache.get(key)
        if cached is not None:
            self._merge_cache.move_to_end(key)  # 刷新热度
        return cached

    def _cache_put(self, key: str, value: bytes) -> None:
        self._merge_cache[key] = value
        while len(self._merge_cache) > MERGE_CACHE_MAX_ENTRIES:
            self._merge_cache.popitem(last=False)

    def _merge_gif_frames(self, data: bytes) -> Optional[bytes]:
        """把多帧 GIF 均匀抽样后拼成网格静态图。

        Returns:
            合成图 bytes；帧数不足 min_frames（视为静态图）时返回 None。
        Raises:
            解码失败等异常向上抛出，由调用方兜底。
        """
        max_frames = max(2, int(self._opt("merge", "max_frames", 9)))
        min_frames = max(2, int(self._opt("merge", "min_frames", 2)))
        draw_index = bool(self._opt("merge", "draw_frame_index", True))
        output_format = str(self._opt("output", "output_format", "jpeg") or "jpeg").lower()
        jpeg_quality = int(self._opt("output", "jpeg_quality", 85))
        max_cell = max(64, int(self._opt("output", "max_cell_size", 480)))
        gap = max(0, int(self._opt("output", "grid_gap", 4)))

        with Image.open(io.BytesIO(data)) as im:
            n_frames = int(getattr(im, "n_frames", 1) or 1)
            if n_frames < min_frames:
                return None

            count = min(n_frames, max_frames)
            if bool(self._opt("merge", "adaptive_frames", True)):
                adaptive = self._adaptive_frame_count(im, n_frames, count)
                if adaptive != count:
                    self.ctx.logger.info(
                        "[视觉增强·分镜] 自适应帧数: %d 帧（运动量与源帧数调节，上限 %d）", adaptive, count
                    )
                count = max(min_frames, min(count, adaptive))

            indices = self._pick_frame_indices(im, n_frames, count)

            frames = []
            for idx in indices:
                im.seek(idx)
                # 转 RGBA：保留透明通道，粘贴时按 alpha 混合到白底
                frames.append(im.convert("RGBA"))

        if not frames:
            return None

        # 网格布局：尽量接近正方形
        count = len(frames)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)

        # 单帧格子尺寸：取各帧最大宽高，超出上限等比缩小
        cell_w = max(frame.width for frame in frames)
        cell_h = max(frame.height for frame in frames)
        scale = min(1.0, max_cell / cell_w, max_cell / cell_h)
        if scale < 1.0:
            cell_w = max(1, round(cell_w * scale))
            cell_h = max(1, round(cell_h * scale))

        # 顶部说明条：极简文案 + 字号自适应，保证任何画布宽度都完整显示
        caption_text = ""
        caption_font = None
        caption_h = 0
        if bool(self._opt("merge", "storyboard_caption", True)):
            canvas_w = cols * cell_w + gap * (cols + 1)
            target_w = max(60, canvas_w - gap * 2 - 4)
            for size in range(18, 11, -2):
                cand_font, has_cjk = self._load_caption_font(size)
                cand_text = self._caption_text(count, has_cjk)
                caption_font, caption_text = cand_font, cand_text
                try:
                    bbox = cand_font.getbbox(cand_text)
                    if (bbox[2] - bbox[0]) <= target_w:
                        break  # 当前字号能放下，就用它
                except Exception:
                    break
            try:
                bbox = caption_font.getbbox(caption_text)
                caption_h = (bbox[3] - bbox[1]) + 10
            except Exception:
                caption_h = 0

        canvas = Image.new(
            "RGB",
            (cols * cell_w + gap * (cols + 1), caption_h + rows * cell_h + gap * (rows + 1)),
            (255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)
        if caption_h and caption_font is not None:
            draw.text((gap + 2, 4), caption_text, font=caption_font, fill=(17, 17, 17))
        font = self._load_caption_font(max(14, min(cell_w, cell_h) // 9))[0] if draw_index else None
        stroke_w = max(1, round(getattr(font, "size", 14) / 10)) if font is not None else 0

        for i, frame in enumerate(frames):
            row, col = divmod(i, cols)
            # 各帧尺寸可能略有差异：等比缩到格子内并居中
            fw, fh = frame.width, frame.height
            frame_scale = min(cell_w / fw, cell_h / fh, 1.0)
            if frame_scale < 1.0:
                frame = frame.resize((max(1, round(fw * frame_scale)), max(1, round(fh * frame_scale))), Image.LANCZOS)
            x = gap + col * (cell_w + gap) + (cell_w - frame.width) // 2
            y = caption_h + gap + row * (cell_h + gap) + (cell_h - frame.height) // 2

            canvas.paste(frame, (x, y), frame)  # 第三参数=alpha 遮罩
            if font is not None:
                # 黑字白描边：浅色/深色背景上都清晰
                draw.text(
                    (x + 4, y + 2),
                    str(i + 1),
                    font=font,
                    fill=(17, 17, 17),
                    stroke_width=stroke_w,
                    stroke_fill=(255, 255, 255),
                )

        buf = io.BytesIO()
        if output_format == "png":
            canvas.save(buf, "PNG")
        else:
            canvas.save(buf, "JPEG", quality=jpeg_quality)

        merged = buf.getvalue()
        self.ctx.logger.info(
            "[视觉增强·分镜] %d/%d 帧 -> %dx%d 网格图，%.1fKB -> %.1fKB（%s）",
            count,
            n_frames,
            canvas.width,
            canvas.height,
            len(data) / 1024,
            len(merged) / 1024,
            output_format,
        )
        return merged

    @staticmethod
    def _sample_frame_indices(n_frames: int, count: int) -> List[int]:
        """按时序均匀抽取 count 个帧索引（含首尾帧）。"""
        if count >= n_frames:
            return list(range(n_frames))
        if count <= 1:
            return [0]
        indices: List[int] = []
        for i in range(count):
            idx = round(i * (n_frames - 1) / (count - 1))
            if not indices or idx != indices[-1]:
                indices.append(idx)
        return indices

    @staticmethod
    def _frame_sharpness(frame: "Image.Image") -> float:
        """帧清晰度近似值：灰度边缘强度 RMS，运动模糊帧显著偏低。"""
        try:
            edges = frame.convert("L").filter(ImageFilter.FIND_EDGES)
            return float(sum(ImageStat.Stat(edges).rms))
        except Exception:
            return 0.0

    @staticmethod
    def _dhash(frame: "Image.Image") -> int:
        """感知哈希（dhash，8x8=64bit）：结构相似的画面之间距离很小。"""
        small = frame.convert("L").resize((9, 8), Image.LANCZOS)
        pixels = list(small.getdata())
        bits = 0
        for row in range(8):
            base = row * 9
            for col in range(8):
                bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
        return bits

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        """两个感知哈希的汉明距离。"""
        return bin(a ^ b).count("1")

    def _adaptive_frame_count(self, im: "Image.Image", n_frames: int, cap: int) -> int:
        """根据运动量自适应决定实际抽帧数（介于 min_frames 与 cap 之间）。

        粗抽 ≤24 个样本帧，以相邻帧感知哈希（dhash）的平均汉明距离衡量运动量：
        平均距离 ≤8 视为几乎静止/循环重复（取最少帧数），≥24 视为运动丰富
        （取最多帧数），中间线性插值。
        """

        def clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))

        min_frames = max(2, int(self._opt("merge", "min_frames", 2)))
        if cap <= min_frames:
            return cap

        sample_idx = self._sample_frame_indices(n_frames, min(n_frames, 24))
        dists: List[int] = []
        prev_dhash = None
        for idx in sample_idx:
            try:
                im.seek(idx)
                dhash = self._dhash(im)
            except Exception:
                continue
            if prev_dhash is not None:
                dists.append(self._hamming(prev_dhash, dhash))
            prev_dhash = dhash

        avg_dist = sum(dists) / len(dists) if dists else 0.0
        motion = clamp01((avg_dist - 8.0) / 16.0)
        return min(cap, min_frames + round((cap - min_frames) * motion))

    @classmethod
    def _pick_frame_indices(cls, im: "Image.Image", n_frames: int, count: int) -> List[int]:
        """分段清晰度抽帧：首尾帧固定保留，中间时间轴均分 count-2 段。

        每段先取清晰度最高的帧；若它与上一选中帧几乎相同（感知哈希距离过小），
        改选段内与上一帧差异最大的一帧——让"新元素入画"（如手出现、姿态突变）
        这类变化帧有机会被保留，而不是连续选中相似姿态。

        等间隔抽帧容易命中快速运动的模糊中间帧，视觉模型对模糊帧的描述
        常与实际内容偏差很大；清晰度与差异度结合，既保持时序覆盖，
        又尽量覆盖动画中的每次变化。
        """
        if count >= n_frames:
            return list(range(n_frames))
        if count < 3:
            return cls._sample_frame_indices(n_frames, count)

        middle = [
            c
            for c in cls._sample_frame_indices(n_frames, min(n_frames - 2, (count - 2) * 3))
            if 0 < c < n_frames - 1
        ]
        if not middle:
            return cls._sample_frame_indices(n_frames, count)

        try:
            im.seek(0)
            prev_dhash = cls._dhash(im)
        except Exception:
            prev_dhash = 0

        picked = [0]
        step = len(middle) / (count - 2)
        for seg_i in range(count - 2):
            lo = min(int(round(seg_i * step)), len(middle) - 1)
            hi = max(min(int(round((seg_i + 1) * step)), len(middle)), lo + 1)
            best_idx, best_score, best_dhash = middle[lo], -1.0, 0
            far_idx, far_dist, far_dhash = middle[lo], -1, 0
            for cand in middle[lo:hi]:
                try:
                    im.seek(cand)
                    gray = im.convert("L")
                except Exception:
                    continue
                score = cls._frame_sharpness(gray)
                dhash = cls._dhash(gray)
                dist = cls._hamming(prev_dhash, dhash)
                if score > best_score:
                    best_idx, best_score, best_dhash = cand, score, dhash
                if dist > far_dist:
                    far_idx, far_dist, far_dhash = cand, dist, dhash

            chosen_idx, chosen_dhash = best_idx, best_dhash
            if cls._hamming(prev_dhash, best_dhash) < DHASH_DUPLICATE_DISTANCE:
                # 段内最清晰帧与上一选中帧几乎相同：改选与上一帧差异最大的帧
                chosen_idx, chosen_dhash = far_idx, far_dhash
            if chosen_idx != picked[-1]:
                picked.append(chosen_idx)
            prev_dhash = chosen_dhash
        picked.append(n_frames - 1)

        picked = sorted(set(picked))
        if len(picked) < count:
            for cand in cls._sample_frame_indices(n_frames, count):
                if cand not in picked:
                    picked.append(cand)
                    if len(picked) >= count:
                        break
            picked.sort()
        return picked[:count]

    @staticmethod
    def _load_font(size: int):
        """加载序号字体：Pillow>=10.1 可指定字号，旧版退回内置字体。"""
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    _CJK_FONT_CANDIDATES: "tuple[str, ...]" = (
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        # Linux（麦麦常见部署环境）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    )

    @classmethod
    def _load_caption_font(cls, size: int) -> "tuple[Any, bool]":
        """加载说明条字体：优先系统中文字体，找不到退回 Pillow 内置字体（仅拉丁）。

        Returns:
            (字体对象, 是否中文字体)。
        """
        for path in cls._CJK_FONT_CANDIDATES:
            try:
                return ImageFont.truetype(path, size=size), True
            except Exception:  # noqa: BLE001 字体缺失/格式不支持一律尝试下一个
                continue
        return cls._load_font(size), False

    @staticmethod
    def _caption_text(count: int, has_cjk: bool) -> str:
        """说明条文案：极简锚点 + 时序连播提示，帮助视觉模型把跨帧动作连成因果。"""
        if has_cjk:
            return f"帧序列 · 共 {count} 帧（按序号连播）"
        return f"Frame sequence: {count} frames, play in order"

    # ── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        self.ctx.logger.info(
            "[视觉增强·分镜] 插件已加载 | 总开关: %s | 处理图片: %s | 最多 %s 帧 | 输出: %s | 动漫识别: %s",
            self._enabled(),
            bool(self._opt("merge", "process_image", True)),
            self._opt("merge", "max_frames", 9),
            self._opt("output", "output_format", "jpeg"),
            bool(self._opt("anime", "enabled", True)),
        )
        await self._check_db_access()

    async def _check_db_access(self) -> None:
        """启动自检：确认 manifest 的 capabilities 已包含 database.query。

        描述搬运依赖 ``ctx.db``；宿主 AuthorizationManager 只对 manifest 里
        声明过的能力签发令牌，未声明时所有 ``database.query`` 调用都会被拒绝，
        插件的视觉增强效果会在"描述回填"这一步静默失效。这里主动探一次，
        把问题在日志里暴露出来。
        """
        try:
            result = await self.ctx.db.query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": "__gif_storyboard_probe__", "image_type": "image"},
                limit=1,
                single_result=True,
            )
        except Exception as exc:  # noqa: BLE001 仅做能力可用性提示
            self._warn_db_unavailable(exc)
            return
        if isinstance(result, dict) and result.get("success") is False:
            self._warn_db_unavailable(result.get("error") or "database.query 返回失败")
            return
        self.ctx.logger.info("[视觉增强·分镜] 数据库能力自检通过（database.query 可用，描述搬运就绪）")

    async def on_unload(self) -> None:
        self._merge_cache.clear()
        self._original_bytes.clear()
        self._replaced_index.clear()
        self._relocate_tasks.clear()
        self._anime_frames.clear()
        self._anime_watched.clear()
        self._anime_attempted.clear()
        self._anime_cache = None
        self._db_warned = False
        self.ctx.logger.info("[视觉增强·分镜] 插件已卸载")

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        self._merge_cache.clear()
        self.ctx.logger.info("[视觉增强·分镜] 配置已更新（scope=%s version=%s），合成缓存已清空", scope, version)



