"""视频理解能力（mixin 模块，B站与抖音）

移植自 YukiSakiko/content_understanding_plugin（MIT；致谢与许可见 CHANGELOG 与
LICENSE）。凭据仅支持在配置里手填 Cookie，不依赖 Command / 二维码 / 发图能力。

- ``chat.receive.after_process`` Hook（BLOCKING）：自动检测入站消息中的
  B 站或抖音视频链接，把视频信息、章节要点与官方 AI 总结附加到消息内容末尾
  （写入上下文历史），**不主动发送任何回复**；
- ``parse_bilibili_video`` / ``parse_douyin_video`` Tool：供 planner 按需
  显式解析指定的视频。

B 站官方 AI 总结需要登录态（在配置里填 SESSDATA 等 Cookie）；未登录时仍可
获取视频基本信息。抖音 AI 总结与章节要点需要抖音 Cookie（选填）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from bilibili_api import Credential
from bilibili_api.video import Video
from maibot_sdk import Field, HookHandler, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

# ============ 链接识别 ============

_BV_PATTERN = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
_AV_PATTERN = re.compile(r"\bav(\d{6,})\b", re.IGNORECASE)
_BILI_VIDEO_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?bilibili\.com/video/(?:BV[0-9A-Za-z]{10}|av\d+)",
    re.IGNORECASE,
)
_B23_SHORT_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:b23\.tv|bili2233\.cn)/[0-9A-Za-z\-_]+",
    re.IGNORECASE,
)

_DOUYIN_SHORT_PATTERN = re.compile(
    r"(?:https?://)?(?:v|jx)\.douyin\.com/[0-9A-Za-z_\-]+",
    re.IGNORECASE,
)
_DOUYIN_WEB_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|m\.|iesdouyin\.com/share/|jingxuan\.)?douyin\.com/(?:video|note|share/(?:video|note)|m/(?:video|note))/(\d+)",
    re.IGNORECASE,
)
_DOUYIN_MODAL_PATTERN = re.compile(r"[?&]modal_id=(\d+)", re.IGNORECASE)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
_IOS_UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"

# ============ 配置模型 ============


class ParseSectionConfig(PluginConfigBase):
    """视频链接解析设置。"""

    __ui_label__ = "视频解析"
    __ui_icon__ = "link"
    __ui_order__ = 1

    enable_ai_summary: bool = Field(
        default=True,
        description="获取官方 AI 视频总结（B站/抖音）",
        json_schema_extra={
            "label": "官方 AI 总结",
            "hint": "开=附带官方总结；B站需登录态（在「视频凭据」填 SESSDATA），部分视频本身没有总结",
        },
    )
    enable_douyin: bool = Field(
        default=True,
        description="启用抖音视频解析（信息 / 章节要点 / AI 总结）",
        json_schema_extra={
            "label": "抖音解析",
            "hint": "开=解析抖音链接；关闭后只处理 B 站链接",
        },
    )
    enable_in_group: bool = Field(
        default=True,
        description="群聊中自动附加视频信息到上下文",
        json_schema_extra={
            "label": "群聊附加",
            "hint": "开=群聊出现视频链接时，把信息附加到上下文（不主动回复）",
        },
    )
    enable_in_private: bool = Field(
        default=True,
        description="私聊中自动附加视频信息到上下文",
        json_schema_extra={
            "label": "私聊附加",
            "hint": "开=私聊出现视频链接时，把信息附加到上下文（不主动回复）",
        },
    )
    cache_ttl_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        description="视频结果缓存时长（秒）",
        json_schema_extra={
            "label": "缓存时长（秒）",
            "hint": "同一视频的结果复用时长，默认 1800（30 分钟）；保存配置会立即清缓存",
        },
    )


class CredentialSectionConfig(PluginConfigBase):
    """视频凭据（B站与抖音 Cookie）。"""

    __ui_label__ = "视频凭据"
    __ui_icon__ = "key"
    __ui_order__ = 2

    sessdata: str = Field(
        default="",
        description="B站 Cookie - SESSDATA",
        json_schema_extra={
            "label": "B站 SESSDATA",
            "hint": "只填值本身（不含键名/引号/分号）；留空则拿不到官方 AI 总结",
        },
    )
    bili_jct: str = Field(
        default="",
        description="B站 Cookie - bili_jct",
        json_schema_extra={
            "label": "B站 bili_jct",
            "hint": "只填值本身；建议与 SESSDATA 一起填，缺少时可能校验不过",
        },
    )
    buvid3: str = Field(
        default="",
        description="B站 Cookie - buvid3",
        json_schema_extra={
            "label": "B站 buvid3",
            "hint": "只填值本身；缺少时可能触发风控",
        },
    )
    dedeuserid: str = Field(
        default="",
        description="B站 Cookie - DedeUserID",
        json_schema_extra={
            "label": "B站 DedeUserID",
            "hint": "只填值本身（纯数字）",
        },
    )
    douyin_cookie: str = Field(
        default="",
        description="抖音 Cookie（选填：用于获取抖音AI总结与章节要点）",
        json_schema_extra={
            "label": "抖音 Cookie（整串）",
            "hint": "浏览器复制的整串 Cookie：k=v; k2=v2（也支持 Netscape/cookies.txt 导出格式）；留空则只取视频基础信息；不含 ttwid 时插件自动申请游客标识",
            "placeholder": "ttwid=xxx; msToken=xxx; odin_tt=xxx",
        },
    )





# ============ 工具函数 ============


def _format_count(value: Any) -> str:
    """格式化播放/点赞等计数（1.2万）"""
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        return "0"
    if n >= 100_000_000:
        return f"{n / 100_000_000:.1f}亿"
    if n >= 10_000:
        return f"{n / 10_000:.1f}万"
    return str(n)


def _format_duration(seconds: Any) -> str:
    """格式化视频时长（分:秒）"""
    try:
        s = int(seconds or 0)
    except (TypeError, ValueError):
        return "未知"
    return f"{s // 60}:{s % 60:02d}"


def _extract_ai_summary(data: Any) -> str:
    """从 get_ai_conclusion 返回中提取总结文本（bilibili_api 版本间结构略有差异）。"""
    if not isinstance(data, dict):
        return ""
    result = data.get("data") if isinstance(data.get("data"), dict) else data
    if not isinstance(result, dict):
        return ""
    model_result = result.get("model_result") or {}
    summary = model_result.get("summary") if isinstance(model_result, dict) else ""
    return str(summary or "").strip()


def _clean_html_tags(text: str) -> str:
    """清理返回文本中的 HTML 标签（如 <mark> 等）"""
    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_cookie_text(text: str) -> dict[str, str]:
    """解析 Cookie 文本，自动兼容 Netscape 制表符格式与分号键值对格式。"""
    res: dict[str, str] = {}
    text = (text or "").strip()
    if not text:
        return res
    if "# Netscape" in text or "\t" in text:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                res[parts[5].strip()] = parts[6].strip()
            elif len(parts) == 2:
                res[parts[0].strip()] = parts[1].strip()
    else:
        for item in text.split(";"):
            if "=" in item:
                k, v = item.strip().split("=", 1)
                res[k.strip()] = v.strip()
    return res


# ============ 凭证管理 ============


class CredentialManager:
    """B站凭证管理：Cookie 持久化与有效性缓存。

    精简自上游实现：**去掉扫码登录**（不再需要二维码/发图能力）。凭证来源按
    优先级：数据目录 ``credential.json``（手工放置或历史扫码写入）> 插件配置里
    手填的 Cookie 字段。SDK 不提供写配置能力，因此凭证持久化在数据目录而非
    config.toml。
    """

    CHECK_INTERVAL_SECONDS = 1800  # 凭证有效性缓存时长

    def __init__(self, data_dir: Path, cookie_config: "CredentialSectionConfig", client: httpx.AsyncClient):
        self._data_dir = data_dir
        self._cookie_config = cookie_config
        self._client = client
        self._cred_file = data_dir / "credential.json"
        self._cached_credential: Optional[Credential] = None
        self._cached_at = 0.0
        self.last_failure: str = ""
        """最近一次没拿到可用凭据的原因（供有 logger 的插件侧输出）。"""

    def _load_from_file(self) -> Optional[Credential]:
        if not self._cred_file.exists():
            return None
        try:
            cookies = json.loads(self._cred_file.read_text(encoding="utf-8"))
            return Credential.from_cookies(cookies) if cookies.get("SESSDATA") else None
        except Exception:
            return None

    def _load_from_config(self) -> Optional[Credential]:
        cfg = self._cookie_config
        if not cfg.sessdata:
            return None
        return Credential.from_cookies(
            {
                "SESSDATA": cfg.sessdata,
                "bili_jct": cfg.bili_jct,
                "buvid3": cfg.buvid3,
                "DedeUserID": cfg.dedeuserid,
            }
        )

    async def get_credential(self) -> Optional[Credential]:
        """获取可用凭证：文件 > 配置；有效性 30 分钟内缓存。

        没拿到凭证时把原因写进 ``last_failure``——凭据类自身没有 logger，
        由插件侧（``_fetch_video_info``）输出，避免"填了值却静默失效"。
        """
        now = time.time()
        if self._cached_credential and now - self._cached_at < self.CHECK_INTERVAL_SECONDS:
            return self._cached_credential
        self.last_failure = ""
        for builder in (self._load_from_file, self._load_from_config):
            cred = builder()
            if cred is None:
                continue
            try:
                if await cred.check_valid():
                    self._cached_credential = cred
                    self._cached_at = now
                    self.last_failure = ""
                    return cred
                self.last_failure = (
                    f"{builder.__name__}：未通过有效性校验（SESSDATA 可能已过期/被风控，"
                    "或缺少 bili_jct / buvid3）"
                )
            except Exception as exc:
                self.last_failure = f"{builder.__name__}：校验异常 {exc}"
        if not self.last_failure:
            self.last_failure = "未配置 SESSDATA（且 credential.json 不存在或不含 SESSDATA）"
        self._cached_credential = None
        self._cached_at = now
        return None

    def invalidate_cache(self) -> None:
        """登录态变化后重置缓存。"""
        self._cached_credential = None
        self._cached_at = 0.0




class DouyinCookieManager:
    """抖音 Cookie 与凭据管理：支持配置与本地凭据文件。

    并在缺少 ttwid 时自动请求字节跳动游客注册接口获取匿名 ttwid。
    """

    TTWID_REGISTER_URL = "https://ttwid.bytedance.com/ttwid/union/register/"
    TTWID_REGISTER_BODY = {
        "region": "cn",
        "aid": 1768,
        "needFid": False,
        "service": "www.ixigua.com",
        "migrate_info": {"ticket": "", "source": "node"},
        "cbUrlProtocol": "https",
        "union": True,
    }

    def __init__(self, data_dir: Path, cookie_config: CredentialSectionConfig, client: httpx.AsyncClient):
        self._data_dir = data_dir
        self._cookie_config = cookie_config
        self._client = client
        self._cookie_file = data_dir / "douyin_cookies.txt"
        self._cookie_dict: dict[str, str] = {}
        self._cached_cookie_str = ""
        self.reload()

    def reload(self) -> None:
        """重新加载并更新抖音 Cookie。"""
        merged: dict[str, str] = {}

        # 1. 本地持久化文件加载
        if self._cookie_file.exists():
            try:
                merged.update(_parse_cookie_text(self._cookie_file.read_text(encoding="utf-8")))
            except Exception:
                pass

        # 2. 插件配置覆盖
        if self._cookie_config.douyin_cookie.strip():
            merged.update(_parse_cookie_text(self._cookie_config.douyin_cookie))

        self._cookie_dict = merged
        if self._cookie_dict:
            self._cached_cookie_str = "; ".join(f"{k}={v}" for k, v in self._cookie_dict.items())
            try:
                self._cookie_file.parent.mkdir(parents=True, exist_ok=True)
                self._cookie_file.write_text(self._cached_cookie_str, encoding="utf-8")
            except Exception:
                pass
        else:
            self._cached_cookie_str = ""

    def get_cookie_str(self) -> str:
        """获取当前抖音 Cookie 字符串。"""
        return self._cached_cookie_str

    async def ensure_ttwid(self) -> None:
        """确保持有 ttwid，缺失时向字节注册接口获取一次。"""
        if self._cookie_dict.get("ttwid"):
            return
        try:
            resp = await self._client.post(
                self.TTWID_REGISTER_URL,
                json=self.TTWID_REGISTER_BODY,
                headers={"User-Agent": _IOS_UA, "Content-Type": "application/json"},
            )
            ttwid = resp.cookies.get("ttwid")
            if not ttwid:
                for c_header in resp.headers.get_list("set-cookie"):
                    if "ttwid=" in c_header:
                        m = re.search(r"ttwid=([^;]+)", c_header)
                        if m:
                            ttwid = m.group(1)
                            break
            if ttwid:
                self._cookie_dict["ttwid"] = ttwid
                self._cached_cookie_str = "; ".join(f"{k}={v}" for k, v in self._cookie_dict.items())
                try:
                    self._cookie_file.write_text(self._cached_cookie_str, encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            pass


# ============ 插件主类 ============


class VideoMixin:
    """视频理解能力（mixin 模块）：由主插件类拼装。

    生命周期方法命名为 ``_video_on_*``，由主插件入口统一编排调用（与
    reply-control 的 mixin 拼装方式一致）；配置来自主配置的 parse / credential 段。
    """

    def __init__(self) -> None:
        super().__init__()
        self._client: Optional[httpx.AsyncClient] = None
        self._cred_mgr: Optional[CredentialManager] = None
        self._douyin_cred_mgr: Optional[DouyinCookieManager] = None
        self._video_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def _video_on_load(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        )
        data_dir = Path(self.ctx.paths.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._cred_mgr = CredentialManager(data_dir, self.config.credential, self._client)
        self._douyin_cred_mgr = DouyinCookieManager(data_dir, self.config.credential, self._client)
        self.ctx.logger.info(
            "[视觉增强·视频理解] 已加载 (AI总结=%s, 抖音=%s, 群聊=%s, 私聊=%s)",
            self.config.parse.enable_ai_summary,
            self.config.parse.enable_douyin,
            self.config.parse.enable_in_group,
            self.config.parse.enable_in_private,
        )

    async def _video_on_unload(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self.ctx.logger.info("[视觉增强·视频理解] 已卸载")

    async def _video_on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        self.ctx.logger.info("[视觉增强·视频理解] 配置已更新 (scope=%s, version=%s)", scope, version)
        if self._cred_mgr is not None:
            self._cred_mgr.invalidate_cache()
        if self._douyin_cred_mgr is not None:
            self._douyin_cred_mgr.reload()
        self._video_cache.clear()

    # ------------------------------------------------------------------ #
    # 链接解析与信息获取
    # ------------------------------------------------------------------ #

    async def _resolve_target(self, text: str) -> Optional[tuple[str, Any]]:
        """从文本解析视频目标。

        Returns:
            ("bvid", str) | ("aid", int) | ("douyin", str)；无法识别时返回 None。
        """
        # 1. B站视频识别
        match = _BILI_VIDEO_URL_PATTERN.search(text) or _BV_PATTERN.search(text)
        if match:
            token = match.group(0)
            if "/video/" in token:
                token = token.split("/video/", 1)[1]
            token = token.split("?", 1)[0]
            if token.upper().startswith("BV"):
                return "bvid", token
            if token.lower().startswith("av"):
                return "aid", int(token[2:])
        match = _AV_PATTERN.search(text)
        if match:
            return "aid", int(match.group(1))

        # b23.tv / bili2233.cn 短链：跟随重定向后再提取
        for short in _B23_SHORT_PATTERN.findall(text):
            if not short.startswith("http"):
                short = f"https://{short}"
            try:
                assert self._client is not None
                resp = await self._client.get(short)
                final_url = str(resp.url)
            except Exception:
                continue
            for pattern in (_BV_PATTERN, _AV_PATTERN):
                m = pattern.search(final_url)
                if m:
                    if pattern is _AV_PATTERN:
                        return "aid", int(m.group(1))
                    return "bvid", m.group(1)

        # 2. 抖音视频识别
        # modal_id 参数
        m = _DOUYIN_MODAL_PATTERN.search(text)
        if m:
            return "douyin", m.group(1)

        # 直链 (video/xxx, note/xxx, share/xxx)
        m = _DOUYIN_WEB_PATTERN.search(text)
        if m:
            return "douyin", m.group(1)

        # 短链 (v.douyin.com, jx.douyin.com)
        for short in _DOUYIN_SHORT_PATTERN.findall(text):
            if not short.startswith("http"):
                short = f"https://{short}"
            try:
                assert self._client is not None
                resp = await self._client.get(short, headers={"User-Agent": _IOS_UA})
                final_url = str(resp.url)
            except Exception:
                continue
            for pat in (_DOUYIN_MODAL_PATTERN, _DOUYIN_WEB_PATTERN):
                m = pat.search(final_url)
                if m:
                    return "douyin", m.group(1)

        return None

    # ------------------------------------------------------------------ #
    # B站视频处理
    # ------------------------------------------------------------------ #

    async def _fetch_video_info(self, kind: str, vid: Any) -> Optional[dict[str, Any]]:
        """获取 B 站视频信息 + AI 总结（带缓存）。失败返回 None。"""
        cache_key = f"{kind}:{vid}"
        now = time.time()
        cached = self._video_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

        assert self._cred_mgr is not None
        video = Video(bvid=vid) if kind == "bvid" else Video(aid=vid)
        try:
            info = await video.get_info()
        except Exception as exc:
            self.ctx.logger.warning("[视觉增强·视频理解] 获取B站视频信息失败 (%s=%s): %s", kind, vid, exc)
            return None

        stat = info.get("stat") or {}
        result: dict[str, Any] = {
            "bvid": info.get("bvid", vid if kind == "bvid" else ""),
            "title": info.get("title", ""),
            "up": (info.get("owner") or {}).get("name", "未知UP"),
            "duration": info.get("duration", 0),
            "view": stat.get("view", 0),
            "like": stat.get("like", 0),
            "desc": (info.get("desc") or "")[:200],
            "ai_summary": "",
        }

        if self.config.parse.enable_ai_summary:
            cred = await self._cred_mgr.get_credential()
            if cred is None:
                # 无凭据时此前是静默跳过：日志里什么都没有，容易被误读为"该视频没有总结"
                self._dbg(
                    "[视频理解] 未拿到B站凭据，跳过官方AI总结——该接口需要登录态（原因：%s）；"
                    "请检查「视频凭据」页签：各字段只填值本身（不带键名/引号/分号），且 SESSDATA 未过期",
                    self._cred_mgr.last_failure,
                )
            else:
                video_with_cred = Video(
                    bvid=result["bvid"] if kind == "bvid" else None,
                    aid=int(info.get("aid", 0)) if kind == "aid" else None,
                    credential=cred,
                )
                try:
                    data = await video_with_cred.get_ai_conclusion(cid=info.get("cid"))
                    result["ai_summary"] = _extract_ai_summary(data)
                except Exception as exc:
                    self._dbg("[视频理解] 获取 B 站 AI 总结失败（多为未登录/未填 SESSDATA 或该视频无官方总结）(%s): %s", result["bvid"], exc)

        if len(self._video_cache) > 100:
            self._video_cache.clear()
        self._video_cache[cache_key] = (now + self.config.parse.cache_ttl_seconds, result)
        return result

    def _build_injected_summary(self, info: dict[str, Any]) -> str:
        """构建附加到用户消息末尾的B站视频信息与AI总结块。"""
        title = info.get("title", "").strip()
        up = info.get("up", "未知UP").strip()
        duration = _format_duration(info.get("duration", 0))
        ai_summary = (info.get("ai_summary") or "").strip()

        lines = [f"[B站视频信息: 《{title}》 | UP: {up} | 时长: {duration}]"]
        if ai_summary:
            lines.append(f"[B站官方AI总结]: {ai_summary}")
        elif info.get("desc"):
            lines.append(f"[视频简介]: {info['desc'].strip()}")
        return "\n".join(lines)

    def _build_tool_content(self, info: dict[str, Any]) -> str:
        """构建返回给 planner 的 B 站 Tool 内容。"""
        lines = [
            f"标题: {info['title']}",
            f"UP主: {info['up']}",
            f"时长: {_format_duration(info['duration'])}",
            f"播放: {_format_count(info['view'])}  点赞: {_format_count(info['like'])}",
        ]
        if info.get("desc"):
            lines.append(f"简介: {info['desc']}")
        if info.get("ai_summary"):
            lines += ["", "B站官方AI总结:", info["ai_summary"]]
        else:
            lines += [
                "",
                "(未获取到AI总结：多为未配置B站凭据（该接口需登录态）或该视频无官方总结；可先看标题与简介)",
            ]
        if info.get("bvid"):
            lines.append(f"链接: https://www.bilibili.com/video/{info['bvid']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 抖音视频处理 (章节要点 + 官方AI总结)
    # ------------------------------------------------------------------ #

    async def _fetch_douyin_basic_info(self, vid: str, cookie_str: str) -> Optional[dict[str, Any]]:
        """获取抖音视频基础信息（标题、作者、时长）。"""
        assert self._client is not None
        urls = (
            f"https://www.iesdouyin.com/share/video/{vid}",
            f"https://m.douyin.com/share/video/{vid}",
        )
        headers = {
            "User-Agent": _IOS_UA,
            "Referer": "https://www.douyin.com/",
        }
        if cookie_str:
            headers["Cookie"] = cookie_str

        for url in urls:
            try:
                resp = await self._client.get(url, headers=headers, follow_redirects=True)
                if resp.status_code != 200:
                    continue
                m = re.search(r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", resp.text, re.DOTALL)
                if not m:
                    continue
                data = json.loads(m.group(1).strip())
                loader = data.get("loaderData", {})
                page = loader.get("video_(id)/page") or loader.get("note_(id)/page") or {}
                item_list = page.get("videoInfoRes", {}).get("item_list", [])
                if not item_list:
                    continue
                item = item_list[0]
                dur = (item.get("video") or {}).get("duration", 0)
                duration_sec = (dur // 1000) if dur > 1000 else dur
                return {
                    "title": item.get("desc", ""),
                    "author": (item.get("author") or {}).get("nickname", "未知创作者"),
                    "duration": duration_sec,
                    "desc": item.get("desc", ""),
                }
            except Exception as exc:
                self._dbg("[视频理解] 解析抖音基本信息失败 (%s): %s", url, exc)
                continue
        return None

    async def _fetch_douyin_chapters(self, vid: str, cookie_str: str) -> Optional[dict[str, Any]]:
        """获取抖音视频章节要点（图1路径：从网页 SSR chapterInfo 提取）。"""
        assert self._client is not None
        url = f"https://www.douyin.com/jingxuan?modal_id={vid}"
        headers = {
            "User-Agent": _UA,
            "Cookie": cookie_str,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        try:
            resp = await self._client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            text = resp.text
            pos = text.find('"chapterInfo"')
            if pos == -1:
                return None
            start = text.find("{", pos)
            depth, in_str, escape, end = 0, False, False, -1
            for i in range(start, min(len(text), start + 30000)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
            if end == -1:
                return None
            raw = text[start:end]
            cleaned = raw.replace('\\\\\\"', '"').replace('\\\\"', '"').replace('\\"', '"')
            cdata = json.loads(cleaned)
            ch_list = []
            for ch in cdata.get("list", []):
                ms = ch.get("timestamp", 0)
                ch_list.append({
                    "time": _format_duration(ms // 1000),
                    "desc": ch.get("desc", ""),
                    "detail": ch.get("detail", ""),
                })
            return {
                "chapterAbstract": cdata.get("chapterAbstract", ""),
                "list": ch_list,
            }
        except Exception as exc:
            self._dbg("[视频理解] 提取抖音章节要点失败 (%s): %s", vid, exc)
            return None

    async def _fetch_douyin_pre_generated_summary(
        self, vid: str, cookie_str: str
    ) -> Optional[dict[str, Any]]:
        """获取抖音预生成的官方AI视频总结与高光片段（图8/9路径：即开即用毫秒级响应）。"""
        assert self._client is not None
        if not cookie_str:
            return None

        url = f"https://so-landing.douyin.com/douyin/select/v1/ai/generation/get/?key={vid}&aid=6383"
        headers = {
            "User-Agent": _UA,
            "Cookie": cookie_str,
            "Referer": "https://so-landing.douyin.com/search_ai_mobile/pc",
        }
        try:
            resp = await self._client.get(url, headers=headers, timeout=10.0)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("status_code") != 0:
                return None

            summary = ""
            highlights: list[dict[str, str]] = []
            for card in data.get("business_data", []):
                cid = card.get("card_id")
                cdata = card.get("data", {})
                bytesync = cdata.get("bytesync_data", [])
                if cid == "ai_chat_message_lynx":
                    tokens = []
                    for bs in bytesync:
                        bso = json.loads(bs) if isinstance(bs, str) else bs
                        disp = bso.get("display", {}) if isinstance(bso, dict) else {}
                        for span in disp.get("generation_spans", []):
                            content = span.get("text", {}).get("content")
                            if content:
                                tokens.append(content)
                    summary = _clean_html_tags("".join(tokens))
                elif cid == "ask_ai_high_light_clip":
                    for bs in bytesync:
                        bso = json.loads(bs) if isinstance(bs, str) else bs
                        if isinstance(bso, dict):
                            for item in bso.get("data", []):
                                st = item.get("startTime", 0)
                                title = item.get("title", "").strip()
                                if title:
                                    highlights.append({
                                        "time": _format_duration(st),
                                        "title": title,
                                    })
            if summary or highlights:
                return {"summary": summary, "highlights": highlights}
        except Exception as exc:
            self._dbg("[视频理解] 获取抖音预生成AI总结失败（未配置抖音 Cookie / Cookie 失效 / 该视频无预生成总结）(%s): %s", vid, exc)
        return None

    async def _fetch_douyin_ai_summary(
        self, vid: str, cookie_str: str, title: str = ""
    ) -> str:
        """获取抖音官方实时AI视频总结（图2/3路径：从 AI 搜索流式接口提取）。"""
        assert self._client is not None
        if not cookie_str:
            return ""

        stream_url = "https://so-landing.douyin.com/douyin/select/v1/ai/stream/"
        device_id = str(random.randint(7000000000000000000, 7999999999999999999))
        keyword = f"总结当前视频内容：{title.strip()}" if title.strip() else "视频总结"
        params = {
            "count": "5",
            "cursor": "0",
            "token": "search",
            "ai_page_type": "ai_chat",
            "search_channel": "aweme_ai_chat",
            "enable_ai_tab_new_framework": "1",
            "need_integration_card": "1",
            "ai_chat_message_use_lynx": "1",
            "version_code": "32.1.0",
            "enter_method": "offline_summary_card",
            "enter_from": "general_search",
            "search_type": "ai_chat_search",
            "aid": "6383",
            "device_id": device_id,
            "keyword": keyword,
            "ai_search_enter_from_group_id": vid,
            "aweme_id": vid,
        }
        headers = {
            "User-Agent": _UA,
            "Cookie": cookie_str,
            "Referer": "https://so-landing.douyin.com/search_ai_mobile/pc",
            "Origin": "https://so-landing.douyin.com",
            "Accept": "text/event-stream",
        }
        try:
            async with self._client.stream("GET", stream_url, params=params, headers=headers, timeout=25.0) as resp:
                if resp.status_code != 200:
                    return ""
                tokens: list[str] = []
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        d = json.loads(line[5:].strip())
                        for item in d.get("data", []):
                            display = item.get("display", {})
                            disp_inner = display.get("display", {}) if isinstance(display, dict) else {}
                            for span in disp_inner.get("generation_spans", []):
                                text_obj = span.get("text", {})
                                if isinstance(text_obj, dict) and "content" in text_obj:
                                    tokens.append(text_obj["content"])
                    except Exception:
                        pass
                text = _clean_html_tags("".join(tokens))
                # 过滤拒答模板，避免将“无法提供总结/信息过于模糊”等废话注入聊天上下文
                refusal_phrases = (
                    "无法为您提供",
                    "无法对您提到",
                    "无法确定",
                    "无法总结",
                    "无法进行总结",
                    "无法定位",
                    "根据现有信息，无法",
                    "根据现有信息，目前无法",
                    "信息不足",
                    "过于模糊",
                    "缺乏可供识别",
                    "缺乏可识别",
                    "无法唯一确定",
                    "建议直接观看",
                    "建议补充视频",
                )
                if any(phrase in text[:150] for phrase in refusal_phrases):
                    return ""
                return text
        except Exception as exc:
            self._dbg("[视频理解] 获取抖音实时AI总结失败（多为未配置/失效 Cookie 或接口拒答）(%s): %s", vid, exc)
            return ""

    async def _fetch_douyin_video_info(self, vid: str) -> Optional[dict[str, Any]]:
        """获取抖音视频信息、高光片段、章节要点与AI总结（带缓存）。"""
        cache_key = f"douyin:{vid}"
        now = time.time()
        cached = self._video_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

        assert self._douyin_cred_mgr is not None
        await self._douyin_cred_mgr.ensure_ttwid()
        cookie_str = self._douyin_cred_mgr.get_cookie_str()

        # 1. 基础信息
        basic_info = await self._fetch_douyin_basic_info(vid, cookie_str)
        if not basic_info:
            return None

        result: dict[str, Any] = {
            "vid": vid,
            "title": basic_info.get("title", ""),
            "author": basic_info.get("author", "未知创作者"),
            "duration": basic_info.get("duration", 0),
            "desc": basic_info.get("desc", "")[:200],
            "chapters": [],
            "chapter_abstract": "",
            "highlights": [],
            "ai_summary": "",
        }

        # 2. 预生成总结/高光片段、SSR章节要点与实时流式AI总结
        if self.config.parse.enable_ai_summary:
            # 优先检查官方预生成的 AI 总结与高光片段（毫秒级极速返回，图8/9路径）
            pre_gen = await self._fetch_douyin_pre_generated_summary(vid, cookie_str)
            if pre_gen:
                result["ai_summary"] = pre_gen.get("summary", "")
                result["highlights"] = pre_gen.get("highlights", [])

            # 并发获取 SSR 章节要点（图1路径）
            chapters_task = asyncio.create_task(self._fetch_douyin_chapters(vid, cookie_str))

            # 若未预生成总结，降级调用流式 AI 总结（图2/3路径）
            stream_task = None
            if not result["ai_summary"]:
                stream_task = asyncio.create_task(
                    self._fetch_douyin_ai_summary(vid, cookie_str, title=result["title"])
                )

            try:
                if stream_task:
                    ch_data, summary = await asyncio.gather(chapters_task, stream_task, return_exceptions=True)
                    if isinstance(summary, str) and summary:
                        result["ai_summary"] = summary
                else:
                    ch_data = await chapters_task

                if isinstance(ch_data, dict):
                    result["chapters"] = ch_data.get("list", [])
                    result["chapter_abstract"] = ch_data.get("chapterAbstract", "")
            except Exception as exc:
                self._dbg("[视频理解] 获取抖音AI总结/章节异常 (%s): %s", vid, exc)

        if len(self._video_cache) > 100:
            self._video_cache.clear()
        self._video_cache[cache_key] = (now + self.config.parse.cache_ttl_seconds, result)
        return result

    def _build_injected_douyin_summary(self, info: dict[str, Any]) -> str:
        """构建附加到用户消息末尾的抖音视频信息、高光片段、章节要点与AI总结块。"""
        title = (info.get("title") or "").strip()
        author = (info.get("author") or "未知创作者").strip()
        duration = _format_duration(info.get("duration", 0))
        highlights = info.get("highlights") or []
        chapters = info.get("chapters") or []
        chapter_abstract = (info.get("chapter_abstract") or "").strip()
        ai_summary = (info.get("ai_summary") or "").strip()

        lines = [f"[抖音视频信息: 《{title}》 | 作者: @{author} | 时长: {duration}]"]
        if highlights:
            lines.append("[高光片段]:")
            for h in highlights:
                lines.append(f"- {h['time']} {h['title']}")

        if chapters:
            lines.append("[抖音章节要点]:")
            if chapter_abstract:
                lines.append(chapter_abstract)
            for ch in chapters:
                time_str = ch.get("time", "")
                desc = ch.get("desc", "").strip()
                detail = ch.get("detail", "").strip()
                if detail:
                    lines.append(f"- {time_str} {desc}: {detail}")
                else:
                    lines.append(f"- {time_str} {desc}")

        if ai_summary:
            lines.append(f"[抖音官方AI总结]:\n{ai_summary}")
        elif not chapters and not highlights and info.get("desc"):
            lines.append(f"[视频简介]: {info['desc'].strip()}")

        return "\n".join(lines)

    def _build_douyin_tool_content(self, info: dict[str, Any]) -> str:
        """构建返回给 planner 的抖音 Tool 内容。"""
        lines = [
            f"标题: {info.get('title', '')}",
            f"作者: @{info.get('author', '未知创作者')}",
            f"时长: {_format_duration(info.get('duration', 0))}",
        ]
        if info.get("desc") and info["desc"] != info.get("title"):
            lines.append(f"简介: {info['desc']}")
        if not (info.get("ai_summary") or "").strip():
            lines.append(
                "AI总结: 未获取到（可能未配置抖音 Cookie、Cookie 失效，或该视频无官方总结）"
            )
        highlights = info.get("highlights") or []
        if highlights:
            lines += ["", "高光片段:"]
            for h in highlights:
                lines.append(f"- {h['time']} {h['title']}")
        chapters = info.get("chapters") or []
        chapter_abstract = (info.get("chapter_abstract") or "").strip()
        if chapters:
            lines += ["", "抖音章节要点:"]
            if chapter_abstract:
                lines.append(chapter_abstract)
            for ch in chapters:
                t = ch.get("time", "")
                d = ch.get("desc", "").strip()
                det = ch.get("detail", "").strip()
                lines.append(f"- {t} {d}: {det}" if det else f"- {t} {d}")
        if info.get("ai_summary"):
            lines += ["", "抖音官方AI总结:", info["ai_summary"]]
        elif not chapters and not highlights:
            lines += [
            "",
            "(未获取到AI总结/章节：多为未配置抖音 Cookie（该接口需登录态）或该视频无官方总结；可先看标题与简介)",
        ]
        if info.get("vid"):
            lines.append(f"链接: https://www.douyin.com/video/{info['vid']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Hook: 拦截消息并在聊天上下文中附加AI总结（不自动发消息回复）
    # ------------------------------------------------------------------ #

    @HookHandler(
        "chat.receive.after_process",
        name="bilibili_summary_injector",
        description="检测入站消息中的B站/抖音视频链接并将AI总结直接附加到消息内容中",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def inject_bilibili_summary(self, **kwargs: Any) -> dict[str, Any]:
        message = kwargs.get("message")
        if not isinstance(message, dict):
            return {"action": "continue"}
        if message.get("is_notify"):
            return {"action": "continue"}

        cfg = self.config.parse
        message_info = message.get("message_info") or {}
        is_group = bool(message_info.get("group_info"))
        if is_group and not cfg.enable_in_group:
            return {"action": "continue"}
        if not is_group and not cfg.enable_in_private:
            return {"action": "continue"}

        text = str(message.get("processed_plain_text") or "")
        if not text:
            raw_parts = []
            for seg in message.get("raw_message") or []:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    data = seg.get("data")
                    if isinstance(data, str) and data:
                        raw_parts.append(data)
            text = " ".join(raw_parts)

        if not text or "[B站视频" in text or "[抖音视频" in text:
            return {"action": "continue"}

        target = await self._resolve_target(text)
        if target is None:
            return {"action": "continue"}

        summary_block = ""
        platform_kind, arg1 = target[0], target[1]
        if platform_kind in ("bvid", "aid"):
            try:
                info = await self._fetch_video_info(platform_kind, arg1)
                if info:
                    summary_block = self._build_injected_summary(info)
            except Exception as exc:  # noqa: BLE001
                self._dbg("[视频理解] 获取B站视频总结异常: %s", exc)
                return {"action": "continue"}
        elif platform_kind == "douyin":
            if not cfg.enable_douyin:
                return {"action": "continue"}
            try:
                info = await self._fetch_douyin_video_info(str(arg1))
                if info:
                    summary_block = self._build_injected_douyin_summary(info)
            except Exception as exc:  # noqa: BLE001
                self._dbg("[视频理解] 获取抖音视频总结异常: %s", exc)
                return {"action": "continue"}

        if not summary_block:
            return {"action": "continue"}

        got_summary = bool(str((info or {}).get("ai_summary") or "").strip())
        self.ctx.logger.info(
            "[视觉增强·视频理解] 为消息 %s 附加%s视频信息（%s）(%s=%s)",
            message.get("message_id"),
            "B站" if platform_kind != "douyin" else "抖音",
            "含官方AI总结" if got_summary else "无AI总结，仅视频信息与简介",
            platform_kind,
            arg1,
        )

        current_plain = str(message.get("processed_plain_text") or "").strip()
        message["processed_plain_text"] = f"{current_plain}\n{summary_block}".strip()

        raw_msg = message.get("raw_message")
        if isinstance(raw_msg, list):
            raw_msg.append({"type": "text", "data": f"\n{summary_block}"})

        return {
            "action": "continue",
            "modified_kwargs": {
                "message": message,
            },
        }

    # ------------------------------------------------------------------ #
    # Tool: 供 planner 显式调用
    # ------------------------------------------------------------------ #

    @Tool(
        "parse_bilibili_video",
        description=(
            "解析B站视频并获取B站官方AI视频总结。当聊天中出现B站视频链接、BV号、"
            "b23.tv短链，或有人提到想看/讨论某个B站视频时，调用此工具获取视频标题、"
            "UP主、时长、播放数据和AI总结，帮助你理解视频内容并参与讨论。"
        ),
        parameters=[
            ToolParameterInfo(
                name="link",
                param_type=ToolParamType.STRING,
                description="B站视频链接、BV号（如 BV1xx411c7mD）、av号或 b23.tv 短链",
                required=True,
            ),
        ],
    )
    async def tool_parse_bilibili_video(self, link: str = "", **kwargs: Any) -> dict[str, str]:
        del kwargs
        link = (link or "").strip()
        if not link:
            return {"name": "parse_bilibili_video", "content": "参数 link 为空，无法解析。"}
        target = await self._resolve_target(link)
        if target is None or target[0] not in ("bvid", "aid"):
            return {"name": "parse_bilibili_video", "content": f"无法从「{link}」中识别出B站视频。"}
        try:
            info = await self._fetch_video_info(target[0], target[1])
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("[视觉增强·视频理解] 解析视频失败 (%s): %s", link, exc, exc_info=True)
            return {"name": "parse_bilibili_video", "content": f"解析B站视频失败: {exc}"}
        if info is None:
            return {"name": "parse_bilibili_video", "content": "获取视频信息失败，视频可能不存在或网络异常。"}
        return {"name": "parse_bilibili_video", "content": self._build_tool_content(info)}

    @Tool(
        "parse_douyin_video",
        description=(
            "解析抖音视频并获取抖音官方AI视频总结、高光片段与章节要点。当聊天中出现抖音视频链接、"
            "v.douyin.com 短链、分享口令文本，或有人提到想了解某个抖音视频内容时，调用此工具获取视频标题、"
            "作者、时长、高光片段、章节要点及官方AI总结，帮助你理解视频内容并参与讨论。"
        ),
        parameters=[
            ToolParameterInfo(
                name="link",
                param_type=ToolParamType.STRING,
                description="抖音视频分享链接、短链（如 https://v.douyin.com/xxxx/）、分享文本口令或19位视频ID",
                required=True,
            ),
        ],
    )
    async def tool_parse_douyin_video(self, link: str = "", **kwargs: Any) -> dict[str, str]:
        del kwargs
        link = (link or "").strip()
        if not link:
            return {"name": "parse_douyin_video", "content": "参数 link 为空，无法解析。"}
        target = await self._resolve_target(link)
        if target is None or target[0] != "douyin":
            # 兼容直接传入19位数字ID
            m = re.search(r"\b(7\d{18})\b", link)
            if m:
                target = ("douyin", m.group(1))
            else:
                return {"name": "parse_douyin_video", "content": f"无法从「{link}」中识别出抖音视频。"}
        try:
            info = await self._fetch_douyin_video_info(str(target[1]))
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("[视觉增强·视频理解] 解析抖音视频失败 (%s): %s", link, exc, exc_info=True)
            return {"name": "parse_douyin_video", "content": f"解析抖音视频失败: {exc}"}
        if info is None:
            return {"name": "parse_douyin_video", "content": "获取抖音视频信息失败，视频可能不存在或网络异常。"}
        return {"name": "parse_douyin_video", "content": self._build_douyin_tool_content(info)}

