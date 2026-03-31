#!/usr/bin/env python3
"""
SuperTechFans Daily Podcast Generator

Pipeline:
  1. Fetch daily HackerNews summaries from supertechfans.com
  2. Generate podcast script via LLM (OpenRouter or OpenClaw agent)
  3. Convert to audio via Microsoft Edge TTS
  4. Upload MP3 to Cloudflare R2
  5. Update podcast RSS feed (podcast.xml)
  6. Push RSS to GitHub Pages
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
from pathlib import Path
import edge_tts
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

(BASE_DIR / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "logs" / "podcast.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("podcast")

RSS_URL = "https://supertechfans.com/cn/index.xml"
ARTICLE_URL_TEMPLATE = "https://supertechfans.com/cn/post/{date}-HackerNews/"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENCLAW_BIN = "/opt/homebrew/bin/openclaw"

DEFAULTS = {
    "TTS_VOICE": "zh-CN-YunxiNeural",
    "TTS_RATE": "-5%",
    "LLM_MODEL": "google/gemini-2.5-flash-preview",
    "R2_BUCKET": "hackernews-podcast",
    "R2_PUBLIC_URL": "",
    "R2_ACCOUNT_ID": "",
    "R2_ACCESS_KEY_ID": "",
    "R2_SECRET_ACCESS_KEY": "",
    "GITHUB_PAGES_REPO": "",
    "PODCAST_TITLE": "超级科技迷·八分日报",
    "PODCAST_DESCRIPTION": "每天8分钟，用梁文道《八分》的风格聊聊 HackerNews 上最有意思的科技文化话题。AI 生成，人文视角。",
    "PODCAST_AUTHOR": "SuperTechFans",
    "PODCAST_LANGUAGE": "zh-cn",
    "PODCAST_IMAGE_URL": "https://supertechfans.com/favicon.png",
    "PODCAST_WEBSITE": "https://supertechfans.com/cn/",
}


def get_config(key: str) -> str:
    return os.getenv(key, DEFAULTS.get(key, ""))


def get_api_key() -> str | None:
    """Resolve the OpenRouter API key. Returns None if not available."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            profiles = cfg.get("auth", {}).get("profiles", {})
            for prof in profiles.values():
                if prof.get("provider") == "openrouter" and prof.get("apiKey"):
                    return prof["apiKey"]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Step 1: Fetch news content
# ---------------------------------------------------------------------------

def fetch_news_from_rss(target_date: str) -> str | None:
    """Try RSS feed first. Returns article HTML or None."""
    log.info("Fetching RSS feed: %s", RSS_URL)
    try:
        resp = requests.get(RSS_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("RSS fetch failed: %s", e)
        return None

    root = ET.fromstring(resp.content)
    for item in root.iter("item"):
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        if target_date in title or target_date in link:
            desc = item.findtext("description", "")
            if desc:
                log.info("Found article via RSS: %s", title)
                return desc
    return None


def fetch_news_from_web(target_date: str) -> str | None:
    """Fallback: fetch article page directly."""
    url = ARTICLE_URL_TEMPLATE.format(date=target_date)
    log.info("Fetching article page: %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        if status_code == 404:
            log.info("Article for %s not published yet (404), skipping this run", target_date)
            return None
        log.warning("Article page fetch failed: %s", e)
        return None
    except requests.RequestException as e:
        log.warning("Article page fetch failed: %s", e)
        return None
    return resp.text


def html_to_text(html: str) -> str:
    """Convert HTML to clean text for LLM consumption."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    article = soup.find("article") or soup.find(class_="book-page") or soup
    text = article.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_today_news(date_str: str | None = None) -> str | None:
    """Fetch and return today's news as plain text, or None if not ready."""
    tz_cn = timezone(timedelta(hours=8))
    if date_str:
        target = date_str
    else:
        target = datetime.now(tz_cn).strftime("%Y-%m-%d")

    html = fetch_news_from_rss(target)
    if not html:
        html = fetch_news_from_web(target)
    if not html:
        log.info("No publish-ready news found for %s, skipping this run", target)
        return None

    text = html_to_text(html)
    if len(text) < 200:
        log.info("Fetched content too short (%d chars), skipping this run", len(text))
        return None

    log.info("Fetched %d chars of news content", len(text))
    return text


# ---------------------------------------------------------------------------
# Step 2: Generate podcast script via LLM
# ---------------------------------------------------------------------------

def _build_prompt(news_text: str) -> str:
    prompt_path = BASE_DIR / "prompt_template.txt"
    template = prompt_path.read_text(encoding="utf-8")
    return template.replace("{news_content}", news_text)


def _generate_via_openrouter(prompt: str, api_key: str) -> str:
    """Direct OpenRouter API call (efficient, recommended)."""
    model = get_config("LLM_MODEL")
    log.info("Calling OpenRouter (%s) directly...", model)

    resp = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://supertechfans.com",
            "X-Title": "SuperTechFans Podcast",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.8,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _generate_via_openclaw(prompt: str) -> str:
    """Fallback: use openclaw agent CLI (uses existing OpenClaw auth)."""
    log.info("Calling OpenClaw agent (no API key in .env, using openclaw auth)...")

    result = subprocess.run(
        [OPENCLAW_BIN, "agent", "--agent", "main", "--json",
         "--message", prompt, "--timeout", "300"],
        capture_output=True, text=True, timeout=360,
    )
    if result.returncode != 0:
        log.error("OpenClaw agent failed: %s", result.stderr[-500:])
        sys.exit(1)

    lines = result.stdout.strip().split("\n")
    json_start = next(i for i, l in enumerate(lines) if l.strip().startswith("{"))
    raw_json = "\n".join(lines[json_start:])
    data = json.loads(raw_json)

    if data.get("status") != "ok":
        log.error("OpenClaw agent error: %s", data.get("summary", "unknown"))
        sys.exit(1)

    return data["result"]["payloads"][0]["text"].strip()


def generate_script(news_text: str) -> str:
    """Generate podcast script via LLM."""
    prompt = _build_prompt(news_text)
    api_key = get_api_key()

    if api_key:
        script = _generate_via_openrouter(prompt, api_key)
    elif Path(OPENCLAW_BIN).exists():
        script = _generate_via_openclaw(prompt)
    else:
        log.error(
            "No OPENROUTER_API_KEY in .env and openclaw CLI not found. "
            "Set OPENROUTER_API_KEY in .env or install openclaw."
        )
        sys.exit(1)

    log.info("Generated podcast script: %d chars", len(script))
    return script


# ---------------------------------------------------------------------------
# Step 3: Convert to audio via Edge TTS
# ---------------------------------------------------------------------------

async def text_to_speech(text: str, output_path: Path) -> None:
    """Convert text to MP3 using Microsoft Edge TTS."""
    voice = get_config("TTS_VOICE")
    rate = get_config("TTS_RATE")

    log.info("Generating audio with voice=%s, rate=%s", voice, rate)
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Audio saved: %s (%.1f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# Step 4: Upload MP3 to Cloudflare R2
# ---------------------------------------------------------------------------

def upload_to_r2(local_path: Path, r2_key: str) -> str | None:
    """Upload file to Cloudflare R2. Returns public URL or None if R2 not configured."""
    account_id = get_config("R2_ACCOUNT_ID")
    access_key = get_config("R2_ACCESS_KEY_ID")
    secret_key = get_config("R2_SECRET_ACCESS_KEY")
    bucket = get_config("R2_BUCKET")
    public_url = get_config("R2_PUBLIC_URL")

    if not all([account_id, access_key, secret_key]):
        log.warning("R2 not configured, skipping upload")
        return None

    import boto3
    from botocore.config import Config

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    log.info("Uploading %s to R2 bucket=%s key=%s", local_path.name, bucket, r2_key)
    s3.upload_file(
        str(local_path), bucket, r2_key,
        ExtraArgs={"ContentType": "audio/mpeg"},
    )

    if public_url:
        url = f"{public_url.rstrip('/')}/{r2_key}"
    else:
        url = f"{endpoint}/{bucket}/{r2_key}"

    log.info("Uploaded to R2: %s", url)
    return url


# ---------------------------------------------------------------------------
# Step 5: Update podcast RSS feed
# ---------------------------------------------------------------------------

def get_mp3_duration_estimate(file_path: Path) -> int:
    """Estimate MP3 duration in seconds from file size (48kbps mono)."""
    size_bytes = file_path.stat().st_size
    return int(size_bytes / (48_000 / 8))


def episode_exists_in_rss(rss_dir: Path, episode_date: str) -> bool:
    """Check whether today's episode is already present in podcast.xml."""
    rss_path = rss_dir / "podcast.xml"
    if not rss_path.exists():
        return False

    try:
        tree = ET.parse(rss_path)
    except ET.ParseError as e:
        log.warning("Failed to parse RSS (%s), continue processing: %s", rss_path, e)
        return False

    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        return False

    episode_guid = f"supertechfans-{episode_date}"
    for item in channel.findall("item"):
        if item.findtext("guid", "") == episode_guid:
            return True
    return False


def update_podcast_rss(
    episode_date: str,
    episode_title: str,
    audio_url: str,
    mp3_path: Path,
    script_text: str,
    rss_dir: Path,
) -> Path:
    """Create or update podcast.xml with a new episode."""
    rss_path = rss_dir / "podcast.xml"

    feed_url = ""
    repo = get_config("GITHUB_PAGES_REPO")
    if repo:
        feed_url = f"https://{repo.split('/')[0]}.github.io/{repo.split('/')[-1]}/podcast.xml"

    ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    ATOM_NS = "http://www.w3.org/2005/Atom"
    ET.register_namespace("itunes", ITUNES_NS)
    ET.register_namespace("atom", ATOM_NS)

    if rss_path.exists():
        tree = ET.parse(rss_path)
        root = tree.getroot()
        channel = root.find("channel")
    else:
        root = ET.Element("rss", attrib={"version": "2.0"})
        channel = ET.SubElement(root, "channel")
        ET.SubElement(channel, "title").text = get_config("PODCAST_TITLE")
        ET.SubElement(channel, "link").text = get_config("PODCAST_WEBSITE")
        ET.SubElement(channel, "language").text = get_config("PODCAST_LANGUAGE")
        ET.SubElement(channel, "description").text = get_config("PODCAST_DESCRIPTION")
        if feed_url:
            ET.SubElement(channel, f"{{{ATOM_NS}}}link", attrib={
                "href": feed_url, "rel": "self", "type": "application/rss+xml",
            })
        ET.SubElement(channel, "lastBuildDate")
        ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = get_config("PODCAST_AUTHOR")
        ET.SubElement(channel, f"{{{ITUNES_NS}}}summary").text = get_config("PODCAST_DESCRIPTION")
        owner = ET.SubElement(channel, f"{{{ITUNES_NS}}}owner")
        ET.SubElement(owner, f"{{{ITUNES_NS}}}name").text = get_config("PODCAST_AUTHOR")
        ET.SubElement(owner, f"{{{ITUNES_NS}}}email").text = "hello@supertechfans.com"
        ET.SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "false"
        ET.SubElement(channel, f"{{{ITUNES_NS}}}type").text = "episodic"
        ET.SubElement(channel, f"{{{ITUNES_NS}}}category", attrib={"text": "Technology"})
        img = ET.SubElement(channel, f"{{{ITUNES_NS}}}image")
        img.set("href", get_config("PODCAST_IMAGE_URL"))

        tree = ET.ElementTree(root)

    existing_guids = {
        item.findtext("guid", "")
        for item in channel.findall("item")
    }
    episode_guid = f"supertechfans-{episode_date}"
    if episode_guid in existing_guids:
        log.info("Episode %s already in RSS, skipping", episode_date)
        return rss_path

    tz_cn = timezone(timedelta(hours=8))
    pub_date = datetime.strptime(episode_date, "%Y-%m-%d").replace(
        hour=9, minute=0, tzinfo=tz_cn,
    )

    file_size = mp3_path.stat().st_size
    duration_secs = get_mp3_duration_estimate(mp3_path)
    duration_str = f"{duration_secs // 60}:{duration_secs % 60:02d}"
    summary = script_text[:300].replace("\n", " ") + "..."

    item = ET.Element("item")
    ET.SubElement(item, "title").text = episode_title
    ET.SubElement(item, "link").text = audio_url
    desc = ET.SubElement(item, "description")
    desc.text = f'<p>{summary}</p><p><a href="{audio_url}">收听音频 (MP3)</a></p>'
    ET.SubElement(item, "pubDate").text = format_datetime(pub_date)
    ET.SubElement(item, "guid", attrib={"isPermaLink": "false"}).text = episode_guid
    ET.SubElement(item, "enclosure", attrib={
        "url": audio_url,
        "length": str(file_size),
        "type": "audio/mpeg",
    })
    ET.SubElement(item, f"{{{ITUNES_NS}}}duration").text = duration_str
    ET.SubElement(item, f"{{{ITUNES_NS}}}summary").text = summary
    ET.SubElement(item, f"{{{ITUNES_NS}}}episodeType").text = "full"
    ET.SubElement(item, f"{{{ITUNES_NS}}}explicit").text = "false"

    channel.append(item)

    lbd = channel.find("lastBuildDate")
    if lbd is None:
        lbd = ET.SubElement(channel, "lastBuildDate")
        items = channel.findall("item")
        if items:
            channel.remove(lbd)
            channel.insert(list(channel).index(items[0]), lbd)
    lbd.text = format_datetime(datetime.now(tz_cn))

    ET.indent(tree, space="  ")
    tree.write(rss_path, encoding="unicode", xml_declaration=True)

    log.info("RSS updated: %s (guid=%s)", rss_path, episode_guid)
    return rss_path


# ---------------------------------------------------------------------------
# Step 6: Push RSS to GitHub Pages
# ---------------------------------------------------------------------------

def push_to_github(rss_dir: Path, episode_date: str) -> None:
    """Git commit and push the updated RSS to GitHub."""
    repo = get_config("GITHUB_PAGES_REPO")
    if not repo:
        log.warning("GITHUB_PAGES_REPO not configured, skipping push")
        return

    status = subprocess.run(
        ["git", "status", "--porcelain", "podcast.xml"],
        cwd=rss_dir, capture_output=True, text=True,
    )
    if not status.stdout.strip():
        log.info("podcast.xml has no changes, skipping git commit/push")
        return

    try:
        # In CI environments there may be no git identity configured.
        if os.getenv("GITHUB_ACTIONS") == "true":
            subprocess.run(
                ["git", "config", "user.name", os.getenv("GIT_USER_NAME", "github-actions[bot]")],
                cwd=rss_dir, check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", os.getenv("GIT_USER_EMAIL", "github-actions[bot]@users.noreply.github.com")],
                cwd=rss_dir, check=True, capture_output=True,
            )

        subprocess.run(
            ["git", "add", "podcast.xml"],
            cwd=rss_dir, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add episode {episode_date}"],
            cwd=rss_dir, check=True, capture_output=True,
        )

        gh_token = os.getenv("GH_TOKEN", "").strip()
        if gh_token:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=rss_dir, check=True, capture_output=True, text=True,
            ).stdout.strip() or "main"
            push_url = f"https://x-access-token:{gh_token}@github.com/{repo}.git"
            subprocess.run(
                ["git", "push", push_url, f"HEAD:{branch}"],
                cwd=rss_dir, check=True, capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "push"],
                cwd=rss_dir, check=True, capture_output=True,
            )
        log.info("Pushed RSS update to GitHub")
    except subprocess.CalledProcessError as e:
        if isinstance(e.stderr, bytes):
            err_msg = e.stderr.decode(errors="ignore")
        else:
            err_msg = e.stderr or str(e)
        log.error("Git push failed: %s", err_msg[-300:])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else None

    tz_cn = timezone(timedelta(hours=8))
    today = date_str or datetime.now(tz_cn).strftime("%Y-%m-%d")
    rss_dir = BASE_DIR / "rss"
    rss_dir.mkdir(exist_ok=True)

    podcasts_dir = BASE_DIR / "podcasts"
    podcasts_dir.mkdir(exist_ok=True)

    output_mp3 = podcasts_dir / f"{today}.mp3"
    output_script = podcasts_dir / f"{today}.txt"

    if episode_exists_in_rss(rss_dir, today):
        log.info("Episode %s already exists in RSS, skipping", today)
        print(f"Already processed: {today}")
        return

    if output_mp3.exists():
        log.info("Podcast already exists: %s", output_mp3)
        print(f"Already generated: {output_mp3}")
        return

    log.info("=== Generating podcast for %s ===", today)

    # 1. Fetch news
    news_text = get_today_news(date_str)
    if not news_text:
        print(f"No publish-ready article for {today}, skipped.")
        return

    # 2. Generate script
    script = generate_script(news_text)
    output_script.write_text(script, encoding="utf-8")
    log.info("Script saved: %s", output_script)

    # 3. Text to speech
    asyncio.run(text_to_speech(script, output_mp3))

    # 4. Upload to R2
    r2_key = f"episodes/{today}.mp3"
    audio_url = upload_to_r2(output_mp3, r2_key)

    # 5. Update RSS (if R2 upload succeeded)
    if audio_url:
        episode_title = f"HackerNews 八分日报 {today}"
        update_podcast_rss(
            episode_date=today,
            episode_title=episode_title,
            audio_url=audio_url,
            mp3_path=output_mp3,
            script_text=script,
            rss_dir=rss_dir,
        )

        # 6. Push to GitHub Pages
        push_to_github(rss_dir, today)
    else:
        log.info("R2 not configured — podcast saved locally only: %s", output_mp3)

    print(f"Podcast generated: {output_mp3}")
    if audio_url:
        print(f"Audio URL: {audio_url}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
