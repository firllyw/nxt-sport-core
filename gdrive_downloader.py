import os
import re
import requests
import gdown
import yt_dlp
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

def extract_gdrive_id(url_or_id):
    """
    Extract Google Drive file ID from URL or return raw ID
    """
    if not url_or_id:
        return None
        
    url = url_or_id.strip()
    
    # Check if already a clean ID
    if re.match(r'^[a-zA-Z0-9_-]{20,50}$', url) and not url.startswith("http"):
        return url
        
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/open\?id=([a-zA-Z0-9_-]+)',
        r'/uc\?id=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
            
    return None

def download_direct_stream(url, output_path, progress_callback=None):
    """
    Download direct video file (e.g. .mp4, .mov) via streaming requests
    """
    resp = requests.get(url, stream=True, timeout=20)
    resp.raise_for_status()
    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and progress_callback:
                    pct = min(98, int((downloaded / total_size) * 90) + 5)
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    progress_callback(pct, f"Mengunduh video langsung: {mb_down:.1f}/{mb_total:.1f} MB ({pct}%)")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return output_path
    return None

def download_gdrive_video(url_or_id, output_path, progress_callback=None):
    """
    Download video supporting:
    1. Google Drive public links (via gdown / yt-dlp)
    2. YouTube links (via yt-dlp)
    3. Direct Video URLs (.mp4, .mov, etc.)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_input = url_or_id.strip()
    file_id = extract_gdrive_id(raw_input)

    if progress_callback:
        progress_callback(5, "Memulai unduhan sumber video...")

    # Scenario A: Direct Video URL (ends in mp4/mov/mkv or direct video link)
    if (raw_input.startswith("http://") or raw_input.startswith("https://")) and not file_id and not ("youtube.com" in raw_input or "youtu.be" in raw_input):
        try:
            res = download_direct_stream(raw_input, output_path, progress_callback=progress_callback)
            if res:
                if progress_callback:
                    progress_callback(100, "Download direct video selesai!")
                return res
        except Exception as e:
            print(f"[downloader] Direct download failed, fallback to yt-dlp: {e}")

    # Scenario B: YouTube or generic video platforms via yt-dlp
    is_youtube = "youtube.com" in raw_input or "youtu.be" in raw_input
    
    target_url = f"https://drive.google.com/uc?id={file_id}&export=download" if file_id else raw_input

    try:
        def yt_dlp_hook(d):
            if d.get('status') == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded = d.get('downloaded_bytes') or 0
                if total > 0:
                    pct = min(95, int((downloaded / total) * 90) + 5)
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    if progress_callback:
                        progress_callback(pct, f"Mengunduh video: {mb_down:.1f} MB / {mb_total:.1f} MB ({pct}%)")

        ydl_opts = {
            'ffmpeg_location': FFMPEG_EXE,
            'outtmpl': output_path,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'progress_hooks': [yt_dlp_hook],
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,
        }

        # Check for cookies.txt if exists
        cookies_file = os.path.join(os.path.dirname(__file__), "cookies.txt")
        if os.path.exists(cookies_file):
            ydl_opts['cookiefile'] = cookies_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([target_url])
            
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            if progress_callback:
                progress_callback(100, "Download selesai via yt-dlp.")
            return output_path
    except Exception as e:
        err_msg = str(e)
        print(f"[downloader] yt-dlp download failed: {err_msg}")
        if is_youtube and "confirm you’re not a bot" in err_msg:
            raise RuntimeError("YouTube memblokir download dari IP server (bot challenge). Untuk YouTube di server, disarankan menggunakan file cookies.txt atau gunakan link Google Drive publik.")

    # Scenario C: Fallback to gdown for Google Drive
    if file_id:
        try:
            if progress_callback:
                progress_callback(10, "Mengunduh via Google Drive fallback (gdown)...")
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            output = gdown.download(gdown_url, output_path, quiet=False, fuzzy=True)
            if output and os.path.exists(output) and os.path.getsize(output) > 1024:
                if progress_callback:
                    progress_callback(100, "Download selesai via gdown.")
                return output
        except Exception as e:
            print(f"[downloader] gdown download failed: {e}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        raise RuntimeError("Gagal mengunduh video. Pastikan link Google Drive publik, link YouTube yang valid, atau direct video URL (.mp4).")

    return output_path
