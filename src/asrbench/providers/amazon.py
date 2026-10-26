from __future__ import annotations

import binascii
import datetime as dt
import hashlib
import hmac
import json
import struct
from urllib.parse import quote

from .. import settings
from .base import Event, ProviderError, SessionContext, WebSocketProvider

_FIXED_HEADER_LEN = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}


def _crc(b: bytes) -> bytes:
    return struct.pack(">I", binascii.crc32(b) & 0xFFFFFFFF)


def encode_event(headers: dict[str, str], payload: bytes) -> bytes:
    hb = bytearray()
    for name, value in headers.items():
        n, v = name.encode(), value.encode()
        hb += bytes([len(n)]) + n + b"\x07" + struct.pack(">H", len(v)) + v
    prelude = struct.pack(">II", len(hb) + len(payload) + 16, len(hb))
    msg = prelude + _crc(prelude) + bytes(hb) + payload
    return msg + _crc(msg)


def decode_event(msg: bytes) -> tuple[dict[str, str], bytes]:
    total, hlen = struct.unpack(">II", msg[:8])
    if _crc(msg[:8]) != msg[8:12] or _crc(msg[:-4]) != msg[-4:]:
        raise ValueError("event-stream CRC mismatch")
    hb, payload = msg[12 : 12 + hlen], msg[12 + hlen : total - 4]
    headers: dict[str, str] = {}
    i = 0
    while i < len(hb):
        nlen = hb[i]
        name = hb[i + 1 : i + 1 + nlen].decode()
        typ = hb[i + 1 + nlen]
        i += 2 + nlen
        if typ in (6, 7):
            vlen = struct.unpack(">H", hb[i : i + 2])[0]
            raw = hb[i + 2 : i + 2 + vlen]
            headers[name] = raw.decode() if typ == 7 else raw.hex()
            i += 2 + vlen
        else:
            i += _FIXED_HEADER_LEN[typ]
    return headers, payload


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def presign_url(
    host: str, path: str, params: dict[str, str], *, region: str, access_key: str, secret_key: str,
    session_token: str | None = None, service: str = "transcribe", now: dt.datetime | None = None,
) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    amz_date, datestamp = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    q = dict(params)
    q.update({
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": "300",
        "X-Amz-SignedHeaders": "host",
    })
    if session_token:
        q["X-Amz-Security-Token"] = session_token
    cqs = "&".join(f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(q.items()))
    creq = "\n".join(["GET", path, cqs, f"host:{host}\n", "host", hashlib.sha256(b"").hexdigest()])
    sts = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(creq.encode()).hexdigest()])
    key = _hmac(_hmac(_hmac(_hmac(f"AWS4{secret_key}".encode(), datestamp), region), service), "aws4_request")
    sig = hmac.new(key, sts.encode(), hashlib.sha256).hexdigest()
    return f"wss://{host}{path}?{cqs}&X-Amz-Signature={sig}"


class AmazonTranscribeProvider(WebSocketProvider):
    """Amazon Transcribe streaming over WebSocket (SigV4-presigned, event-stream framed).

    The streaming API has no finalize call, so forced endpointing is unsupported and the
    harness uses its no-ack fallback; results finalize on Transcribe's own segmentation.
    """

    family = "amazon"
    sample_rates = (8000, 16000)
    supports_forced = False
    has_endpoint_ack = False
    api_key_env = "AWS_ACCESS_KEY_ID"
    hosts = ()

    def _region(self) -> str:
        return settings.env("AWS_REGION") or self.params.get("region", "ap-south-1")

    def url(self, ctx: SessionContext) -> str:
        secret = settings.env("AWS_SECRET_ACCESS_KEY")
        if not secret:
            raise ProviderError("AWS_SECRET_ACCESS_KEY is not set (add it to .env)")
        lp = self.lang_params(ctx.lang)
        region = self._region()
        params = {
            "language-code": lp.get("language_code", "hi-IN"),
            "media-encoding": "pcm",
            "sample-rate": str(ctx.sample_rate),
        }
        if self.params.get("partial_results_stability"):
            params["enable-partial-results-stabilization"] = "true"
            params["partial-results-stability"] = str(self.params["partial_results_stability"])
        return presign_url(
            f"transcribestreaming.{region}.amazonaws.com:8443", "/stream-transcription-websocket", params,
            region=region, access_key=self.api_key(), secret_key=secret,
            session_token=settings.env("AWS_SESSION_TOKEN"),
        )

    def audio_message(self, chunk: bytes) -> bytes:
        return encode_event(
            {":content-type": "application/octet-stream", ":event-type": "AudioEvent", ":message-type": "event"},
            chunk,
        )

    def finish_message(self) -> bytes:
        return self.audio_message(b"")

    def parse(self, raw) -> list[Event]:
        if isinstance(raw, str):
            return [Event("error", raw[:500])]
        headers, payload = decode_event(raw)
        if headers.get(":message-type") == "exception":
            body = payload.decode(errors="replace")
            return [Event("error", f"{headers.get(':exception-type')}: {body[:300]}")]
        if headers.get(":event-type") != "TranscriptEvent":
            return []
        results = (json.loads(payload).get("Transcript") or {}).get("Results") or []
        evs: list[Event] = []
        for res in results:
            alts = res.get("Alternatives") or [{}]
            text = alts[0].get("Transcript") or ""
            if res.get("IsPartial"):
                if text:
                    evs.append(Event("partial", text, meta={"result_id": res.get("ResultId")}))
            else:
                evs.append(Event("final", text, meta={"result_id": res.get("ResultId")}))
                evs.append(Event("turn_end", text))
        return evs
