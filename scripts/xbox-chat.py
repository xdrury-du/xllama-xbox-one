#!/usr/bin/env python3
# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT

"""Loopback-only Mac chat front end for the Xbox One Gemma experiment."""

import argparse
import ipaddress
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_MODEL = "gemma3-270m"
MODELS = {"gemma3-270m": "Gemma 3 270M", "ministral3-3b-instruct": "Ministral 3 3B Instruct"}
PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xbox · local chat</title>
<style>
:root{color-scheme:dark;font:16px/1.55 system-ui,-apple-system,sans-serif;background:#111612;color:#e9f1ea}
*{box-sizing:border-box}body{margin:0}main{max-width:900px;margin:auto;padding:30px 24px}
header{display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:26px;margin:0}p{margin:6px 0;color:#aabaaa}
.badge{font-size:13px;color:#b5e997}.panel{background:#1b241d;border:1px solid #344437;border-radius:14px;padding:18px;margin-top:22px}
#messages:empty:after{content:'Type on your Mac. Gemma runs on your Xbox.';color:#8d9e90;display:block;padding:35px 0;text-align:center}
.message{margin:0 0 18px}.role{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:#a8d99b}.text{white-space:pre-wrap;overflow-wrap:anywhere;margin-top:5px}
textarea{font:inherit;background:#111912;color:inherit;border:1px solid #4b604d;border-radius:9px;width:100%;min-height:112px;padding:12px;resize:vertical}
textarea:focus,button:focus-visible,input:focus,select:focus{outline:2px solid #9edd79;outline-offset:3px}
button{font:inherit;border:1px solid #526954;border-radius:8px;padding:9px 16px;background:#263b28;color:inherit;cursor:pointer}
button.primary{background:#b0e58c;color:#14210d;border-color:#b0e58c;font-weight:650}button:disabled{opacity:.45;cursor:wait}
.bar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:12px;flex-wrap:wrap}
select{font:inherit;background:#111912;color:inherit;border:1px solid #4b604d;border-radius:5px;padding:8px;max-width:100%}input{font:inherit;background:#111912;color:inherit;border:1px solid #4b604d;border-radius:5px;width:80px;padding:5px}
#status{font-size:14px}#metrics{color:#b5c7b7;font-size:14px;min-height:24px;margin-top:12px}.hint{font-size:12px;color:#8e9f91}
</style>
<main>
<header><div><h1>Xbox · local chat</h1><p>One X · CPU · text models</p></div><button id="new" type="button">New chat</button></header>
<p id="status" role="status">Checking Xbox connection…</p>
<section class="panel"><div class="bar" style="margin:0 0 18px"><label for="model">Model <select id="model"><option value="gemma3-270m">Gemma 3 270M</option><option value="ministral3-3b-instruct">Ministral 3 3B Instruct</option></select></label><button id="load" type="button">Download / load model</button></div><div id="messages" role="log" aria-label="Conversation" aria-live="polite"></div>
<form id="form"><label for="prompt">Message</label><textarea id="prompt" placeholder="Ask Gemma something…" required></textarea>
<div class="bar"><label for="limit">Reply limit <input id="limit" type="number" min="1" max="512" value="128"> tokens</label><button id="send" class="primary" type="submit">Send to Xbox</button></div></form>
<div id="metrics" role="status"></div><p class="hint">⌘ Enter to send · Replies stream here · Chats and request timings are saved on this Mac.</p></section>
<p class="hint">Keep xllama open on the Xbox. Total time includes network, loading and prompt processing. Internal decode tok/s is not available through this API.</p>
</main><script>
const token=__TOKEN__, q=s=>document.querySelector(s);
let history=[];
function message(role,text){const box=document.createElement('div');box.className='message';const label=document.createElement('div');label.className='role';label.textContent=role;const body=document.createElement('div');body.className='text';body.textContent=text;box.append(label,body);q('#messages').append(box);return body;}
function busy(on){for(const id of ['send','new','prompt','limit','model','load'])q('#'+id).disabled=on;}
function newChat(){history=[];q('#messages').replaceChildren();q('#metrics').textContent='New conversation';q('#prompt').focus();}
q('#new').onclick=newChat;q('#model').onchange=newChat;
async function refreshModels(){const response=await fetch('/models');const data=await response.json();if(!response.ok)throw Error(data.error||'Model discovery failed');for(const model of data.models){const option=q('#model').querySelector('option[value="'+model.id+'"]');option.textContent=model.label+(model.installed?'':' · download required');}}
async function readLines(response,onEvent){if(!response.ok){const value=await response.json();throw Error(value.error||'HTTP '+response.status);}const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';for(;;){const part=await reader.read();buffer+=decoder.decode(part.value||new Uint8Array(),{stream:!part.done});let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(line.trim())onEvent(JSON.parse(line));}if(part.done)break;}}
q('#load').onclick=async()=>{const model=q('#model').value;busy(true);q('#metrics').textContent='Contacting Xbox…';let complete=false;try{const response=await fetch('/pull',{method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':token},body:JSON.stringify({model})});await readLines(response,event=>{if(event.error)throw Error(event.error);q('#metrics').textContent=event.total?event.status+' · '+Math.round(event.completed/event.total*100)+'%':event.status;if(event.status==='success')complete=true;});if(!complete)throw Error('Model loading ended without success.');await refreshModels();history=[];q('#messages').replaceChildren();q('#metrics').textContent='Model loaded on Xbox. Ready to chat.';}catch(error){q('#metrics').textContent='Error: '+error.message;}finally{busy(false);}};
q('#prompt').onkeydown=e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();q('#form').requestSubmit();}};
q('#form').onsubmit=async e=>{e.preventDefault();if(q('#send').disabled)return;const text=q('#prompt').value.trim();if(!text)return;
 const pending=[...history,{role:'user',content:text}];message('You',text);const output=message(q('#model').selectedOptions[0].textContent.replace(' · download required',''),'');busy(true);q('#prompt').value='';q('#metrics').textContent='Waiting for Xbox…';let reply='',done=false;
 try{const response=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':token},body:JSON.stringify({model:q('#model').value,messages:pending,max_tokens:Number(q('#limit').value)})});
 if(!response.ok){const err=await response.json();throw Error(err.error||'HTTP '+response.status);}
 const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
 for(;;){const part=await reader.read();buffer+=decoder.decode(part.value||new Uint8Array(),{stream:!part.done});let end;
 while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(!line.trim())continue;const event=JSON.parse(line);
 if(event.error)throw Error(event.error);
 if(event.delta){reply+=event.delta;output.textContent=reply;q('#metrics').textContent='Generating on Xbox…';}
 if(event.done){done=true;const u=event.usage||{};q('#metrics').textContent=(u.completion_tokens??'?')+' reply tokens · '+event.total_seconds.toFixed(2)+'s total · '+(event.first_text_seconds===null?'no text':event.first_text_seconds.toFixed(2)+'s to first text')+' · '+event.finish_reason;}}
 if(part.done)break;}
 if(!done)throw Error('Xbox stream ended before completion.');history=[...pending,{role:'assistant',content:reply}];
 }catch(err){q('#metrics').textContent='Error: '+err.message;if(!reply)output.textContent='No completed reply. Your earlier conversation is preserved.';q('#prompt').value=text;}
 finally{busy(false);q('#prompt').focus();}};
fetch('/health').then(r=>r.json()).then(r=>{q('#status').textContent=r.status==='ok'?'● Connected to Xbox at __XBOX__':'Xbox unavailable: '+(r.error||r.status);q('#status').className=r.status==='ok'?'badge':'';}).catch(()=>q('#status').textContent='Unable to check Xbox connection.');
refreshModels().catch(error=>q('#metrics').textContent=error.message);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def local_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def json_response(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.local_host():
            self.json_response(403, {"error": "Use the loopback chat URL"})
            return
        if self.path == "/":
            body = PAGE.replace("__TOKEN__", json.dumps(self.server.token)).replace(
                "__XBOX__", self.server.xbox
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/models":
            try:
                with self.server.opener.open(self.server.base + "/v1/models", timeout=8) as response:
                    installed = {item["id"] for item in json.load(response).get("data", [])}
                self.json_response(200, {"models": [{"id": key, "label": label, "installed": key in installed}
                                                     for key, label in MODELS.items()]})
            except Exception as error:
                self.json_response(502, {"error": str(error)})
        elif self.path == "/health":
            try:
                with self.server.opener.open(self.server.base + "/health", timeout=5) as response:
                    self.json_response(200, json.load(response))
            except Exception as error:
                self.json_response(502, {"error": str(error)})
        else:
            self.json_response(404, {"error": "Not found"})

    def do_POST(self):
        if not self.local_host():
            self.json_response(403, {"error": "Use the loopback chat URL"})
            return
        if self.path not in ("/chat", "/pull"):
            self.json_response(404, {"error": "Not found"})
            return
        if not secrets.compare_digest(self.headers.get("X-Local-Token", ""), self.server.token):
            self.json_response(403, {"error": "Open the local chat page to send messages"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Message is too large or empty")
            incoming = json.loads(self.rfile.read(size))
            model = incoming.get("model", DEFAULT_MODEL)
            if not isinstance(model, str) or model not in MODELS:
                raise ValueError("Model is not approved for this experiment")
            if self.path == "/pull":
                self.pull_model(model)
                return
            messages = incoming["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("A conversation is required")
            for item in messages:
                if item.get("role") not in ("user", "assistant") or not isinstance(item.get("content"), str):
                    raise ValueError("Invalid conversation message")
            if messages[-1]["role"] != "user":
                raise ValueError("The last message must be from the user")
            limit = incoming.get("max_tokens", 128)
            if type(limit) is not int or not 1 <= limit <= 512:
                raise ValueError("Reply limit must be 1–512 tokens")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.json_response(400, {"error": str(error)})
            return
        if not self.server.gate.acquire(blocking=False):
            self.json_response(409, {"error": "Another Xbox request is still running"})
            return
        started = False
        t0 = time.perf_counter()
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "hardware": "xbox-one-x-operator-declared",
                  "model": model, "messages": messages, "max_tokens": limit,
                  "text": "", "first_text_seconds": None, "usage": {}, "error": None}

        def emit(event):
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()

        try:
            payload = {"model": model, "messages": messages, "max_tokens": limit,
                       "temperature": 0, "stream": True}
            request = urllib.request.Request(self.server.base + "/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with self.server.opener.open(request, timeout=120) as response:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                started = True
                finished = False
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        finished = True
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    if event.get("usage"):
                        record["usage"] = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {}).get("content") or ""
                        if delta:
                            if record["first_text_seconds"] is None:
                                record["first_text_seconds"] = time.perf_counter() - t0
                            record["text"] += delta
                            emit({"delta": delta})
                        if choice.get("finish_reason"):
                            record["finish_reason"] = choice["finish_reason"]
                if not finished:
                    raise RuntimeError("Xbox stream closed without its completion marker")
                record["total_seconds"] = time.perf_counter() - t0
                emit({"done": True, "usage": record["usage"],
                      "first_text_seconds": record["first_text_seconds"],
                      "total_seconds": record["total_seconds"],
                      "finish_reason": record.get("finish_reason", "unknown")})
        except Exception as error:
            record["error"] = str(error)
            record["total_seconds"] = time.perf_counter() - t0
            if isinstance(error, urllib.error.HTTPError):
                try:
                    record["error"] = error.read().decode()
                except Exception:
                    pass
            try:
                if started:
                    emit({"error": record["error"]})
                else:
                    self.json_response(502, {"error": record["error"]})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            try:
                with self.server.log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record) + "\n")
            finally:
                self.server.gate.release()

    def pull_model(self, model):
        if not self.server.gate.acquire(blocking=False):
            self.json_response(409, {"error": "Another Xbox request is still running"})
            return
        started = False
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": model,
                  "operation": "pull", "success": False, "error": None}
        begin = time.perf_counter()
        try:
            request = urllib.request.Request(self.server.base + "/api/pull",
                data=json.dumps({"model": model, "stream": True}).encode(),
                headers={"Content-Type": "application/json"})
            with self.server.opener.open(request, timeout=180) as response:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                started = True
                for line in response:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
                    if event.get("status") == "success":
                        record["success"] = True
                if not record["success"]:
                    raise RuntimeError("Xbox pull ended without success")
        except Exception as error:
            record["error"] = str(error)
            if isinstance(error, urllib.error.HTTPError):
                try:
                    record["error"] = error.read().decode()
                except Exception:
                    pass
            try:
                if started:
                    self.wfile.write((json.dumps({"error": record["error"]}) + "\n").encode())
                    self.wfile.flush()
                else:
                    self.json_response(502, {"error": record["error"]})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            record["total_seconds"] = time.perf_counter() - begin
            try:
                with self.server.log_path.with_name("pulls.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record) + "\n")
            finally:
                self.server.gate.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbox", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    address = ipaddress.IPv4Address(args.xbox)
    if not address.is_private:
        parser.error("Use the Xbox's private LAN IPv4 address")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.xbox = str(address)
    server.base = f"http://{address}:11434"
    server.token = secrets.token_urlsafe(32)
    server.gate = threading.Lock()
    server.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server.log_path = Path(__file__).resolve().with_name("runs.jsonl")
    print(f"Xbox chat: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Xbox: {server.base} · approved models: {', '.join(MODELS)}", flush=True)
    print(f"Local chat and timing log: {server.log_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
