#!/usr/bin/env python3
"""Read-only, frame-matched depth stages HTTP preview; no camera/control access."""
import argparse
from collections import OrderedDict
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json, struct, sys, threading, time
import cv2
import numpy as np
import zmq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.common import TerrainDepthMessage
from utils.config import load_yaml_config
from rl_policy.direct_depth import DirectDepthConfig, preprocess_direct_depth_frame

HTML='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>实时深度处理对比</title><style>body{background:#10151e;color:#e5edf8;font:16px system-ui;margin:24px}img{width:100%;image-rendering:pixelated;border:1px solid #394558}p{line-height:1.6}#status{white-space:pre-wrap}</style><h2>实时深度处理对比</h2><p>左：相机原始 Z16（未加软件滤波）　中：当前深度滤波＋训练射线采样＋裁剪　右：同一帧＋Gaussian</p><img src="/stream.mjpg"><p>统一灰度：0 m 黑 → 最大深度白。原始无效点为黑；处理中无效点按当前模型规则转换。图像仅以 nearest 放大，不增加显示模糊。</p><p>中、右使用相同源帧和部署预处理函数；这是单帧处理对比，不代表 Actor 实际消费的完整历史帧。网页约 10 FPS，不改变控制频率。</p><div id="status"></div><script>async function tick(){try{let r=await fetch('/status',{cache:'no-store'});let s=await r.json();document.getElementById('status').textContent=(s.age_s===null||s.age_s>1?'⚠ 数据未更新 / 过期':'● 实时')+'\n'+JSON.stringify(s,null,2)}catch(e){document.getElementById('status').textContent='⚠ 网页服务连接中断'}}setInterval(tick,1000);tick()</script>'''
class State:
 def __init__(self):
  self.lock=threading.Lock();self.jpeg=None;self.updated=0;self.meta={};self.error=None
 def status(self):
  with self.lock:return dict(self.meta,age_s=round(time.monotonic()-self.updated,3) if self.updated else None,error=self.error)

def collect(state,args):
 cfg=DirectDepthConfig.from_policy_config(load_yaml_config(args.policy_config))
 ctx=zmq.Context();poll=zmq.Poller();socks=[]
 for port in (5596,5592):
  s=ctx.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.RCVHWM,120);s.connect(f'tcp://127.0.0.1:{port}');poll.register(s,zmq.POLLIN);socks.append(s)
 raw=OrderedDict();pending=OrderedDict();header=struct.Struct('<4sIIQd');last=0
 while True:
  try:
   events=dict(poll.poll(100))
   for i,s in enumerate(socks):
    if s not in events:continue
    for _ in range(120):
     try:data=s.recv(zmq.NOBLOCK)
     except zmq.Again:break
     if i==0:
      magic,w,h,n,scale=header.unpack_from(data)
      if magic!=b'RZ16':continue
      raw[n]=np.frombuffer(data,dtype='<u2',offset=header.size).reshape(h,w).astype(np.float32)*scale
      while len(raw)>120:raw.popitem(last=False)
     else:
      m=TerrainDepthMessage.from_bytes(data);pending[int(m.frame_number)]=m
      while len(pending)>120:pending.popitem(last=False)
   if time.monotonic()-last<1/args.fps:continue
   matches=raw.keys() & pending.keys()
   if not matches:continue
   n=max(matches);m=pending[n]
   before=preprocess_direct_depth_frame(m.depth_z,replace(cfg,gaussian_enabled=False))
   after=preprocess_direct_depth_frame(m.depth_z,replace(cfg,gaussian_enabled=True))
   original=np.rint(np.clip(raw[n]/cfg.max_depth_m,0,1)*255).astype(np.uint8)
   panels=[]
   for name,im in [('RAW Z16',original),('SAMPLE + CROP',before),('GAUSSIAN',after)]:
    canvas=np.full((350,512,3),20,np.uint8)
    cv2.putText(canvas,f'{name}  {im.shape[1]}x{im.shape[0]}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1)
    factor=min(512/im.shape[1],288/im.shape[0]);w=max(1,int(im.shape[1]*factor));h=max(1,int(im.shape[0]*factor))
    rgb=cv2.cvtColor(cv2.resize(im,(w,h),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    x=(512-w)//2;y=45+(288-h)//2;canvas[y:y+h,x:x+w]=rgb;panels.append(canvas)
   ok,jpg=cv2.imencode('.jpg',np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,90])
   if not ok:raise RuntimeError('JPEG encode failed')
   with state.lock:
    state.jpeg=jpg.tobytes();state.updated=time.monotonic();state.error=None
    state.meta={'frame':n,'profile':cfg.processing_profile,'policy_config':args.policy_config,'encoder_gaussian_enabled':cfg.gaussian_enabled,'max_depth_m':cfg.max_depth_m,'raw_size':list(original.shape),'processed_size':list(before.shape),'capture_age_ms':round((time.monotonic()-m.capture_monotonic_s)*1000,1),'matched_raw_frame':True}
   last=time.monotonic()
   for k in list(pending):
    if k<=n:del pending[k]
  except Exception as e:
   with state.lock:state.error=str(e)
   time.sleep(.1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--policy-config',required=True);p.add_argument('--port',type=int,default=8090);p.add_argument('--fps',type=float,default=10);args=p.parse_args()
 if not 0<args.fps<=30:p.error('fps must be in (0,30]')
 state=State();threading.Thread(target=collect,args=(state,args),daemon=True).start()
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def do_GET(self):
   if self.path.split('?')[0]=='/stream.mjpg':
    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
    try:
     while True:
      with state.lock:jpg=state.jpeg
      if jpg:self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'+jpg+b'\r\n');self.wfile.flush()
      time.sleep(1/args.fps)
    except (BrokenPipeError,ConnectionResetError):pass
   elif self.path.split('?')[0] in ('/','/status','/snapshot.jpg'):
    if self.path.startswith('/status'):data=json.dumps(state.status(),ensure_ascii=False).encode();kind='application/json'
    elif self.path.startswith('/snapshot'):
     with state.lock:data=state.jpeg
     if data is None:self.send_error(503,'Waiting for frames');return
     kind='image/jpeg'
    else:data=HTML.encode();kind='text/html; charset=utf-8'
    self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
   else:self.send_error(404)
 server=ThreadingHTTPServer(('0.0.0.0',args.port),Handler);server.daemon_threads=True
 print(f'Depth preview http://0.0.0.0:{args.port}',flush=True);server.serve_forever()
if __name__=='__main__':main()
