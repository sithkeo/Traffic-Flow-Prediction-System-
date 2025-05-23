"""
Traffic‑Flow‑Prediction‑System – **Single‑file GUI** (`gui.py`)
==============================================================
A *self‑contained* Tkinter application bundling every helper module so the repo
only needs one Python file.  Now updated to:

* **Average traffic flow** is always calculated and displayed – even when the
  deep‑learning models aren’t available (we fall back to a nominal 60 veh/5 min
  instead of zero).
* **Map colouring** – only the *best* route (first in the list) is highlighted
  blue; alternative routes are faint grey.
* **Resource tweaks** – CSV path updated to `output/neighbouring_intersections.csv`
  and the default background image is now `gui_image/forest.png` (matches your
  latest repo state).

The code still degrades gracefully if heavy dependencies (`numpy`, `pandas`,
`keras`, `geopy`, `folium`, `Pillow`) are missing.
"""
from __future__ import annotations

# ──────────────────── stdlib ────────────────────
import csv, json, textwrap, webbrowser
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

# ───────────────── third‑party (optional) ─────────────────
try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except ModuleNotFoundError:
    print("[WARN] pandas / numpy not available – ML models disabled"); pd = np = None  # type: ignore

try:
    from keras.models import load_model  # type: ignore
except ModuleNotFoundError:
    def load_model(*_a, **_k): raise RuntimeError("Keras missing – cannot load models")  # type: ignore

try:
    from geopy.distance import geodesic  # type: ignore
except ModuleNotFoundError:
    def geodesic(p1, p2): (lat1, lon1), (lat2, lon2) = p1, p2; return type("G", (), {"km": ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111})()  # type: ignore

try:
    import folium  # type: ignore
except ModuleNotFoundError:
    folium = None

try:
    from PIL import Image, ImageEnhance, ImageTk  # type: ignore
except ModuleNotFoundError:
    Image = ImageEnhance = ImageTk = None  # type: ignore

# ───────────────────── Tkinter ─────────────────────
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# ───────── paths & constants ─────────
BASE_DIR   = Path(__file__).resolve().parent
TRAFFIC_CSV = BASE_DIR / "output" / "neighbouring_intersections.csv"
IMG_BG     = BASE_DIR / "gui_image" / "forest.png"

# ═════════ Section 1 – ML helpers ═════════
model_cache: dict[str, Any] = {}
_neighbors: dict[str, List[str]] | None = None

def load_neighbors() -> dict[str, List[str]]:
    global _neighbors
    if _neighbors is not None: return _neighbors
    if pd is None or not TRAFFIC_CSV.exists():
        _neighbors = {}; return _neighbors
    df = pd.read_csv(TRAFFIC_CSV)
    _neighbors = {str(r["Scats_number"]): str(r["Neighbours"]).split(";") for _, r in df.iterrows()}
    return _neighbors


def _prep(ts: datetime, n: int):
    base = [ts.hour/23, ts.minute/59, ts.weekday()/6, int(ts.weekday()<5), ts.day/31, (ts.month-1)/11]
    arr = [base]*n if n>1 else base
    return np.array(arr).reshape((1, n, len(base))) if n>1 else np.array(arr).reshape(1,-1)

def _denorm(x, lo=0, hi=500): return int(round(lo + x*(hi-lo)))

def _time_fac(v: float, ts: datetime): return v*1.5 if 7<=ts.hour<=9 or 16<=ts.hour<=18 else v

@lru_cache(maxsize=10000)
def predict_site(site: str, ts: datetime, mdl: str) -> int | None:
    if np is None: return None
    key = f"{mdl.lower()}_{site}"
    if key not in model_cache:
        try:
            model_cache[key] = load_model(BASE_DIR/"model"/"sites_models"/f"{key}.h5", compile=False)
            print(f"Loaded {mdl} model for site {site}")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] {exc}"); model_cache[key]=None
    m = model_cache[key]
    if m is None: return None
    try:
        seq = m.input_shape[1] if len(m.input_shape)==3 else 1
        val = m.predict(_prep(ts,seq), verbose=0)
        v = val[0][-1] if val.ndim==3 else val[0][0]
        return _denorm(_time_fac(v,ts))
    except Exception as exc:
        print(f"[WARN] predict failed {site}: {exc}"); return None

# ═════════ Section 2 – distance helpers ═════════
_inter: dict[str, Tuple[float,float]] | None = None

def _load_inter():
    global _inter
    if _inter is not None: return _inter
    _inter={}
    if pd is None or not TRAFFIC_CSV.exists(): return _inter
    df=pd.read_csv(TRAFFIC_CSV); _inter={str(r["Scats_number"]):(float(r["Latitude"]),float(r["Longitude"])) for _,r in df.iterrows()}; return _inter

@lru_cache(maxsize=1_000_000)
def dist_km(a:str,b:str)->float:
    d=_load_inter();
    return geodesic(d.get(a,(0,0)),d.get(b,(0,0))).km if a in d and b in d else 0.0

# ═════════ Section 3 – path‑finder ═════════

def _speed(flow:int|None,peak:bool)->float:
    if flow is None: return 50 if not peak else 35
    flow=min(flow,400)
    return 60-(flow-60)*0.1 if 60<=flow<250 else (25 if flow>=250 else (60 if not peak else 50))

def pathfinder(src:str,dst:str,ts:datetime,mdl:str,k:int=5):
    from heapq import heappush, heappop
    nbr=load_neighbors(); paths=[]; seen=set()
    heap=[(0.0,0.0,[src],ts,0.0,0)]  # est,dist,path,cur_ts,flow_sum,steps
    while heap and len(paths)<k:
        est,dist_sofar,path,cur_ts,flow_sum,steps=heappop(heap)
        node=path[-1]; state=(node,cur_ts.strftime('%H%M'))
        if state in seen: continue
        seen.add(state)
        if node==dst:
            avg=flow_sum/max(1,steps)
            paths.append((est,dist_sofar,path,avg))
            continue
        flow=predict_site(node,cur_ts,mdl)
        if flow is None: flow=60  # nominal fallback
        peak=7<=cur_ts.hour<=9 or 16<=cur_ts.hour<=18
        for nb in nbr.get(node,[]):
            if nb in path: continue
            d=dist_km(node,nb); sp=_speed(flow,peak); mins=(d/sp)*60
            heappush(heap,(est+mins,dist_sofar+d,path+[nb],cur_ts+timedelta(minutes=mins),flow_sum+flow,steps+1))
    return sorted(paths,key=lambda x:x[0])

# ═════════ Section 4 – GUI ═════════
ALL_SCATS=sorted(load_neighbors().keys(), key=int)
if ALL_SCATS:
    print("SCATS nodes:\n    "+textwrap.fill(", ".join(ALL_SCATS),120,subsequent_indent="    ")+"\n")
else:
    print("[WARN] neighbours list empty – check CSV")

class TrafficFlowGUI(tk.Tk):
    BG_WINDOW="#F0F0F0"; BG_FRAME="#FFFFFF"; FG_TEXT="#333333"; BTN_GREEN="#4CAF50"
    def __init__(self):
        super().__init__(); self.title("Traffic Flow Prediction System"); self.geometry("820x600"); self.configure(bg=self.BG_WINDOW)
        self._canvas=tk.Canvas(self,width=800,height=540,highlightthickness=0); self._canvas.pack(fill="both",expand=True)
        self._bg(); self._form(); self._results(); self._status_bar(); self.generated:List[Tuple[float,float,List[str],float]]=[]

    def _bg(self):
        if Image and IMG_BG.exists():
            img=Image.open(IMG_BG).convert("RGBA"); img=ImageEnhance.Brightness(img).enhance(0.7); img=img.resize((800,540),Image.LANCZOS)
            self._bg_img=ImageTk.PhotoImage(img); self._canvas.create_image(0,0,image=self._bg_img,anchor="nw")

    def _form(self):
        f=tk.Frame(self._canvas,bg=self.BG_FRAME,highlightbackground=self.BG_WINDOW,highlightthickness=2); self._canvas.create_window(400,190,window=f)
        r=0; tk.Label(f,text="Route Detection",font=("Helvetica",13,"bold"),bg=self.BG_FRAME).grid(row=r,column=0,columnspan=2,pady=4); r+=1
        tk.Label(f,text="Select Model:",bg=self.BG_FRAME).grid(row=r,column=0,sticky="e"); self._model=tk.StringVar(value="LSTM"); ttk.Combobox(f,textvariable=self._model,state="readonly",values=("LSTM","GRU","SAES")).grid(row=r,column=1,sticky="ew"); r+=1
        tk.Label(f,text="Origin Node:",bg=self.BG_FRAME).grid(row=r,column=0,sticky="e"); self._src=tk.Entry(f); self._src.grid(row=r,column=1,sticky="ew"); r+=1
        tk.Label(f,text="Destination Node:",bg=self.BG_FRAME).grid(row=r,column=0,sticky="e"); self._dest=tk.Entry(f); self._dest.grid(row=r,column=1,sticky="ew"); r+=1
        tk.Label(f,text="Date/Time:",bg=self.BG_FRAME).grid(row=r,column=0,sticky="e"); self._dt=tk.Entry(f,fg="grey"); self._dt.insert(0,"YYYY-MM-DD HH:MM"); self._dt.bind("<FocusIn>",lambda *_:(self._dt.delete(0,tk.END),self._dt.configure(fg="black"))); self._dt.grid(row=r,column=1,sticky="ew"); r+=1
        tk.Button(f,text="Generate Route",bg=self.BTN_GREEN,fg="white",bd=0,command=self._generate).grid(row=r,column=0,columnspan=2,pady=10,sticky="ew"); f.columnconfigure(0,weight=1); f.columnconfigure(1,weight=2)

    def _results(self):
        self._txt=ScrolledText(self._canvas,height=8,width=90,wrap="word",bg="#F5F5F5",font=("Arial",10)); self._txt.config(state="disabled"); self._canvas.create_window(400,390,window=self._txt)
        self._canvas.create_window(400,460,window=tk.Button(self._canvas,text="View Route",bg=self.BTN_GREEN,fg="white",bd=0,command=self._view))

    def _status_bar(self):
        bar=tk.Frame(self); bar.pack(side=tk.BOTTOM,fill=tk.X); self._status=tk.Label(bar,text="Ready",bd=1,relief=tk.SUNKEN,anchor="w"); self._status.pack(side=tk.LEFT,fill=tk.X,expand=True)

    def _set_status(self,msg): self._status.configure(text=msg); self.update_idletasks()
    def _print(self,msg): self._txt.config(state="normal"); self._txt.delete("1.0",tk.END); self._txt.insert(tk.END,msg); self._txt.config(state="disabled")

    def _coords(self,sc):
        with open(TRAFFIC_CSV,newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["Scats_number"].strip()==sc:
                    return float(r["Longitude"])+0.00123, float(r["Latitude"])+0.00123, r.get("Site description","")
        return None

    # ----- Generate # TODO FIX VEH/5MIN | CURRENTLY JUST PLACEHOLDER AND DISPLAYS 60/5MIN CAUSE NO MODEL DATA -----
    def _generate(self):
        src,dst,mdl=self._src.get().strip(),self._dest.get().strip(),self._model.get().upper()
        if not src or not dst:
            messagebox.showerror("Input","Please enter both nodes"); return
        try: when=datetime.strptime(self._dt.get().strip(),"%Y-%m-%d %H:%M")
        except ValueError: when=datetime.now()
        self._set_status("Finding paths…")
        try: self.generated=pathfinder(src,dst,when,mdl)
        except Exception as exc:
            self.generated=[]; self._print(f"Error: {exc}"); self._set_status("Error"); print(exc); return
        if not self.generated:
            self._print("No routes found"); self._set_status("Done – 0 routes"); return
        lines=[f"Routes {src} → {dst} ({mdl}) {when:%Y-%m-%d %H:%M}","—"*80]
        for i,(t,d,p,avg) in enumerate(self.generated,1):
            lines+= [f"Route {i}", f"  Time     : {t:.2f} min", f"  Distance : {d:.2f} km", f"  Avg flow : {avg:.0f} veh/5min", f"  Path     : {' → '.join(p)}", ""]
        self._print("\n".join(lines)); self._set_status(f"Done – {len(self.generated)} routes")

    # ----- Mapping helpers -----
    def _geojson(self):
        best=self.generated[0][2] if self.generated else []
        feats=[]
        for path_tuple in self.generated[::-1]:
            path=path_tuple[2]; colour="#3484F0" if path is best else "#B3B3B3"
            coords=[[lon,lat] for sc in path if (c:=self._coords(sc)) for lon,lat,*_ in [c]]
            if coords:
                feats.append({"type":"Feature","properties":{"stroke":colour,"stroke-width":5},"geometry":{"type":"LineString","coordinates":coords}})
        return json.dumps({"type":"FeatureCollection","features":feats})

    def _draw_nodes(self,fmap):
        with open(TRAFFIC_CSV,newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lon=float(r["Longitude"])+0.00123; lat=float(r["Latitude"])+0.00123; desc=r.get("Site description","")
                folium.Circle(location=[lat,lon],radius=5,color="#5A5A5A",fill=True,fill_opacity=0.7,tooltip=f"SCATS {r['Scats_number']}: {desc}").add_to(fmap)

    def _view(self):
        if folium is None:
            messagebox.showerror("Missing","Install 'folium'"); return
        fmap=folium.Map(location=[-37.831219,145.056965],zoom_start=13,tiles="cartodbpositron")
        if self.generated:
            for idx, (t, dist, path, flow) in enumerate(self.generated):
                coords = [
                    [lat, lon]                       # Folium wants [lat, lon]
                    for sc in path
                    if (c := self._coords(sc))       # look-up lat/lon for every SCATS
                    for lon, lat, *_ in [c]          # unpack tuple
                ]
                if not coords:
                    continue

                folium.PolyLine(
                    coords,
                    color="#3484F0" if idx == 0 else "#363636",
                    weight=6 if idx == 0 else 3,
                    opacity=0.9 if idx == 0 else 0.65,
                ).add_to(fmap)

            # Mark start & end of the *best* (blue) route
            best_path = self.generated[0][2]                # Path list is in slot 2
            if best_path:
                start_sc, end_sc = best_path[0], best_path[-1]
                for sc_id, col in ((start_sc, "green"), (end_sc, "red")):
                    if (c := self._coords(sc_id)):
                        lon, lat, desc = c
                        folium.Marker(
                            [lat, lon],
                            popup=f"SCATS {sc_id}<br>{desc}",
                            icon=folium.Icon(color=col),
                        ).add_to(fmap)
        self._draw_nodes(fmap); fmap.save("index.html"); webbrowser.open("index.html")

# ═════ main ═════
if __name__=="__main__": TrafficFlowGUI().mainloop()
