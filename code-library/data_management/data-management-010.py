import sqlite3, pickle, threading, atexit, time, pathlib, os
from .logger import log
DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "lab.db"
con = sqlite3.connect(DB, check_same_thread=False)
cur = con.cursor()
cur.execute("create table if not exists kv(key text primary key,val blob,ts real)")
lock = threading.Lock()
def put(k, v):
    with lock:
        cur.execute("replace into kv values(?,?,?)", (k, pickle.dumps(v), time.time()))
        con.commit()
    log("store", k, "saved")
def get(k, default=None):
    row = cur.execute("select val from kv where key=?",(k,)).fetchone()
    return pickle.loads(row[0]) if row else default
def scan(prefix):
    return [(k, pickle.loads(v)) for k,v in cur.execute("select key,val from kv where key like ?", (f"{prefix}%",))]
atexit.register(con.close)
log("store", "ready @", DB)