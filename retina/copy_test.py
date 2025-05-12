import struct
import numpy as np

def parse_evt3_to_numpy(evt3_path):
    """
    解析 EVT3 .raw 文件，返回 {'xy': Nx2, 't': N, 'p': N}，
    其中 t 用 uint64 存，避免溢出。
    """
    xs, ys, ps, ts = [], [], [], []
    pair_struct = struct.Struct("<QQ")

    with open(evt3_path, "rb") as f:
        # 跳过所有空行和注释行
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise ValueError("整个文件都是注释或空行")
            if line.strip() == b"" or line.startswith(b"%") or line.startswith(b"#"):
                continue
            f.seek(pos)
            break

        # 读二进制 addr+timestamp
        chunk = f.read(pair_struct.size)
        while chunk:
            addr, timestamp = pair_struct.unpack(chunk)
            xs.append(addr & 0xFFF)
            ys.append((addr >> 12) & 0xFFF)
            ps.append((addr >> 24) & 0x1)
            ts.append(timestamp)
            chunk = f.read(pair_struct.size)

    return {
        "xy": np.vstack([xs, ys]).T.astype(np.int64),
        # ⇓ 改这里，用 uint64
        "t":  np.array(ts, dtype=np.uint64),
        "p":  np.array(ps, dtype=np.uint8),
    }
