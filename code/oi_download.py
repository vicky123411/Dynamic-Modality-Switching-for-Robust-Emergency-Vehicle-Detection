import pandas as pd, cv2, numpy as np, urllib.request, os, sys
from concurrent.futures import ThreadPoolExecutor
sel=pd.read_csv(sys.argv[1]); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
def get(r):
    p=f"{out}/{r.image_id}.jpg"
    if os.path.exists(p): return True
    url=f"https://open-images-dataset.s3.amazonaws.com/{r.oi_split}/{r.image_id}.jpg"
    for _ in range(3):
        try:
            b=urllib.request.urlopen(url,timeout=60).read()
            im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
            if im is None: return False
            h,w=im.shape[:2]; s=256/min(h,w)
            im=cv2.resize(im,(max(1,round(w*s)),max(1,round(h*s))),interpolation=cv2.INTER_AREA)
            cv2.imwrite(p,im,[cv2.IMWRITE_JPEG_QUALITY,92]); return True
        except Exception as e: err=e
    return False
with ThreadPoolExecutor(8) as ex: ok=list(ex.map(get, sel.itertuples()))
print('ok',sum(ok),'of',len(ok))
