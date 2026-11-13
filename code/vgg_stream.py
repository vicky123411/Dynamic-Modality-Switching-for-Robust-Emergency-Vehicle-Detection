"""Stream one VGGSound shard from the HF mirror and keep only the clips we need.
For each wanted clip: centre frame (t=5 s) -> JPEG (short side 256),
centre 4 s of audio (3-7 s) -> 22.05 kHz mono WAV.  The mp4 itself is not kept.
usage: python vgg_stream.py SHARD_INDEX OUTDIR CSV
"""
import csv, os, subprocess, sys, tarfile, tempfile, urllib.request, json

shard, outdir, csv_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
POS = {'ambulance siren', 'fire truck siren', 'police car (siren)'}
NEG = {'car passing by', 'driving buses', 'vehicle horn, car horn, honking',
       'driving motorcycle', 'car engine idling', 'engine accelerating, revving, vroom',
       'civil defense siren'}
want = {}
for r in csv.reader(open(csv_path)):
    if r[2] in POS or r[2] in NEG:
        want['%s_%06d' % (r[0], int(r[1]))] = (r[2], r[3])
os.makedirs(os.path.join(outdir, 'frames'), exist_ok=True)
os.makedirs(os.path.join(outdir, 'audio'), exist_ok=True)
url = 'https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_%02d.tar.gz' % shard
local = os.environ.get('VGG_LOCAL')
if local:
    proc = subprocess.Popen(['cat', local], stdout=subprocess.PIPE)
else:
    proc = subprocess.Popen(['curl', '-sL', '--retry', '5', url], stdout=subprocess.PIPE)
kept, seen = [], 0
tmpdir = tempfile.mkdtemp()
complete = True
try:
  with tarfile.open(fileobj=proc.stdout, mode='r|gz') as tf:
    for m in tf:
        if not m.isfile():
            continue
        seen += 1
        key = os.path.splitext(os.path.basename(m.name))[0]
        if key not in want:
            continue
        jpg = os.path.join(outdir, 'frames', key + '.jpg')
        wav = os.path.join(outdir, 'audio', key + '.wav')
        if os.path.exists(jpg) and os.path.exists(wav) and os.path.getsize(wav) > 1000:
            lab, split = want[key]; kept.append({'key': key, 'class': lab, 'split': split, 'ok': True}); continue
        data = tf.extractfile(m).read()
        mp4 = os.path.join(tmpdir, 'c.mp4')
        open(mp4, 'wb').write(data)
        jpg = os.path.join(outdir, 'frames', key + '.jpg')
        wav = os.path.join(outdir, 'audio', key + '.wav')
        r1 = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '5', '-i', mp4, '-frames:v', '1',
                             '-vf', "scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'",
                             '-q:v', '3', jpg], capture_output=True)
        r2 = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '3', '-t', '4', '-i', mp4, '-vn',
                             '-ac', '1', '-ar', '22050', wav], capture_output=True)
        ok = os.path.exists(jpg) and os.path.getsize(jpg) > 0 and os.path.exists(wav) and os.path.getsize(wav) > 1000
        lab, split = want[key]
        kept.append({'key': key, 'class': lab, 'split': split, 'ok': ok})
except Exception as e:
    complete = False
    print('shard', shard, 'interrupted:', repr(e)[:100])
with open(os.path.join(outdir, 'shard_%02d.json' % shard), 'w') as f:
    json.dump({'shard': shard, 'members': seen, 'kept': kept, 'complete': complete}, f)
print('shard', shard, 'complete', complete, 'members', seen, 'kept', len(kept), 'ok', sum(k['ok'] for k in kept))
