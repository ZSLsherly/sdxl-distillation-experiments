set -eu
/data1/shenglin.zeng/dymd_prior_8_20261101/env/bin/python - <<'PY'
import json,pathlib,imageio_ffmpeg
b=pathlib.Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
cache=b/'video_audit.json';results=json.loads(cache.read_text()) if cache.exists() else {}
for p in sorted((b/'videos').rglob('*.mp4')):
 key=str(p.relative_to(b));stat=p.stat()
 if key in results and results[key]['size']==stat.st_size and results[key]['mtime_ns']==stat.st_mtime_ns:continue
 reader=imageio_ffmpeg.read_frames(str(p),output_params=['-threads','1'])
 try:
  meta=next(reader);frames=sum(1 for _ in reader)
 finally:reader.close()
 assert frames==81,(key,frames)
 assert abs(meta['fps']-16)<1e-6,(key,meta['fps'])
 results[key]={'frames':frames,'fps':meta['fps'],'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'dimensions':meta['size']}
cache.write_text(json.dumps(results,indent=2))
print(json.dumps({'validated_videos':len(results),'all_81_frames_16fps':all(v['frames']==81 and v['fps']==16 for v in results.values())}))
PY
