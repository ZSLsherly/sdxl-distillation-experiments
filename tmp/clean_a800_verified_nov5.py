"""Remove only four named closed data-disk student checkpoints after independent proof."""
from backup_a800_nov5 import arms,report,ROOT,LAB,cloud,lab,run,q
from pathlib import PurePosixPath
import json,time
record=report/'cleaned.json'
done=json.loads(record.read_text()) if record.exists() else {}
with cloud() as s,lab() as d:
 while len(done)<4:
    try: proof=json.loads((report/'weights.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):proof={}
    for arm in arms[:4]:
        if arm in done or arm not in proof:continue
        r=proof[arm];origin=r['src'];dst=r['dest'];h=r['sha256'];size=5135506134
        p=PurePosixPath(origin)
        assert p.parent.parent==PurePosixPath(ROOT,'outputs',arm,'model','weights') and p.name=='student_rank0.pt'
        assert dst==LAB+'/'+p.relative_to(ROOT).as_posix()
        assert run(d,'realpath '+q(dst))==dst
        assert run(s,'realpath '+q(origin))==origin
        assert int(run(d,'stat -c %s '+q(dst)))==size and run(d,'sha256sum '+q(dst)).split()[0]==h
        # Perform source validation and unlink within the same short-lived operation.
        run(s,f'test -f {q(origin)} && test ! -L {q(origin)} && test "$(stat -c %s {q(origin)})" = {size} && test "$(sha256sum {q(origin)} | cut -d" " -f1)" = {h} && unlink {q(origin)}')
        done[arm]=r;record.write_text(json.dumps(done,indent=2))
        print('REMOVED_VERIFIED_COPY',arm,run(s,'df -h --output=avail '+q(ROOT)).splitlines()[-1],flush=True)
    if len(done)<4:time.sleep(45)

