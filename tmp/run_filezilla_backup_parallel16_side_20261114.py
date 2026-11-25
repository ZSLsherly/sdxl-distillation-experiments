"""Run the FileZilla download on the laboratory host; no model bytes visit this PC."""
from pathlib import Path
import json,re,shlex,sys
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko

pw=sys.stdin.readline().strip()
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
match=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('172.18.33.60',username=match[1],password=match[2].strip(),timeout=20,look_for_keys=False,allow_agent=False)
client.get_transport().set_keepalive(20)
code=Path('tmp/filezilla_backup_A_terminal_parallel16_20261114.py').read_text(encoding='utf8')
command='nice -n 15 ionice -c2 -n7 python3 -u -c '+shlex.quote(code)
inp,out,_=client.exec_command(command,timeout=600);out.channel.set_combine_stderr(True)
inp.write(pw+'\n');inp.flush();inp.channel.shutdown_write()
receipt=None
try:
    for line in out:
        line=line.rstrip();print(line.replace(pw,'[redacted]'),flush=True)
        if line.startswith('BACKUP_COMPLETE '):receipt=json.loads(line[len('BACKUP_COMPLETE '):])
    status=out.channel.recv_exit_status()
    if status or receipt is None:raise RuntimeError('Backup not verified; exit='+str(status))
    dest=Path('output/sdxl-crossdistill/a800_archive_20261114_side');dest.mkdir(parents=True,exist_ok=True)
    (dest/'A_terminal_only_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    note='A去中段最终权重备份完成\n'+receipt['completed_at']+'\n'
    for k in ['group','source_server','source','destination_server','destination','bytes','sha256','tool','scope']:
        note+=k+'：'+str(receipt[k])+'\n'
    note+='源/目标SHA256一致；源文件未删除；权重未落盘本机。\n'
    (dest/'A去中段_备份记录.txt').write_text(note,encoding='utf8')
finally:client.close()
